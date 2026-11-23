# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import cast

import numpy as np
import torch
from neucodec import NeuCodec
from torch import Tensor, nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.qwen3.model import (
    QWEN3_FP_ATTENTION_MASK_MIN_CLIP,
    Qwen3Base,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

MODEL_ID = __name__.split(".")[-2]

BACKBONE_REPO = "neuphonic/neutts-2e"
CODEC_REPO = "neuphonic/neucodec"

# Checked against the checkpoint config in Backbone._verify_ckpt.
NUM_LAYERS = 28
HIDDEN_SIZE = 512
NUM_ATTN_HEADS = 12
NUM_KEY_VALUE_HEADS = 4
# Explicit in Qwen3; not hidden_size // num_attn_heads.
HEAD_DIM = 128

# Upstream caps generation at 2048 despite max_position_embeddings=32768.
CONTEXT_LENGTH = 2048
PREFILL_SEQ_LEN = 128
DECODE_SEQ_LEN = 1

SAMPLE_RATE = 24_000

SAMPLE_PROMPT = "My name is Jo and I am a text to speech model."


def graph_name(sequence_length: int) -> str:
    kind = "token" if sequence_length == 1 else "prompt"
    return f"{kind}_ar{sequence_length}_cl{CONTEXT_LENGTH}"


def build_attention_mask(query_len: int, num_valid_keys: int) -> Tensor:
    """4D additive causal mask over a right-aligned ``CONTEXT_LENGTH`` key window."""
    mask_2d = torch.zeros((1, CONTEXT_LENGTH))
    mask_2d[:, -num_valid_keys:] = 1.0
    return (
        AttentionMaskConverter(True)
        .to_4d(
            mask_2d,
            query_length=query_len,
            key_value_length=CONTEXT_LENGTH,
            dtype=torch.float32,
        )
        .clip(QWEN3_FP_ATTENTION_MASK_MIN_CLIP, 0)
    )


def empty_kv_cache(kv_seq_len: int) -> list[Tensor]:
    """Zeroed per-layer key/value cache tensors in graph input order."""
    caches: list[Tensor] = []
    for _ in range(NUM_LAYERS):
        caches.append(torch.zeros(NUM_KEY_VALUE_HEADS, 1, HEAD_DIM, kv_seq_len))
        caches.append(torch.zeros(NUM_KEY_VALUE_HEADS, 1, kv_seq_len, HEAD_DIM))
    return caches


@lru_cache(maxsize=1)
def load_codec() -> nn.Module:
    # Variable-length speech tokenizer; runs on CPU outside the compiled graphs.
    codec = cast(nn.Module, NeuCodec.from_pretrained(CODEC_REPO))
    codec.to("cpu")
    codec.eval()
    return codec


class Backbone(Qwen3Base):
    """NeuTTS-2E Qwen3 causal LM with the KV cache exposed as graph I/O."""

    min_memory_recommended = 0

    def __init__(
        self,
        sequence_length: int = PREFILL_SEQ_LEN,
        context_length: int = CONTEXT_LENGTH,
    ) -> None:
        if (sequence_length, context_length) != (PREFILL_SEQ_LEN, CONTEXT_LENGTH):
            raise ValueError(
                f"NeuTTS-2E only supports sequence_length={PREFILL_SEQ_LEN} and "
                f"context_length={CONTEXT_LENGTH}; got ({sequence_length}, "
                f"{context_length})."
            )
        super().__init__(
            checkpoint=BACKBONE_REPO,
            sequence_length=sequence_length,
            context_length=context_length,
        )

    def _verify_ckpt(self) -> None:
        """Reject a checkpoint whose shape does not match the exported graphs."""
        super()._verify_ckpt()
        declared = {
            "num_hidden_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "num_attention_heads": NUM_ATTN_HEADS,
            "num_key_value_heads": NUM_KEY_VALUE_HEADS,
            "head_dim": HEAD_DIM,
        }
        mismatched = {
            name: (getattr(self.llm_config, name, None), expected)
            for name, expected in declared.items()
            if getattr(self.llm_config, name, None) != expected
        }
        if mismatched:
            raise ValueError(
                f"{BACKBONE_REPO} config does not match the architecture constants "
                f"in {__name__} (found, expected): {mismatched}"
            )

    @classmethod
    def from_pretrained(
        cls,
        sequence_length: int = PREFILL_SEQ_LEN,
        context_length: int = CONTEXT_LENGTH,
    ) -> Self:
        return cls(sequence_length, context_length)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = PREFILL_SEQ_LEN,
        context_length: int = CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        if context_length != CONTEXT_LENGTH or llm_io_type != LLMIOType.genie_input_ids:
            raise ValueError(
                f"NeuTTS-2E only supports context_length={CONTEXT_LENGTH} and "
                f"llm_io_type={LLMIOType.genie_input_ids}; got ({context_length}, "
                f"{llm_io_type})."
            )
        return self._get_input_spec(
            num_hidden_layers=NUM_LAYERS,
            sequence_length=sequence_length,
            context_length=CONTEXT_LENGTH,
            hidden_size=HIDDEN_SIZE,
            num_key_value_heads=NUM_KEY_VALUE_HEADS,
            num_attention_heads=NUM_ATTN_HEADS,
            head_dim=HEAD_DIM,
            llm_io_type=llm_io_type,
        )

    def get_output_spec(self) -> OutputSpec:
        return self._get_output_spec(NUM_LAYERS)

    def sample_graph_inputs(self, sequence_length: int) -> SampleInputsType:
        """Inputs for one graph, tiled so no position is masked padding.

        Masked positions would still be scored in on-device comparisons.
        """
        ids = self.tokenizer(SAMPLE_PROMPT, return_tensors="pt")["input_ids"]
        repeats = -(-sequence_length // int(ids.shape[1]))
        input_ids = ids.repeat(1, repeats)[:, :sequence_length].to(torch.int32)

        position_ids = torch.arange(sequence_length, dtype=torch.long).reshape(1, -1)
        cos, sin = self.embedding.get_embedding(position_ids)

        inputs: SampleInputsType = {
            "input_ids": [input_ids.numpy()],
            "attention_mask": [
                build_attention_mask(sequence_length, sequence_length).numpy()
            ],
            "position_ids_cos": [cos.numpy()],
            "position_ids_sin": [sin.numpy()],
        }
        caches = empty_kv_cache(CONTEXT_LENGTH - sequence_length)
        for layer in range(NUM_LAYERS):
            inputs[f"past_key_{layer}_in"] = [caches[2 * layer].numpy()]
            inputs[f"past_value_{layer}_in"] = [caches[2 * layer + 1].numpy()]
        return inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return self.sample_graph_inputs(self.sequence_length)


class NeuTTS2E(MultiGraphWorkbenchModel):
    """NeuTTS-2E backbone as prefill and decode graphs of one weight-shared binary."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone
        self._graph_sequence_lengths = {
            graph_name(seq_len): seq_len
            for seq_len in (PREFILL_SEQ_LEN, DECODE_SEQ_LEN)
        }

    @property
    def graph_names(self) -> list[str]:
        return list(self._graph_sequence_lengths)

    def get_graph_input_spec(self, graph_name: str) -> InputSpec:
        return self.backbone.get_input_spec(
            sequence_length=self._graph_sequence_lengths[graph_name]
        )

    def get_graph_output_spec(self, graph_name: str) -> OutputSpec:
        return self.backbone.get_output_spec()

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        return self.backbone.sample_graph_inputs(
            self._graph_sequence_lengths[graph_name]
        )

    def local_inference(
        self, graph_name: str, sample_inputs: SampleInputsType
    ) -> list[np.ndarray]:
        inputs = [torch.from_numpy(value[0]) for value in sample_inputs.values()]
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(*inputs)
        return [out.numpy() for out in outputs]

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Export the backbone with ``torch.export`` at this graph's shape.

        Parameters
        ----------
        graph_name
            Graph to export; selects the sequence length.
        output_dir
            Directory to write the exported program into.
        input_spec
            Unused -- shapes come from the graph's sequence length.

        Returns
        -------
        Path
            Path to the serialized ``.pt2`` program.
        """
        inputs = tuple(
            torch.from_numpy(value[0])
            for value in self.backbone.sample_graph_inputs(
                self._graph_sequence_lengths[graph_name]
            ).values()
        )
        self.backbone.eval()
        output_path = Path(output_dir) / f"{self.name}_{graph_name}.pt2"
        with torch.no_grad():
            exported = torch.export.export(self.backbone, inputs)
        torch.export.save(exported, output_path)
        return output_path

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(Backbone.from_pretrained())
