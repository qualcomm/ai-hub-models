# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    PositionProcessorBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import json
import os
from collections.abc import Collection
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.qwen3 import modeling_qwen3

from qai_hub_models.models._shared.llama3.model import RopeEmbedding
from qai_hub_models.models._shared.qwen3.model_adaptations import (
    QcQwen3_apply_rotary_pos_emb,
    QCQwen3ForCausalLM,
    QCQwen3MLP,
    SHAQwen3Attention,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.base_model import Precision

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_qwen"

DATA_DIR = "data"
USE_CACHED_DATA = True

# Qwen3 uses the same ChatML format as Qwen2
START_HEADER = "<|im_start|>"
END_HEADER = "<|im_end|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
END_TOKENS = {"<|im_end|>", "<|endoftext|>"}


class Qwen3_Optimizations(str, Enum):  # Inherit from str and Enum
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class Qwen3Base(LLMBase):
    LMClass = modeling_qwen3.Qwen3ForCausalLM
    EmbeddingClass = RopeEmbedding

    # Default prompts for demos
    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant"

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Qwen3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            # In transformers 4.51.0+, Qwen3 directly instantiates Qwen3Attention
            # instead of using an ATTENTION_CLASSES dict
            modeling_qwen3.Qwen3Attention = SHAQwen3Attention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_qwen3.Qwen3RotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(modeling_qwen3.Qwen3RotaryEmbedding, "_original_forward"):
            modeling_qwen3.Qwen3RotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore [reportAttributeAccessIssue]
                modeling_qwen3.Qwen3RotaryEmbedding.forward
            )
            modeling_qwen3.Qwen3RotaryEmbedding.forward = bypass_RotaryEmbedding
        modeling_qwen3.apply_rotary_pos_emb = QcQwen3_apply_rotary_pos_emb

        modeling_qwen3.Qwen3MLP = QCQwen3MLP  # type: ignore[misc, unused-ignore]
        modeling_qwen3.Qwen3ForCausalLM = QCQwen3ForCausalLM  # type: ignore[misc, unused-ignore]

    def _verify_ckpt(self) -> None:
        if not (
            self.llm_config.architectures[0] == "Qwen3ForCausalLM"  # type: ignore[index, unused-ignore]
            and self.llm_config.model_type == "qwen3"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> list[torch.Tensor]:
        return super().forward(
            input_tokens,
            self.attention_mask_multiplier * attention_mask,
            *rest,
        )


class Qwen3PositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: PretrainedConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len, config=config)  # type: ignore[arg-type, unused-ignore]

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        position_ids_cos, position_ids_sin = self.rope_embedding.get_embedding(
            position_ids
        )
        attention_mask_converter = AttentionMaskConverter(True)
        attention_mask = attention_mask_converter.to_4d(
            attention_mask_before_processor,
            query_length=position_ids.shape[1],
            key_value_length=attention_mask_before_processor.shape[1],
            dtype=torch.float32,
        )
        attention_mask = attention_mask.clip(-50, 0)
        return attention_mask, position_ids_cos, position_ids_sin


class Qwen3Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = RopeEmbedding
    FPModel = Qwen3Base

    ada_scale_model_type: str | None = "qwen3"

    def __init__(
        self,
        quant_sim: QuantizationSimModel,
        host_device: torch.device,
        checkpoint: str | os.PathLike | Path | None = None,
        tokenizer: PreTrainedTokenizer | None = None,
        llm_config: PretrainedConfig | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            quant_sim=quant_sim,
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

    @classmethod
    def prepare_genie_assets(
        cls,
        hub_device: hub.Device,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        context_length: int,
        model_list: list[str],
        output_path: Path,
        precision: Precision,
        encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
    ) -> None:
        from transformers import AutoTokenizer

        super().prepare_genie_assets(
            hub_device,
            checkpoint,
            llm_config,
            context_length,
            model_list,
            output_path,
            precision,
            encodings_path,
            input_specs,
            output_specs,
        )

        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        sample_prompt = cls.get_input_prompt_with_tags(tokenizer=tokenizer)
        with open(output_path / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *rest: torch.Tensor,
    ) -> torch.Tensor | Collection[torch.Tensor]:
        return super().forward(
            input_tokens,
            self.attention_mask_multiplier * attention_mask,
            *rest,
        )

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Make sure AIMET encodings are ready for ONNX split."""
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path)

        model_input_names = {}
        for node in model.graph.node:
            model_input_names[node.name] = node.input

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        assert uses_lists

        # Convert encodings to dictionaries for faster look-ups
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings["activation_encodings"]
        }
        encodings["param_encodings"] = {
            v["name"]: v for v in encodings["param_encodings"]
        }

        # See _shared/llama3/model.py for why this is needed.
        embed_a_name = "/model/model/embed_tokens/Gather_output_0"
        embed_w_name = "model.model.embed_tokens.weight"
        encodings["activation_encodings"][embed_a_name] = copy.deepcopy(
            encodings["activation_encodings"][embed_w_name]
        )
        for key in encodings["activation_encodings"]:
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        # convert back
        encodings["activation_encodings"] = list(
            encodings["activation_encodings"].values()
        )
        encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


class Qwen3Base_QNN(LLM_QNN):
    FPModel = Qwen3Base
    EmbeddingClass = RopeEmbedding
    num_layers_per_split: int

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names
