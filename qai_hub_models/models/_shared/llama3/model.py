# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models._shared.llm.model import (
    LLMBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

# isort: on
import copy
import json
import os
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import torch

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
import transformers
from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama import LlamaConfig, modeling_llama

from qai_hub_models.models._shared.llama3.model_adaptations import (
    QcLlama_apply_rotary_pos_emb,
    QCLlamaForCausalLM,
    QCLlamaMLP,
    SHALlamaAttention,
)
from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    Embedding,
    PositionProcessorBase,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Configs
AIMET_ENCODINGS_PREFIX = "config"
AIMET_CONFIG = "default_config_llama"

DATA_DIR = "data"
USE_CACHED_DATA = True

## Ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1
BEGIN_TEXT = "<|begin_of_text|>"
END_TEXT = "<|end_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
SYSTEM_ID = "system"
ASSISTANT_ID = "assistant"
USER_ID = "user"
EOT_ID = "<|eot_id|>"
END_TOKENS = {"<|eot_id|>", "<|end_of_text|>"}


@unique
class Llama3_Optimizations(str, Enum):  # Inherit from str and Enum
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class RopeEmbedding(Embedding):
    def __init__(
        self,
        head_dim: int | None = None,
        max_length: int = 2048,
        config: LlamaConfig | None = None,
    ) -> None:
        if config is None:
            config = LlamaConfig()
        head_dim = head_dim or (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        self.cos, self.sin = self.precompute(head_dim, max_length, config)

    def precompute(
        self, head_dim: int, max_length: int, config: LlamaConfig
    ) -> list[torch.Tensor]:
        kwargs: dict[str, Any] = {
            "config": config,
        }
        if Version(transformers.__version__) < Version("4.48"):
            kwargs |= {
                "max_position_embeddings": config.max_position_embeddings,
                "base": config.rope_theta,
                "dim": head_dim,
            }

        if not hasattr(config, "rope_scaling"):
            config.rope_scaling = None

        rope = modeling_llama.LlamaRotaryEmbedding(**kwargs)
        dummy_x = torch.tensor([1.0])
        position_ids = torch.arange(max_length).view(1, -1)
        if hasattr(rope, "_original_forward") and callable(rope._original_forward):
            embeddings = rope._original_forward(dummy_x, position_ids)
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        # for adapted llama
        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        return [emb.unsqueeze(0) for emb in embeddings]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class LlamaPositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: LlamaConfig,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = RopeEmbedding(max_length=self.context_len, config=config)

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


class Llama3Base(LLMBase):
    LMClass = QCLlamaForCausalLM
    EmbeddingClass = RopeEmbedding

    # Default prompts for demos
    default_user_prompt = "What do llamas eat? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant"

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Llama3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        elif hasattr(modeling_llama, "LLAMA_ATTENTION_CLASSES"):
            modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = SHALlamaAttention
        else:
            modeling_llama.LlamaAttention = SHALlamaAttention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_llama.LlamaRotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if not hasattr(modeling_llama.LlamaRotaryEmbedding, "_original_forward"):
            modeling_llama.LlamaRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_llama.LlamaRotaryEmbedding.forward
            )
            modeling_llama.LlamaRotaryEmbedding.forward = bypass_RotaryEmbedding
        modeling_llama.apply_rotary_pos_emb = QcLlama_apply_rotary_pos_emb

        def LlamaRMSNorm_forward(
            self: modeling_llama.LlamaRMSNorm, hidden_states: torch.Tensor
        ) -> torch.Tensor:
            # Raise to rank 4
            hidden_states = hidden_states.unsqueeze(0)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return (hidden_states * self.weight).squeeze(0)

        if (
            skip_optimizations
            and Llama3_Optimizations.RMS_NORM_4_RANK in skip_optimizations
        ):
            print("Skip rank4_rms_norm optimization")
        else:
            modeling_llama.LlamaRMSNorm.forward = LlamaRMSNorm_forward

        modeling_llama.LlamaMLP = QCLlamaMLP  # type: ignore[misc, unused-ignore]
        modeling_llama.LlamaForCausalLM = QCLlamaForCausalLM  # type: ignore[misc, unused-ignore]

    def _verify_ckpt(self) -> None:
        if (
            not (
                self.llm_config.architectures
                and self.llm_config.architectures[0] == "LlamaForCausalLM"
                and self.llm_config.model_type == "llama"
            )
            and self.llm_config.rope_scaling is not None
            and self.llm_config.rope_scaling["rope_type"] != "llama3"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )


class Llama3Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = RopeEmbedding
    FPModel = Llama3Base

    ada_scale_model_type: str | None = "llama"

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

    @staticmethod
    def _get_output_names(num_hidden_layers: int) -> list[str]:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return output_names

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

        if uses_lists:
            # Convert encodings to dictionaries for faster look-ups
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        if self.llm_io_type in {
            LLMIOType.genie_input_ids,
            LLMIOType.huggingface_input_ids,
        }:
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

        if uses_lists and self.llm_io_type in {
            LLMIOType.genie_input_ids,
            LLMIOType.huggingface_input_ids,
        }:
            encodings["activation_encodings"][embed_a_name]["name"] = embed_a_name

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            # convert back
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


class Llama3Base_QNN(LLM_QNN):
    FPModel = Llama3Base
    EmbeddingClass = RopeEmbedding
    num_layers_per_split: int
