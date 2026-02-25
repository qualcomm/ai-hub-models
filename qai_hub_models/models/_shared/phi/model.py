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

from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.phi3 import Phi3Config, modeling_phi3

from qai_hub_models.models._shared.llm.common import LLMIOType
from qai_hub_models.models._shared.llm.model import (
    Embedding,
    PositionProcessorBase,
)
from qai_hub_models.models._shared.phi.model_adaptations import (
    Phi35SHAAttention,
    QcPhi3_apply_rotary_pos_emb,
    QCPhi3ForCausalLM,
    QCPhi3MLP,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

END_TOKENS = {"<|end|>"}


@unique
class Phi3_Optimizations(str, Enum):  # Inherit from str and Enum
    SHA_ATTENTION = "sha_attention"
    RMS_NORM_4_RANK = "rank4_rms_norm"


class Phi3LongRoPEScaledRotaryEmbedding(Embedding):
    """
    Phi3's Long RoPE Scaled Rotary Embeddings implementation.
    This is the exact implementation used in Phi-3.5-mini-instruct.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        max_length: int = 2048,
        config: Phi3Config | None = None,
    ) -> None:
        if config is None:
            config = Phi3Config()
        self.config = config
        if head_dim is None:
            assert getattr(config, "hidden_size", None) and getattr(
                config, "num_attention_heads", None
            ), "Could not determine head_dim"
            head_dim = config.hidden_size // config.num_attention_heads
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.dim = int(head_dim * partial_rotary_factor)

        self.max_length = max_length
        self.max_position_embeddings = getattr(
            config, "max_position_embeddings", 131072
        )
        self.original_max_position_embeddings = getattr(
            config, "original_max_position_embeddings", 4096
        )
        self.rope_theta = getattr(config, "rope_theta", 10000.0)

        # Get rope_scaling parameters
        rope_scaling = getattr(config, "rope_scaling", {}) or {}
        self.rope_type = rope_scaling.get("type", "longrope")
        self.short_factor = rope_scaling.get("short_factor", [1.0] * (self.dim // 2))
        self.long_factor = rope_scaling.get("long_factor", [1.0] * (self.dim // 2))

        # Precompute for the given max_length
        self.cos, self.sin = self.precompute(max_length)

    def precompute(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute cos and sin values for rotary embeddings."""
        if not hasattr(modeling_phi3, "Phi3LongRoPEScaledRotaryEmbedding"):
            rope = modeling_phi3.Phi3RotaryEmbedding(config=self.config)
        else:
            rope = modeling_phi3.Phi3LongRoPEScaledRotaryEmbedding(
                dim=self.dim, config=self.config
            )
        dummy_x = torch.Tensor([1.0])
        position_ids = torch.arange(seq_len).view(1, -1)
        if hasattr(rope, "_original_forward"):
            embeddings = rope._original_forward(dummy_x, position_ids)  # type: ignore[operator, unused-ignore]
        else:
            embeddings = rope.forward(dummy_x, position_ids)

        emb_size = embeddings[0].size(-1) // 2
        embeddings = [emb[:, :, :emb_size] for emb in embeddings]
        return tuple(emb.unsqueeze(0) for emb in embeddings)

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        seq_len: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin embeddings for given position_ids.

        Parameters
        ----------
        position_ids
            [batch_size, sequence_length]
        seq_len
            Optional sequence length for dynamic computation.
        dtype
            Output dtype.

        Returns
        -------
        cos : torch.Tensor
            Cosine embeddings with shape [batch_size, 1, sequence_length, head_dim].
        sin : torch.Tensor
            Sine embeddings with shape [batch_size, 1, sequence_length, head_dim].
        """
        cos = self.cos[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        sin = self.sin[0, 0, :, :].to(position_ids.device)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class Phi3PositionProcessor(PositionProcessorBase):
    """Prepares positions (RopeEmbedding and attention mask preparation); used by ORT GenAI."""

    def __init__(
        self,
        context_length: int,
        config: Phi3Config,
    ) -> None:
        super().__init__(context_length, config=config)
        self.context_len = context_length
        self.rope_embedding = Phi3LongRoPEScaledRotaryEmbedding(
            max_length=self.context_len,
            config=config,
        )

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


class Phi35Base(LLMBase):
    LMClass = QCPhi3ForCausalLM
    EmbeddingClass = Phi3LongRoPEScaledRotaryEmbedding

    # Default prompts for demos
    default_user_prompt = "What is the capital of France? Answer in one sentence."
    default_system_prompt = "You are a helpful AI assistant."

    @staticmethod
    def eval_datasets() -> list[str]:
        return [*LLMBase.eval_datasets(), "tricky_llm_prompts_phi35"]

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        if (
            skip_optimizations
            and Phi3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        elif hasattr(modeling_phi3, "PHI3_ATTENTION_CLASSES"):
            modeling_phi3.PHI3_ATTENTION_CLASSES["eager"] = Phi35SHAAttention
        else:
            modeling_phi3.Phi3Attention = Phi35SHAAttention  # type: ignore[misc, unused-ignore]

        def bypass_RotaryEmbedding(
            self: modeling_phi3.Phi3RotaryEmbedding,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        # Bypass rotary_emb module
        if hasattr(modeling_phi3, "Phi3RotaryEmbedding"):
            if not hasattr(modeling_phi3.Phi3RotaryEmbedding, "_original_forward"):
                modeling_phi3.Phi3RotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore [reportAttributeAccessIssue]
                    modeling_phi3.Phi3RotaryEmbedding.forward
                )
                modeling_phi3.Phi3RotaryEmbedding.forward = bypass_RotaryEmbedding
        elif not hasattr(
            modeling_phi3.Phi3LongRoPEScaledRotaryEmbedding,  # type: ignore[attr-defined, unused-ignore]
            "_original_forward",
        ):
            modeling_phi3.Phi3LongRoPEScaledRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore [reportAttributeAccessIssue]
                modeling_phi3.Phi3LongRoPEScaledRotaryEmbedding.forward  # type: ignore[attr-defined, unused-ignore]
            )
            modeling_phi3.Phi3LongRoPEScaledRotaryEmbedding.forward = (  # type: ignore[attr-defined, unused-ignore]
                bypass_RotaryEmbedding
            )
        modeling_phi3.apply_rotary_pos_emb = QcPhi3_apply_rotary_pos_emb

        def Phi3RMSNorm_forward(
            self: modeling_phi3.Phi3RMSNorm, hidden_states: torch.Tensor
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
            and Phi3_Optimizations.RMS_NORM_4_RANK in skip_optimizations
        ):
            print("Skip rank4_rms_norm optimization")
        else:
            modeling_phi3.Phi3RMSNorm.forward = Phi3RMSNorm_forward

        modeling_phi3.Phi3MLP = QCPhi3MLP  # type: ignore[misc, unused-ignore]
        modeling_phi3.Phi3ForCausalLM = QCPhi3ForCausalLM  # type: ignore[misc, unused-ignore]

    def _verify_ckpt(self) -> None:
        if (
            not (
                self.llm_config.architectures[0] == "Phi3ForCausalLM"  # type: ignore[index, unused-ignore]
                and self.llm_config.model_type == "phi3"
            )
            and self.llm_config.rope_scaling is not None
            and self.llm_config.rope_scaling["rope_type"] != "longrope"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )


class Phi35Base_AIMETOnnx(LLM_AIMETOnnx):
    EmbeddingClass = Phi3LongRoPEScaledRotaryEmbedding
    FPModel = Phi35Base

    ada_scale_model_type: str | None = "phi3"

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

    eval_datasets = Phi35Base.eval_datasets

    @classmethod
    def _configure_quant_sim(
        cls, quant_sim: QuantizationSimModel, precision: Precision
    ) -> QuantizationSimModel:
        quant_sim = LLM_AIMETOnnx._configure_quant_sim(quant_sim, precision)

        if precision == Precision.w4a16:
            from qai_hub_models.models._shared.llm._utils import (
                _set_4bit_weights_to_lpbq,
            )

            _set_4bit_weights_to_lpbq(quant_sim)
        return quant_sim

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
        zero_keys = []

        for layer in range(self.llm_config.num_hidden_layers):
            for sec in ["input", "post_attention"]:
                zero_keys += [
                    f"/model/model/layers.{layer}/{sec}_layernorm/Pow_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/ReduceMean_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Add_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Sqrt_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Div_output_0",
                    f"/model/model/layers.{layer}/{sec}_layernorm/Mul_output_0",
                ]

        zero_keys += [
            "/model/model/norm/Pow_output_0",
            "/model/model/norm/ReduceMean_output_0",
            "/model/model/norm/Add_output_0",
            "/model/model/norm/Sqrt_output_0",
            "/model/model/norm/Div_output_0",
            "/model/model/norm/Mul_output_0",
        ]

        for key in zero_keys:
            if uses_lists:
                # aimet format 1.0
                zero_entry: Any = {
                    "bw": 16,
                    "dtype": "INT",
                    "enc_type": "PER_TENSOR",
                    "is_sym": False,
                    "name": key,
                    "offset": [0],
                    "scale": [1e-20],
                }
            else:
                # aimet format 0.x
                zero_entry = [
                    {
                        "bitwidth": 16,
                        "dtype": "int",
                        "is_symmetric": "False",
                        "max": 0.0,
                        "min": 0.0,
                        "offset": 0,
                        "scale": 1e-20,
                    }
                ]
            encodings["activation_encodings"][key] = zero_entry
        propagate_memory_encodings(encodings, model)

        if uses_lists:
            # convert back
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)


class Phi35Base_QNN(LLM_QNN):
    FPModel = Phi35Base
    EmbeddingClass = Phi3LongRoPEScaledRotaryEmbedding
    num_layers_per_split: int
