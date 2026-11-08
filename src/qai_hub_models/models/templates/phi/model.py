# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.utils.input_spec import OutputSpec
from qai_hub_models.models.templates.llm.model import (
    LLMBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    DynamicSplitCollectionBase,
    DynamicSplitPartBase,
    LLMDynamicBase,
    LLMDynamic_AIMETOnnx,
    SingleSlotCacheMixin,
    get_onnx_model,
)

# isort: on
import copy
import json
import logging
import os
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import onnx
import torch
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

from packaging.version import Version
from transformers import PretrainedConfig, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.phi3 import Phi3Config, modeling_phi3

from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.model import (
    Embedding,
    PositionProcessorBase,
)
from qai_hub_models.models.templates.phi.model_adaptations import (
    Phi35SHAAttention,
    QcPhi3_apply_rotary_pos_emb,
    QCPhi3ForCausalLM,
    QCPhi3MLP,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

logger = logging.getLogger(__name__)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Shared by the whole Phi family: Phi-3.5 and Phi-4 are architecturally identical
# (both are `Phi3ForCausalLM` with longrope scaling), so they share these bases and
# the upstream `modeling_phi3` names. Class names keep the Phi3/Phi35 prefixes to
# stay aligned with the transformers module being patched -- they are not specific
# to Phi-3.5.
#
# Only the turn-end token (<|end|>, id=200020) is used as the EOS stop
# signal.  <|endoftext|> (id=199999) is pre-filled in every prompt file via
# genie_generation_prompt_suffix so it is never emitted as a generated token,
# and therefore does not need to be in the stop-token list.
END_TOKENS = {"<|end|>"}

# Phi-3.5 and Phi-4 both apply rotary embeddings to 75% of each head's dimensions,
# so position_ids_{cos,sin} carry int(head_dim * 0.75) // 2 entries rather than
# head_dim // 2. Concrete models override this if their variant differs.
DEFAULT_PARTIAL_ROTARY_FACTOR = 0.75


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

    def get_embedding(  # type: ignore[override]
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

    def forward(  # type: ignore[override]
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

    # No get_eval_dataset_classes override here: TrickyLLMPromptsPhi35 is an
    # abstract base (see dataset.py), so each concrete model registers its own
    # subclass with the right tokenizer and prompt format.

    # Jinja2 chat template used by the HuggingFace tokenizer and written into
    # tokenizer_config.json for on-device inference.  Supports an optional
    # "tools" field on system messages (function-calling); all other messages
    # use the standard <|role|>content<|end|> format.
    CHAT_TEMPLATE: str = (
        "{%- for message in messages -%}"
        '\t{%- if message["role"] == "system" and "tools" in message and message["tools"] is not none -%}'
        '\t\t{{- "<|" + message["role"] + "|>" + message["content"] + "<|tool|>" + message["tools"] + "<|/tool|>" + "<|end|>" -}}'
        "\t{%- else -%}"
        '\t\t{{- "<|" + message["role"] + "|>" + message["content"] + "<|end|>" -}}'
        "\t{%- endif -%}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        '\t{{- "<|assistant|>" -}}'
        "{%- endif -%}"
    )

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        """Return the Phi-3/4 chat template tokens for Genie / GenieX."""
        return {
            "system_prefix": "<|system|>",
            "system_suffix": "<|end|>",
            "user_prefix": "<|user|>",
            "user_suffix": "<|end|>",
            "assistant_prefix": "<|assistant|>",
            "assistant_suffix": "<|end|>",
            "default_system_prompt": cls.default_system_prompt,
        }

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
        modeling_phi3.apply_rotary_pos_emb = QcPhi3_apply_rotary_pos_emb  # type: ignore[assignment, unused-ignore]

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
        # The `not` must wrap every condition: previously it covered only the
        # architecture/model_type pair, so a checkpoint with the right
        # architecture but the wrong rope_scaling slipped through. The key is
        # `type` (not `rope_type`) and longrope is what we require, not what we
        # reject.
        if not (
            self.llm_config.architectures[0] == "Phi3ForCausalLM"  # type: ignore[index, unused-ignore]
            and self.llm_config.model_type == "phi3"
            and self.llm_config.rope_scaling is not None
            and self.llm_config.rope_scaling.get("type") == "longrope"
        ):
            raise ValueError(
                "Model config is not compatible with this model implementation. "
                "Expected a Phi3ForCausalLM architecture with phi3 model_type and "
                "longrope rope_scaling."
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

    # Load-bearing: LLM_AIMETOnnx does not inherit LLMBase, so without this alias
    # the quantized class has no get_eval_dataset_classes at all. It now resolves to
    # LLMBase's base list (Phi35Base no longer overrides it); concrete models add
    # their own variant-specific datasets.
    get_eval_dataset_classes = Phi35Base.get_eval_dataset_classes

    @staticmethod
    def _get_output_spec(num_hidden_layers: int) -> OutputSpec:
        output_names = ["logits"]
        for layer in range(num_hidden_layers):
            output_names.append(f"past_key_{layer}_out")
            output_names.append(f"past_value_{layer}_out")
        return {x: TensorSpec() for x in output_names}

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
            # See templates/llama3/model.py for why this is needed.
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


class PhiDynamicBase(LLMDynamicBase, Phi35Base):
    """Phi-family FP base with dynamic-shape ONNX export."""

    def get_full_onnx_bundle(self, temp_path: Path) -> Any:
        from qai_hub_models.utils.onnx.helpers import ONNXBundle

        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=self.context_length,
            sequence_length=self.sequence_length,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            quiet=True,
        )
        return ONNXBundle.from_bundle_path(onnx_dir, "model")


class PhiDynamicBase_AIMETOnnx(LLMDynamic_AIMETOnnx, Phi35Base_AIMETOnnx):
    """Dynamic-shape variant of Phi35Base_AIMETOnnx."""

    FPModel = PhiDynamicBase  # type: ignore[assignment]

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        with open(src_encodings_path) as f:
            encodings = json.load(f)

        model = onnx.load(onnx_model_path, load_external_data=False)

        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        if uses_lists:
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

        gather_node = next((n for n in model.graph.node if n.op_type == "Gather"), None)
        if gather_node is not None:
            gather_output = gather_node.output[0]
            embed_weight_name = gather_node.input[0]

            if self.precision == Precision.w4:
                encodings["activation_encodings"][gather_output] = {
                    "bw": 16,
                    "dtype": "FLOAT",
                    "enc_type": "PER_TENSOR",
                    "name": gather_output,
                }
            else:
                weight_enc = encodings["activation_encodings"].get(embed_weight_name)
                if weight_enc is not None:
                    embedding_enc = copy.deepcopy(weight_enc)
                    embedding_enc["name"] = gather_output
                    encodings["activation_encodings"][gather_output] = embedding_enc

        for key, value in list(encodings["activation_encodings"].items()):
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(value)

        propagate_memory_encodings(encodings, model)

        if uses_lists:
            encodings["activation_encodings"] = list(
                encodings["activation_encodings"].values()
            )
            encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as f:
            json.dump(encodings, f, indent=4, sort_keys=True)


# ---------------------------------------------------------------------------
# Phi-4-mini PreSplit / Part / Collection bases
# ---------------------------------------------------------------------------


def _fix_rope_embed_dim_input_spec(input_spec: InputSpec, embed_dim: int) -> InputSpec:
    fixed: InputSpec = dict(input_spec)
    for name, spec in input_spec.items():
        if name in ("position_ids_cos", "position_ids_sin"):
            fixed[name] = TensorSpec(
                shape=(*spec.shape[:-1], embed_dim), dtype=spec.dtype
            )
    return fixed


class PhiPreSplitBase(SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, PhiDynamicBase):
    """FP PreSplit base for Phi-family models."""

    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    hf_repo_name: str = ""
    partial_rotary_factor: float = 1.0

    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    min_memory_recommended: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    default_precision: Precision = Precision.w4a16

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint or self.hf_repo_name, **kwargs)

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        if not (
            self.llm_config.num_hidden_layers == self.num_layers
            and self.llm_config.hidden_size == self.hidden_size
            and self.llm_config.num_attention_heads == self.num_attention_heads
            and self.llm_config.num_key_value_heads == self.num_key_value_heads
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: str | os.PathLike | Path | None = None,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        # Always skip the rank-4 RMSNorm patch (mtorabza, 07b14e0733). It wraps every
        # RMSNorm in Unsqueeze/Squeeze, which breaks AIMET's decoder-block boundary
        # detection: AdaScale then either fails to extract a bounded subgraph
        # ("graph inputs are required but not provided: input_ids") or OOMs. The
        # breakage is a property of the rank-4 rewrite, not of one variant, so it is
        # skipped for the whole Phi family here.
        if _skip_optimizations is None:
            _skip_optimizations = [Phi3_Optimizations.RMS_NORM_4_RANK]
        elif Phi3_Optimizations.RMS_NORM_4_RANK not in _skip_optimizations:
            _skip_optimizations = [
                *_skip_optimizations,
                Phi3_Optimizations.RMS_NORM_4_RANK,
            ]
        checkpoint = checkpoint or cls.hf_repo_name
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached
        instance = cls(
            checkpoint=checkpoint,
            host_device=host_device,
            load_pretrained=True,
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_spec(self) -> OutputSpec:
        return PhiDynamicBase._get_output_spec(self.num_layers)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return self._static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type
        )

    @classmethod
    def _static_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": cls.num_layers,
                "hidden_size": cls.hidden_size,
                "num_key_value_heads": cls.num_key_value_heads,
                "num_attention_heads": cls.num_attention_heads,
            }
        hidden_size = llm_config.get("hidden_size", cls.hidden_size)
        num_attention_heads = llm_config.get(
            "num_attention_heads", cls.num_attention_heads
        )
        partial_rotary_factor = llm_config.get(
            "partial_rotary_factor", cls.partial_rotary_factor
        )
        input_spec = PhiDynamicBase._get_input_spec(
            num_hidden_layers=llm_config.get("num_hidden_layers", cls.num_layers),
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=hidden_size,
            num_key_value_heads=llm_config.get(
                "num_key_value_heads", cls.num_key_value_heads
            ),
            num_attention_heads=num_attention_heads,
            llm_io_type=llm_io_type,
        )
        head_dim = hidden_size // num_attention_heads
        embed_dim = int(head_dim * partial_rotary_factor) // 2
        return _fix_rope_embed_dim_input_spec(input_spec, embed_dim)


PhiPreSplitT = TypeVar("PhiPreSplitT", bound=PhiPreSplitBase)


class PhiQuantizablePreSplitBase(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin[PhiPreSplitT],
    PhiDynamicBase_AIMETOnnx,
    Generic[PhiPreSplitT],
):
    """Quantizable PreSplit base for Phi-family models.

    Uses the base DynamicQuantizablePreSplitMixin.resolve_default_checkpoint,
    which downloads the full .zip asset (dynamic ONNX + encodings + weights +
    tokenizer + config), as Qwen3 does.

    The full zip is required, not the encodings-only strategy Llama uses: this
    model's recipe applies SpinQuant R1 and AdaScale, which *rewrite the weights*.
    Re-exporting ONNX from the pristine HuggingFace weights and then loading these
    encodings would pair encodings computed on rotated/scaled weights with
    unrotated ones. The published asset therefore ships model.data alongside
    model_dynamic.onnx.
    """

    FPModel: type[PhiPreSplitT]

    num_layers: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    def get_output_spec(self) -> OutputSpec:
        return PhiDynamicBase._get_output_spec(self.num_layers)

    def _postprocess_full_onnx_bundle(self, bundle: Any) -> Any:
        if bundle.aimet_encodings_path is not None:
            self._adapt_aimet_encodings(
                str(bundle.aimet_encodings_path),
                str(bundle.aimet_encodings_path),
                str(bundle.onnx_graph_path),
            )
        return super()._postprocess_full_onnx_bundle(bundle)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        return self.FPModel._static_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
        )


class PhiPartBase(DynamicSplitPartBase):
    """Unified Phi-family Part base."""

    partial_rotary_factor: float = DEFAULT_PARTIAL_ROTARY_FACTOR

    def get_graph_input_spec(self, graph_name: str) -> InputSpec:
        # Rewrite position_ids_{cos,sin}'s last dim to the *partial* rotary size,
        # matching PhiPreSplitBase._static_input_spec (which part 1 goes through).
        input_spec = super().get_graph_input_spec(graph_name)
        head_dim = self.head_dim or (self.hidden_size // self.num_attention_heads)
        embed_dim = int(head_dim * self.partial_rotary_factor) // 2
        return _fix_rope_embed_dim_input_spec(input_spec, embed_dim)


class PhiPreSplitCollectionBase(DynamicSplitCollectionBase):
    """Unified Collection base with N Parts for a Phi-family model."""

    default_context_lengths: list[int] = [512, 1024, 2048, 3072, 4096]

    # Appended verbatim to every chat-templated prompt file written for
    # genie-t2t-run evaluation.  The Phi-4-mini tokenizer's Jinja template
    # ends the generation primer at "<|assistant|>", but the model treats
    # <|endoftext|> (id=199999) as both BOS and EOS.  On prompts where the
    # model would otherwise emit <|endoftext|> as its very first token,
    # genie-t2t-run sees an immediate EOS hit and returns an empty output.
    # Pre-filling "<|endoftext|>" in the prompt file shifts it from a
    # generated token to a context token, so the model never triggers the
    # EOS check on it and proceeds to generate a real response.
    genie_generation_prompt_suffix: str = "<|endoftext|>"

    @staticmethod
    def _get_end_token_ids(tokenizer_src: str) -> list[int] | None:
        """Return the token ID for '<|end|>' (id=200020), or None on failure.

        Phi-4-mini uses '<|end|>' as the sole EOS stop token for on-device
        inference.  <|endoftext|> (id=199999) is pre-filled in every prompt
        file via genie_generation_prompt_suffix so it is never emitted as a
        generated token and does not need to be in the stop-token list.
        """
        try:
            from transformers import AutoTokenizer as _AutoTokenizer

            _tokenizer = _AutoTokenizer.from_pretrained(tokenizer_src)
            encoded = _tokenizer.encode("<|end|>", add_special_tokens=False)
            if len(encoded) == 1:
                return [int(encoded[0])]
            return None
        except Exception:
            logger.warning(
                "Could not encode <|end|>; genie_config.json eos-token "
                "will use the default llm_config.eos_token_id value."
            )
            return None

    # Token IDs for Phi-4-mini-instruct.
    # pad_token_id=200029 (<|endoftext|> is NOT the pad token for this model).
    # eos_token_id=200020 (<|end|> is the sole turn-end / stop token).
    # unk_token uses the Unicode replacement character U+FFFD.
    PAD_TOKEN_ID: int = 200029
    EOS_TOKEN_ID: int = 200020  # <|end|>
    UNK_TOKEN: str = "\ufffd"  # U+FFFD REPLACEMENT CHARACTER

    def _patch_tokenizer_files(self, output_path: Path) -> None:
        """Patch tokenizer_config.json with the correct special-token IDs,
        unk_token, and chat_template after the tokenizer has been saved.

        Changes applied
        ---------------
        * ``pad_token_id``  → 200029
        * ``eos_token_id``  → 200020  (``<|end|>``)
        * ``unk_token``     → U+FFFD  (replacement character)
        * ``chat_template`` → ``Phi35Base.CHAT_TEMPLATE``

        These values are consumed by:
        - ``genie-t2t-run`` (reads ``tokenizer_config.json`` for special IDs)
        - GenieX / ``geniex-bench`` (reads ``metadata.json`` chat template)
        - ``_write_eval_prompts_to_dir`` (applies chat template via
          ``tokenizer.apply_chat_template``)
        """
        tc_path = output_path / "tokenizer_config.json"
        if not tc_path.exists():
            return
        with open(tc_path, encoding="utf-8") as _f:
            tc = json.load(_f)

        tc["pad_token_id"] = self.PAD_TOKEN_ID
        tc["eos_token_id"] = self.EOS_TOKEN_ID
        tc["unk_token"] = self.UNK_TOKEN
        tc["chat_template"] = Phi35Base.CHAT_TEMPLATE

        with open(tc_path, "w", encoding="utf-8") as _f:
            json.dump(tc, _f, indent=2, ensure_ascii=False)

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """Write genie_config.json, tokenizer files, and sample_prompt.txt.

        Extends the base implementation with three Phi-4-mini-specific patches:

        1. **EOS token**: ``genie_config.json`` ``eos-token`` is set to 200020
           (``<|end|>``), the sole turn-end stop token.  ``<|endoftext|>``
           (id=199999) is pre-filled in every prompt file via
           ``genie_generation_prompt_suffix`` and therefore never needs to be
           in the stop-token list.

        2. **Pad token / unk token / chat template**: ``tokenizer_config.json``
           is patched so that ``pad_token_id=200029``, ``eos_token_id=200020``,
           ``unk_token=U+FFFD``, and ``chat_template`` matches
           ``Phi35Base.CHAT_TEMPLATE``.  This ensures both ``genie-t2t-run``
           and ``_write_eval_prompts_to_dir`` use the correct template.

        3. **BOS token in genie_config.json**: ``bos-token`` is set to 200029
           (the pad token ID), matching the model's actual BOS configuration.
        """
        output_path = Path(output_dir)

        # Resolve the tokenizer source the same way super() does.
        tok_src = (
            str(output_path)
            if (output_path / "tokenizer.json").exists()
            else self.hf_repo_name
        )
        end_token_ids = self._get_end_token_ids(tok_src)

        # Let super() run first: it saves tokenizer.json / config.json if
        # missing, then writes genie_config.json.
        super().write_supplementary_files(output_dir, metadata)

        # --- Patch tokenizer_config.json -----------------------------------
        self._patch_tokenizer_files(output_path)

        # --- Patch genie_config.json ---------------------------------------
        genie_config_path = output_path / "genie_config.json"
        if not genie_config_path.exists():
            return

        with open(genie_config_path) as _f:
            genie_cfg = json.load(_f)

        changed = False

        # Patch bos-token to pad_token_id (200029).
        try:
            if genie_cfg["dialog"]["context"].get("bos-token") != self.PAD_TOKEN_ID:
                genie_cfg["dialog"]["context"]["bos-token"] = self.PAD_TOKEN_ID
                changed = True
        except (KeyError, TypeError):
            logger.warning(
                "Unexpected genie_config.json structure; could not patch bos-token."
            )

        # Patch eos-token to the resolved <|end|> ID (200020).
        if end_token_ids:
            try:
                current_eos = genie_cfg["dialog"]["context"]["eos-token"]
            except (KeyError, TypeError):
                logger.warning(
                    "Unexpected genie_config.json structure; could not patch eos-token."
                )
                current_eos = None

            if current_eos is not None:
                # end_token_ids has exactly one entry: the <|end|> ID.
                desired_eos: int | list[int] = (
                    end_token_ids[0] if len(end_token_ids) == 1 else end_token_ids
                )
                current_eos_norm = (
                    current_eos if isinstance(current_eos, list) else [current_eos]
                )
                desired_eos_norm = (
                    desired_eos if isinstance(desired_eos, list) else [desired_eos]
                )
                if sorted(current_eos_norm) != sorted(desired_eos_norm):
                    genie_cfg["dialog"]["context"]["eos-token"] = desired_eos
                    changed = True

        if changed:
            with open(genie_config_path, "w") as _f:
                json.dump(genie_cfg, _f, indent=4)
