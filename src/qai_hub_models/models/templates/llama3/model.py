# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models.templates.llm.model import (
    LLMBase,
    LLM_AIMETOnnx,
    LLM_QNN,
    LLMDynamicBase,
    LLMDynamic_AIMETOnnx,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    FPModelT,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    DynamicSplitCollectionBase,
    DynamicSplitPartBase,
    SingleSlotCacheMixin,
    get_onnx_model,
)

# isort: on
import copy
import json
import os
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import onnx
import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from aimet_onnx.quantsim import QuantizationSimModel

import qai_hub as hub
import transformers
from packaging.version import Version
from transformers import (
    PretrainedConfig,
    PreTrainedTokenizer,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama import LlamaConfig, modeling_llama

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.templates.llama3.model_adaptations import (
    QcLlama_apply_rotary_pos_emb,
    QCLlamaForCausalLM,
    QCLlamaMLP,
    SHALlamaAttention,
)
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.model import (
    Embedding,
    PositionProcessorBase,
)
from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.envvars import CheckpointEnvvar
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec
from qai_hub_models.utils.onnx.helpers import (
    ONNXBundle,
)
from qai_hub_models.utils.printing import print_with_box

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
        self.cos, self.sin = self.precompute(head_dim, max_length, config)  # type: ignore[arg-type, unused-ignore]

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


class Llama3Base(LLMBase):
    LMClass = QCLlamaForCausalLM
    EmbeddingClass = RopeEmbedding

    # Default prompts for demos
    default_user_prompt = "What do llamas eat? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        return {
            "global_prefix": BEGIN_TEXT,
            "system_prefix": f"{START_HEADER}{SYSTEM_ID}{END_HEADER}\n\n",
            "system_suffix": EOT_ID,
            "user_prefix": f"{START_HEADER}{USER_ID}{END_HEADER}\n\n",
            "user_suffix": EOT_ID,
            "assistant_prefix": f"{START_HEADER}{ASSISTANT_ID}{END_HEADER}\n\n",
            "assistant_suffix": EOT_ID,
            "default_system_prompt": cls.default_system_prompt,
        }

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
        context_lengths: list[int],
        model_list: list[str],
        output_path: Path,
        runtime: TargetRuntime,
        precision: Precision,
        encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
        model_id: str,
        model_name: str,
    ) -> None:
        super().prepare_genie_assets(
            hub_device,
            checkpoint,
            llm_config,
            context_lengths,
            model_list,
            output_path,
            runtime,
            precision,
            encodings_path,
            input_specs,
            output_specs,
            model_id=model_id,
            model_name=model_name,
        )

    @staticmethod
    def _get_output_spec(num_hidden_layers: int) -> OutputSpec:
        output_spec: OutputSpec = {"logits": TensorSpec()}
        for layer in range(num_hidden_layers):
            output_spec[f"past_key_{layer}_out"] = TensorSpec()
            output_spec[f"past_value_{layer}_out"] = TensorSpec()
        return output_spec

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
            LLMIOType.genie_input_ids_native_kv,
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
            LLMIOType.genie_input_ids_native_kv,
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


# ---------------------------------------------------------------------------
# Dynamic-shape Llama3 classes
# ---------------------------------------------------------------------------


class Llama3DynamicBase(LLMDynamicBase, Llama3Base):
    """Llama3 FP base with dynamic-shape ONNX export.

    Provides get_full_onnx_bundle() which exports the torch model to ONNX
    with dynamic shapes, caching the result alongside model.encodings.
    """

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Export full ONNX from PyTorch with dynamic shapes.

        Always exports fresh into ``temp_path``. The FP export is deliberately
        NOT cached to disk: the natural cache dir (the default checkpoint dir,
        keyed by precision) is the same path the published quantized asset
        extracts to, whose ``model_dynamic.onnx`` could be SpinQuant-rotated. Reusing
        that path let a rotated graph pose as the FP export and get fed back
        into rotation (double-SpinQuant -> R1 fails on a leftover R3 node).
        """
        print_with_box(
            [
                "Exporting ONNX model with dynamic shapes.",
                "This may take around 30 minutes.",
            ]
        )
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


class Llama3DynamicBase_AIMETOnnx(LLMDynamic_AIMETOnnx, Llama3Base_AIMETOnnx):
    """Dynamic-shape variant of Llama3Base_AIMETOnnx."""

    FPModel = Llama3DynamicBase  # type: ignore[assignment]

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """Adapt AIMET encodings for the dynamic-shape ONNX model.

        The dynamic ONNX uses node names from torch.export, so this override
        finds the embedding Gather by op_type rather than hardcoded static
        names. Single pass:
        1. Sets the embedding Gather output encoding (precision-dependent)
        2. Promotes weight activation encodings to param_encodings
        3. Propagates encodings through memory ops for correct splitting
        """
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

        # Find the embedding Gather node and set its output encoding.
        # For w4 (no activation quantization): use fp16.
        # For w4a16: copy the embedding weight's quantization parameters.
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

        # Promote weight entries in activation_encodings to param_encodings
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


class LlamaDynamicQuantizablePreSplitMixin(DynamicQuantizablePreSplitMixin[FPModelT]):
    """Llama-specific DynamicQuantizablePreSplit that exports ONNX from torch.

    Overrides resolve_default_checkpoint to download only the quantization
    encodings from the asset store, then export the ONNX model locally
    from the FP torch model. This avoids storing large ONNX files in the
    asset store at the cost of a longer first-load time (~30 min).
    """

    @classmethod
    def resolve_default_checkpoint(
        cls,
        precision: Precision,
        host_device: torch.device,
        fp_model: FPModelT | None,
    ) -> tuple[str, FPModelT | None]:
        """Fetch encodings only and export ONNX from the FP torch model.

        QAIHM_TEST_CHECKPOINT takes the encodings from a private-bucket zip
        instead of the asset store; the ONNX is still re-exported.

        Parameters
        ----------
        precision
            Quantization precision (already validated).
        host_device
            Device for computation.
        fp_model
            Optional FP model passed by the evaluate framework.

        Returns
        -------
        tuple[str, FPModelT | None]
            (resolved_checkpoint_path, fp_model).
        """
        from qai_hub_models.utils.printing import print_with_box

        # Encodings only, from the private zip rather than the asset store. This
        # family's license does not cover redistributing the weights, so the
        # ONNX is re-exported below either way.
        private_key = CheckpointEnvvar.get()
        if private_key:
            checkpoint = str(cls.fetch_default_checkpoint(precision))
        else:
            precision_checkpoint = cls.default_checkpoint[precision]
            encodings_path = str(
                CachedWebModelAsset.from_asset_store(
                    cls.model_id,
                    cls.model_asset_version,
                    f"{precision_checkpoint}/model.encodings",
                ).fetch()
            )
            checkpoint = str(Path(encodings_path).parent)

        # Create FP model for ONNX export + tokenizer/config
        if fp_model is None:
            fp_model = cls.FPModel.from_pretrained(  # type: ignore[call-arg]
                host_device=host_device,
            )

        # Export ONNX into checkpoint dir (skips if already exists)
        ckpt_path = Path(checkpoint)
        if (
            not (ckpt_path / "model_dynamic.onnx").exists()
            or not (ckpt_path / "model.data").exists()
        ):
            print_with_box(
                [
                    "Exporting ONNX model with dynamic shapes.",
                    "This may take around 30 minutes.",
                ]
            )
        cls.create_onnx_models(  # type: ignore[attr-defined]
            checkpoint=checkpoint,
            fp_model=fp_model,
            context_length=fp_model.context_length,
            host_device=host_device,
            llm_io_type=fp_model.llm_io_type,
        )
        cls.save_tokenizer_and_config(  # type: ignore[attr-defined]
            checkpoint=checkpoint, fp_model=fp_model
        )
        return checkpoint, fp_model


class LlamaPreSplitBase(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Llama3DynamicBase
):
    """FP PreSplit base for Llama3 models.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.

    Concrete subclasses must set the architecture constants below.
    """

    # --- per-model configuration (override in subclass) ---
    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    hf_repo_name: str = ""

    # DynamicPreSplitOnnxMixin config
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    # Asset / cache config
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
        """Verify checkpoint compatibility."""
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
        """
        Load or return a cached FP PreSplit.

        Uses dynamic shapes so sequence_length/context_length are not
        needed at construction time.
        """
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
        """Get output names for the full model."""
        return Llama3DynamicBase._get_output_spec(self.num_layers)

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
        """
        Build the full-model input spec from the model's architecture
        constants (or an explicit ``llm_config`` override).

        This is a classmethod (not an instance method) so the Quantizable
        PreSplit can build the spec from the FP class without an instance.
        """
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": cls.num_layers,
                "hidden_size": cls.hidden_size,
                "num_key_value_heads": cls.num_key_value_heads,
                "num_attention_heads": cls.num_attention_heads,
            }
        return Llama3DynamicBase._get_input_spec(
            num_hidden_layers=llm_config.get("num_hidden_layers", cls.num_layers),
            sequence_length=sequence_length,
            context_length=context_length,
            hidden_size=llm_config.get("hidden_size", cls.hidden_size),
            num_key_value_heads=llm_config.get(
                "num_key_value_heads", cls.num_key_value_heads
            ),
            num_attention_heads=llm_config.get(
                "num_attention_heads", cls.num_attention_heads
            ),
            llm_io_type=llm_io_type,
        )


LlamaPreSplitT = TypeVar("LlamaPreSplitT", bound=LlamaPreSplitBase)


class LlamaQuantizablePreSplitBase(  # type: ignore[misc]
    LlamaDynamicQuantizablePreSplitMixin[LlamaPreSplitT],
    Llama3DynamicBase_AIMETOnnx,
    Generic[LlamaPreSplitT],
):
    """Quantizable PreSplit base for Llama3 models.

    Manages QuantSim and calibration. Uses class-level cache keyed by
    checkpoint to reuse instances across calls with different
    sequence/context lengths (dynamic shapes). When a different checkpoint
    is requested, the old instance is evicted and freed.

    Concrete subclasses must set ``FPModel`` and the config attributes.
    """

    # Set by subclass.
    FPModel: type[LlamaPreSplitT]

    # Config
    num_layers: int = 0
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    # DynamicPreSplitOnnxMixin config
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0

    def get_output_spec(self) -> OutputSpec:
        """Get output names for the full model."""
        return Llama3DynamicBase._get_output_spec(self.num_layers)

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
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


class LlamaPartBase(DynamicSplitPartBase):
    """Unified Llama Part base.

    All split/inference machinery is generic and lives in
    :class:`DynamicSplitPartBase`. Concrete models supply the architecture
    constants and the FP / Quantizable PreSplit classes via class attributes.
    """


class LlamaPreSplitCollectionBase(DynamicSplitCollectionBase):
    """Unified Collection base with N Parts for a Llama3 model.

    All deployment machinery is generic and lives in
    :class:`DynamicSplitCollectionBase`. Concrete models register their Part
    classes via ``add_component`` and set ``hf_repo_name`` / ``fp_presplit_cls``
    / ``part_base_cls``.
    """
