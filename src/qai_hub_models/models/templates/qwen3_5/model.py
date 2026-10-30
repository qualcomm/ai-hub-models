# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3.5 text decoder as a single static-shape graph (float).

Qwen3.5 interleaves Gated DeltaNet (linear attention) layers with gated full
attention layers. Linear layers carry a fixed-size conv state and recurrent
state instead of a KV cache; the I/O names follow lm_driver's layer_cache
conventions so HubCompatibleGenerator can drive the model unchanged.
"""

from __future__ import annotations

# isort: off
# This verifies aimet is installed, and this must be included first.
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    DynamicSplitCollectionBase,
    DynamicSplitPartBase,
    Embedding,
    LLMBase,
    LLMDynamic_AIMETOnnx,
    LLMDynamicBase,
    SingleSlotCacheMixin,
    get_onnx_model,
    sample_input,
)

# isort: on
import contextlib
import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar

import onnx
import torch
from qai_hub.client import Device
from torch import nn
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig
from transformers.models.qwen3_5 import modeling_qwen3_5
from typing_extensions import Self

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import aimet_onnx.common.quantsim as qs
    from aimet_onnx.quantsim import QuantizationSimModel, _quantize_subgraphs_enabled

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.model_adaptations import ConvInplaceLinear
from qai_hub_models.models.templates.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models.templates.lm_driver.utils.layer_cache import (
    AttentionType,
    LayerCacheDescriptor,
    build_layer_cache_descriptors,
    cache_state_names,
)
from qai_hub_models.models.templates.qwen3_5.model_adaptations import (
    SHAFullAttention,
    full_attention_forward,
    linear_attention_forward,
)
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle
from qai_hub_models.utils.quantization_aimet_onnx import ensure_min_aimet_onnx_version

END_TOKENS = {"<|im_end|>", "<|endoftext|>"}

MIN_AIMET_ONNX_VERSION = "2.42.0"

# Compiled with --force_qdq_conversion, which needs 2.0.0's explicit
# per-channel axis rather than one inferred from tensor shapes.
ENCODING_VERSION = "2.0.0"


class Qwen3_5RopeEmbedding(Embedding):
    """Precomputed RoPE tables for the rotated slice of each head.

    Qwen3.5 uses 3D interleaved MRoPE, but for text the three position axes
    are equal, which reduces exactly to 1D RoPE over ``partial_rotary_factor *
    head_dim`` channels.
    """

    def __init__(
        self,
        head_dim: int | None = None,
        max_length: int = 2048,
        config: PretrainedConfig | None = None,
    ) -> None:
        assert config is not None
        rope = config.rope_parameters
        rotary_dim = int(config.head_dim * rope.get("partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (
            rope["rope_theta"]
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        freqs = torch.arange(max_length, dtype=torch.float32)[:, None] * inv_freq
        self.cos = freqs.cos()[None, None]
        self.sin = freqs.sin()[None, None]

    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, rotary_dim // 2][2]
        """
        cos = self.cos[0, 0].to(position_ids.device)
        sin = self.sin[0, 0].to(position_ids.device)
        cos = cos[position_ids].unsqueeze(1).to(dtype=dtype)
        sin = sin[position_ids].unsqueeze(1).to(dtype=dtype)
        return cos, sin


class Qwen3_5StateIOMixin:
    """Input/output specs for Qwen3.5's per-layer KV and linear-attention state.

    Shared by the float model and the QuantSim model, which both carry
    ``llm_config``, ``embedding`` and ``tokenizer``.
    """

    llm_config: PretrainedConfig
    embedding: Embedding
    tokenizer: Any
    sequence_length: int
    context_length: int

    @property
    def layer_cache_descriptors(self) -> list[LayerCacheDescriptor]:
        return build_layer_cache_descriptors(self.llm_config)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Unused; the spec is derived from the loaded config.
        sequence_length
            Tokens processed per forward call.
        context_length
            Total tokens of context (KV cache size plus sequence length).
        llm_io_type
            Only genie_input_ids is supported.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        assert llm_io_type == LLMIOType.genie_input_ids
        assert sequence_length < context_length
        rotary_half = self.embedding.cos.shape[-1]  # type: ignore[attr-defined]
        spec: InputSpec = {
            "input_ids": TensorSpec(shape=(1, sequence_length), dtype="int32"),
            "attention_mask": TensorSpec(
                shape=(1, 1, sequence_length, context_length), dtype="float32"
            ),
            "position_ids_cos": TensorSpec(
                shape=(1, 1, sequence_length, rotary_half), dtype="float32"
            ),
            "position_ids_sin": TensorSpec(
                shape=(1, 1, sequence_length, rotary_half), dtype="float32"
            ),
        }
        descs = self.layer_cache_descriptors
        names = cache_state_names(descs, "in")
        for desc, name_a, name_b in zip(descs, names[::2], names[1::2], strict=True):
            shape_a, shape_b = desc.dummy_state_shapes(
                1, context_length, sequence_length
            )
            if desc.attention_type != AttentionType.LINEAR:
                # Hub KV layout: keys (H, B, D, S), values (H, B, S, D).
                b, h, s, d = shape_a
                shape_a, shape_b = (h, b, d, s), (h, b, s, d)
            spec[name_a] = TensorSpec(shape=shape_a, dtype="float32")
            spec[name_b] = TensorSpec(shape=shape_b, dtype="float32")
        return spec

    def get_output_spec(self) -> OutputSpec:
        names = ["logits", *cache_state_names(self.layer_cache_descriptors, "out")]
        return {name: TensorSpec() for name in names}

    def _get_output_spec(self, num_hidden_layers: int) -> OutputSpec:
        # get_onnx_model names ONNX outputs through this; the LLMBase version
        # would call every layer's state past_key/past_value.
        return self.get_output_spec()

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
            )
        inputs = sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),  # type: ignore[attr-defined]
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )
        for name, (shape, _) in input_spec.items():
            if name.startswith(("conv_state_", "recurrent_state_")):
                inputs[name] = [torch.zeros(shape).numpy()]
        return {name: inputs[name] for name in input_spec}


class Qwen3_5Base(Qwen3_5StateIOMixin, LLMBase):
    """FP Qwen3.5 text decoder with explicit per-layer state I/O."""

    LMClass = modeling_qwen3_5.Qwen3_5ForCausalLM
    EmbeddingClass = Qwen3_5RopeEmbedding
    GeneratorClass = HubCompatibleGenerator

    default_user_prompt = "What is gravity? Keep the answer under ten words."
    default_system_prompt = "You are a helpful AI assistant."

    hf_repo_name: str = ""

    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        # The config declares bfloat16, so from_pretrained would round the
        # checkpoint's fp32 tensors (e.g. linear_attn.norm) before .float().
        llm_config.dtype = torch.float32
        return llm_config

    def _verify_ckpt(self) -> None:
        if self.llm_config.model_type != "qwen3_5_text":
            raise ValueError(
                "Model config is not compatible with this model implementation."
            )

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT_UNQUANTIZED",
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        """
        Load the float model.

        Parameters
        ----------
        checkpoint
            Hugging Face repo name or local folder. DEFAULT_UNQUANTIZED loads
            the model's Hugging Face repo.
        sequence_length
            Tokens processed per forward call.
        context_length
            Total tokens of context (KV cache size plus sequence length).
        host_device
            Torch device to load the model on.
        _skip_optimizations
            Pass ["sha_attention"] to keep the batched multi-head attention.

        Returns
        -------
        Self
            The loaded model.
        """
        if checkpoint == "DEFAULT_UNQUANTIZED":
            checkpoint = cls.hf_repo_name
        return cls(
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            context_length=context_length,
            host_device=host_device,
            _skip_optimizations=_skip_optimizations,
        )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        use_sha = not (
            self.skip_optimizations and "sha_attention" in self.skip_optimizations
        )
        causal_lm: Any = self.model
        text_model = causal_lm.model
        # Same conversions as QCQwen3MLP / QCQwen3ForCausalLM. up_proj and
        # gate_proj stay Linear there too (AISW-148745).
        for layer in text_model.layers:
            layer.mlp.down_proj = ConvInplaceLinear(layer.mlp.down_proj)
        causal_lm.lm_head = ConvInplaceLinear(causal_lm.lm_head)
        self.sha_attention = nn.ModuleDict(
            {
                str(desc.layer_idx): SHAFullAttention(
                    text_model.layers[desc.layer_idx].self_attn
                )
                for desc in self.layer_cache_descriptors
                if use_sha and desc.attention_type != AttentionType.LINEAR
            }
        )

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        rope_cos, rope_sin, *states = args
        seq_len = input_tokens.shape[1]
        # A token is real iff it attends to itself, for left or right padding.
        # Kept in float: HTP has no integer ReduceMax, which any() lowers to.
        attends = (attention_mask[:, 0, :, -seq_len:] == 0).to(torch.float32)
        eye = torch.eye(seq_len, device=attention_mask.device)
        valid = (attends * eye).sum(dim=-1)

        causal_lm: Any = self.model
        text_model = causal_lm.model
        hidden_states = text_model.embed_tokens(input_tokens)
        new_states: list[torch.Tensor] = []
        for desc, layer in zip(
            self.layer_cache_descriptors, text_model.layers, strict=True
        ):
            state_a, state_b = states[2 * desc.layer_idx : 2 * desc.layer_idx + 2]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            if desc.attention_type == AttentionType.LINEAR:
                hidden_states, state_a, state_b = linear_attention_forward(
                    layer.linear_attn, hidden_states, valid, state_a, state_b
                )
            else:
                sha = (
                    self.sha_attention[str(desc.layer_idx)]
                    if self.sha_attention
                    else None
                )
                attention = sha or partial(full_attention_forward, layer.self_attn)
                hidden_states, state_a, state_b = attention(
                    hidden_states,
                    attention_mask,
                    rope_cos,
                    rope_sin,
                    state_a,
                    state_b,
                )
            hidden_states = residual + hidden_states
            hidden_states = hidden_states + layer.mlp(
                layer.post_attention_layernorm(hidden_states)
            )
            new_states += [state_a, state_b]

        logits = causal_lm.lm_head(text_model.norm(hidden_states))
        return [logits, *new_states]

    @classmethod
    def get_input_prompt_with_tags(
        cls,
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        tokenizer: Any = None,
        include_image: bool = False,
        **kwargs: Any,
    ) -> str:
        # Qwen3.5 thinks by default; demos and sample inputs answer directly.
        kwargs.setdefault("enable_thinking", False)
        return super().get_input_prompt_with_tags(
            user_input_prompt=user_input_prompt,
            system_context_prompt=system_context_prompt,
            tokenizer=tokenizer,
            include_image=include_image,
            **kwargs,
        )


def _tie_linear_state_quantizers(quant_sim: QuantizationSimModel) -> None:
    """Share one quantizer between each conv/recurrent state input and output.

    The int8 kv_cache rule only covers past_key/past_value, so these states stay
    at 16 bits; in and out must share a scale or the next step reads it mis-scaled.
    """
    mapping = {}
    for tensor in quant_sim.model.graph().input:
        if not tensor.name.startswith(("conv_state_", "recurrent_state_")):
            continue
        output_name = tensor.name.removesuffix("_in") + "_out"
        quantizer = quant_sim._get_enabled_quantizer(output_name)  # pylint: disable=protected-access
        if quantizer is not None:
            mapping[tensor.name] = quantizer
    # set_quantizers also rewires the graph node; writing the dict alone does not.
    quant_sim.set_quantizers(mapping)


class Qwen3_5DynamicBase(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, LLMDynamicBase, Qwen3_5Base
):
    """Qwen3.5 float model with one dynamic-shape ONNX export (quantize/evaluate)."""

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

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: str | os.PathLike | Path | None = None,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        """
        Load or return the cached float model.

        Parameters
        ----------
        checkpoint
            Hugging Face repo name or local folder. Defaults to hf_repo_name.
        host_device
            Torch device to load the model on.
        _skip_optimizations
            Pass ["sha_attention"] to keep the batched multi-head attention.

        Returns
        -------
        Self
            The cached or newly loaded model.
        """
        checkpoint = checkpoint or cls.hf_repo_name
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached
        instance = cls(
            checkpoint=checkpoint,
            host_device=host_device,
            _skip_optimizations=_skip_optimizations,
        )
        cls.cache_store(instance, cache_key)
        return instance

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        get_onnx_model(
            fp_model=self,
            context_length=self.context_length,
            sequence_length=self.sequence_length,
            path=str(onnx_dir / "model.onnx"),
            llm_io_type=self.llm_io_type,
            quiet=True,
        )
        return ONNXBundle.from_bundle_path(onnx_dir, "model")


Qwen3_5DynamicT = TypeVar("Qwen3_5DynamicT", bound=Qwen3_5DynamicBase)


class Qwen3_5QuantizableBase(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin[Qwen3_5DynamicT],
    Qwen3_5StateIOMixin,
    LLMDynamic_AIMETOnnx,
    Generic[Qwen3_5DynamicT],
):
    """QuantSim Qwen3.5 text decoder for quantize and evaluate.

    Needs aimet-onnx 2.42.0 for Scan-body quantization, the qwen3_5 AdaScale
    config, and the ONNX Scan converter (aimet#7952).
    """

    EmbeddingClass = Qwen3_5RopeEmbedding
    GeneratorClass = HubCompatibleGenerator
    FPModel: type[Qwen3_5DynamicT]

    ada_scale_model_type: str | None = "qwen3_5"

    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict[Precision, str] = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    @classmethod
    def create_quantsim(
        cls,
        onnx_model: onnx.ModelProto,
        host_device: torch.device,
        precision: Precision,
    ) -> QuantizationSimModel:
        ensure_min_aimet_onnx_version(MIN_AIMET_ONNX_VERSION, cls.model_id or None)
        # The delta rule exports as a Scan; the setting is captured at sim
        # construction, so calibration and export keep quantizing the bodies.
        with _quantize_subgraphs_enabled():
            return super().create_quantsim(onnx_model, host_device, precision)

    def save_calibrated_checkpoint(  # type: ignore[override]
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: Qwen3_5DynamicT | None = None,
    ) -> None:
        # _build_quantsim pins 1.0.0 for every LLM; only this export needs 2.0.0.
        default_version = qs.encoding_version
        qs.encoding_version = ENCODING_VERSION
        try:
            super().save_calibrated_checkpoint(output_checkpoint, fp_model)
        finally:
            qs.encoding_version = default_version

    @classmethod
    def _configure_quant_sim(
        cls,
        quant_sim: QuantizationSimModel,
        precision: Precision,
    ) -> QuantizationSimModel:
        quant_sim = super()._configure_quant_sim(quant_sim, precision)
        _tie_linear_state_quantizers(quant_sim)
        return quant_sim


def _graph_tensor_names(graph: onnx.GraphProto) -> set[str]:
    """Every tensor name in graph, including those inside Scan bodies."""
    names = {t.name for t in graph.input}
    names |= {t.name for t in graph.output}
    names |= {t.name for t in graph.initializer}
    for node in graph.node:
        names.update(node.output)
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                names |= _graph_tensor_names(attr.g)
    return names


class Qwen3_5PartBase(DynamicSplitPartBase):
    """One split of the Qwen3.5 text decoder.

    Input shapes come from the full model's spec: the conv/recurrent states and
    partial-rotary RoPE tables don't follow LLMPartBase's KV-only conventions.
    """

    def get_graph_input_spec(self, graph_name: str) -> InputSpec:
        sequence_length, context_length = self._graph_names[graph_name]
        full_spec = self._presplit.get_input_spec(
            sequence_length=sequence_length, context_length=context_length
        )
        hidden = TensorSpec(
            shape=(1, sequence_length, self.hidden_size), dtype="float32"
        )
        return {
            name: full_spec.get(name, hidden) for name in self._get_onnx_input_names()
        }

    def _get_quant_sim(self) -> QuantizationSimModel:
        with _quantize_subgraphs_enabled():
            return super()._get_quant_sim()

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        # QDQ conversion keeps the onnx2onnx optimizer on, which folds the
        # delta-rule Trilu masks and unrolls the Scan that qairt-converter rejects.
        return super().get_graph_hub_compile_options(
            graph_name,
            target_runtime,
            precision,
            other_compile_options + " --force_qdq_conversion -O2",
            device,
        )

    def transform_split_encodings(
        self,
        onnx_model: onnx.ModelProto,
        quant_sim: QuantizationSimModel,
        encodings_path: str,
    ) -> str:
        # The generic split keeps encodings for top-level node outputs only,
        # dropping the Scan-body activations and quantized constants.
        checkpoint = getattr(self._presplit, "checkpoint", None)
        full_path = Path(str(checkpoint)) / "model.encodings"
        if checkpoint is None or not full_path.exists():
            return encodings_path
        names = _graph_tensor_names(onnx_model.graph)
        # Gather outputs carry no activation encodings (AISW-152612).
        names -= {n.output[0] for n in onnx_model.graph.node if n.op_type == "Gather"}
        with open(full_path) as f:
            encodings = json.load(f)
        for key in ("encodings", "activation_encodings", "param_encodings"):
            if key in encodings:
                encodings[key] = [e for e in encodings[key] if e["name"] in names]
        part_path = Path(encodings_path).with_suffix(".scan.encodings")
        with open(part_path, "w") as f:
            json.dump(encodings, f, indent=4, sort_keys=True)
        return str(part_path)


def _fix_qwen3_5_genie_config(inner: dict[str, Any], bundle_dir: Path) -> None:
    """Apply the Qwen3.5 settings create_genie_config can't express."""
    llm_config = AutoConfig.from_pretrained(str(bundle_dir))
    rope = llm_config.rope_parameters
    rotary_dim = int(llm_config.head_dim * rope.get("partial_rotary_factor", 1.0))

    inner["context"]["bos-token"] = -1
    # <|im_end|> ends a chat turn; config.json's eos_token_id is <|endoftext|>.
    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    inner["context"]["eos-token"] = tokenizer.convert_tokens_to_ids("<|im_end|>")

    qnn_htp = inner["engine"]["backend"]["QnnHtp"]
    qnn_htp["kv-dim"] = llm_config.head_dim
    qnn_htp["enable-graph-switching"] = False
    # Expressed as positional-encoding below; two sources would conflict.
    qnn_htp.pop("pos-id-dim", None)
    qnn_htp.pop("rope-theta", None)

    model = inner["engine"]["model"]
    model["linear-attention"] = True
    model["positional-encoding"] = {
        "type": "rope",
        "rope-dim": rotary_dim // 2,
        "rope-theta": int(rope["rope_theta"]),
        "rope-scaling": {
            "rope-type": rope.get("rope_type", "default"),
            "mrope-section": list(rope["mrope_section"]),
        },
    }


class Qwen3_5CollectionBase(DynamicSplitCollectionBase):
    """Collection of Qwen3.5 split Parts, exported as one Genie bundle."""

    supports_thinking = True
    think_start = "<think>\n"
    think_end = "\n</think>"

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        super().write_supplementary_files(output_dir, metadata)
        config_path = Path(output_dir) / "genie_config.json"
        with open(config_path) as f:
            config = json.load(f)
        _fix_qwen3_5_genie_config(config["dialog"], Path(output_dir))
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
