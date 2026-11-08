# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import copy
import functools
import gc
import itertools
import json
import logging
import os
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, cast

import numpy as np
import onnx
import onnxruntime
import torch
from transformers import AutoProcessor, PretrainedConfig, PreTrainedTokenizer
from transformers.models.llama import LlamaConfig
from transformers.models.qwen3_vl import modeling_qwen3_vl

# isort: off
# This verifies aimet is installed, and this must be included first.
with contextlib.suppress(ImportError, ModuleNotFoundError):
    from aimet_onnx.common.defs import QuantScheme
    from aimet_onnx.quantsim import QuantizationSimModel, load_encodings_to_sim
# isort: on

from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
)
from qai_hub_models.configs.model_metadata import (
    GenieChatTemplate,
    GenieMetadata,
    GeniePipeline,
    GeniePipelineConnection,
    GenieSampleInput,
    GenieVisionPreprocessing,
    ModelMetadata,
)
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.llm_helpers import (
    create_genie_config,
    export_embedding_weights_from_tensor,
    generate_genie_app_script,
    get_rope_scaling,
    save_htp_config_for_genie_bundle,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    DynamicPreSplitOnnxMixin,
    DynamicQuantizablePreSplitMixin,
    LLMCollectionBase,
    LLMDynamic_AIMETOnnx,
    LLMPartBase,
    SingleSlotCacheMixin,
    SplitForwardMixin,
    get_onnx_model,
    get_tokenizer,
)
from qai_hub_models.models.templates.lm_driver.generator import (
    PrecomputedCosSinGeneratorMixin,
    TransposedKVGeneratorMixin,
    VLM_Generator,
)
from qai_hub_models.models.templates.lm_driver.qwen3_vl import (
    Qwen3VL_Generator,
    Qwen_3_VL,
)
from qai_hub_models.models.templates.qwen3.model import (
    Qwen3Base,
    Qwen3Base_AIMETOnnx,
    Qwen3Base_QNN,
    Qwen3PositionProcessor,
)
from qai_hub_models.models.templates.qwen3_vl.vision_encoder import (
    Qwen3VLVisionEncoder,
    Qwen3VLVisionWrapper,
)
from qai_hub_models.models.templates.vlm.model import (
    DEFAULT_IMAGE_SIZE,
    VLMBase,
    VLMDynamic_AIMETOnnx,
)
from qai_hub_models.models.templates.vlm.processor_factory import VLMProcessorLike
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_multi_graph_model import (
    MultiGraphWorkbenchModel,
)
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec
from qai_hub_models.utils.onnx.helpers import ONNXBundle, mock_torch_onnx_inference

if TYPE_CHECKING:
    from qai_hub_models.utils.base_dataset import BaseDataset
    from qai_hub_models.utils.base_model import BaseModel


from qai_hub_models.utils.system_info import has_recommended_memory

logger = logging.getLogger(__name__)

END_TOKENS = {"<|im_end|>", "<|endoftext|>"}

DEFAULT_PROMPT_CONTEXT = "You are a helpful AI assistant."
DEFAULT_USER_PROMPT = "Give me a short introduction to large language model."


def _vlm_eval_dataset_classes() -> list[type[BaseDataset]]:
    """Eval datasets for VLM models: the text-only LLM tasks plus MMMU and multimodal prompts."""
    from qai_hub_models.datasets.grace import MultimodalPrompts
    from qai_hub_models.datasets.mmmu import MMMU
    from qai_hub_models.models.templates.llm.model import LLMBase

    return [*LLMBase.get_eval_dataset_classes(), MMMU, MultimodalPrompts]


class HubCompatibleQwen3VLGenerator(  # type: ignore[misc]
    PrecomputedCosSinGeneratorMixin, TransposedKVGeneratorMixin, Qwen3VL_Generator
):
    pass


class _VLMCausalLMWrapper(torch.nn.Module):
    """Wrap language_model + lm_head so the whole forward lives inside one Module.

    This is necessary for ``torch.export`` (dynamo) tracing: when
    ``self.model`` is just the text encoder and the lm_head sits outside,
    dynamo cannot capture the KV-cache output tensors. By combining them
    here, the forward graph is self-contained and all outputs
    (logits + KV) are preserved.

    For Qwen3-VL, the language model's forward accepts deepstack kwargs
    (visual_pos_masks, deepstack_visual_embeds) which must be passed through.
    """

    def __init__(self, text_model: torch.nn.Module, lm_head: torch.nn.Module) -> None:
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return {
            "logits": logits,
            "past_key_values": outputs.past_key_values,
        }


def get_vlm_config(model_ckpt: str | os.PathLike | Path | None) -> PretrainedConfig:
    """Construct and return a HuggingFace LLM config for Qwen3-VL."""
    from transformers import AutoConfig

    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    # The config may be the full VLM config (Qwen3VLConfig, has .text_config) when
    # loaded from the HF repo, or the bare text config (Qwen3VLTextConfig, no
    # .text_config) when loaded from a split/quantized checkpoint. Resolve the
    # text config for either layout, mirroring _verify_ckpt's handling.
    text_config = getattr(llm_config, "text_config", llm_config)
    text_config._attn_implementation = "eager"
    text_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    text_config.use_cache = True

    return llm_config


class Qwen3VLTextBase(VLMBase, Qwen3Base):
    """
    Base class for Qwen3-VL text model.

    Key differences from Qwen3Base:
    - Uses LLMIOType.genie_input_embeds
    - Input is embeddings, not token IDs
    - Loads from full VLM checkpoint and extracts text model
    - Handles deepstack visual embeddings injected at intermediate layers
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    GeneratorClass: type[VLM_Generator] = HubCompatibleQwen3VLGenerator

    # We use the full VLM class for loading, then extract text model
    LMClass = modeling_qwen3_vl.Qwen3VLForConditionalGeneration  # type: ignore[assignment, unused-ignore]

    VisionModelWrapper: type[torch.nn.Module] = Qwen3VLVisionWrapper

    # Store reference to full VLM for embedding extraction
    _full_vlm: torch.nn.Module | None = None

    @classmethod
    def get_visual_output_names(cls, config: PretrainedConfig) -> tuple[str, ...]:
        return Qwen_3_VL.get_visual_output_names()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        spec = super().get_chat_template()
        assert spec is not None
        spec["vision_start"] = "<|vision_start|>"
        spec["vision_end"] = "<|vision_end|>"
        return spec

    @classmethod
    def edit_llm_config(cls, llm_config: PretrainedConfig) -> PretrainedConfig:
        """Extract text_config from the full Qwen3VL config."""
        if llm_config.model_type == "qwen3":
            return llm_config

        if hasattr(llm_config, "text_config"):
            return llm_config.text_config

        return llm_config

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int | None = None,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        num_deepstack_layers: int = 0,
        num_visual_tokens: int = 256,
    ) -> InputSpec:
        """
        Get input spec for VLM text model.

        Uses inputs_embeds instead of input_ids. Position embeddings (cos/sin)
        are pre-computed externally and passed as inputs.
        Includes deepstack visual embeddings injected at intermediate layers.
        """
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        embed_dim = head_dim // 2

        input_spec: InputSpec = {}

        # VLM uses inputs_embeds
        input_spec["inputs_embeds"] = TensorSpec(
            shape=(1, sequence_length, hidden_size),
            dtype="float32",
        )

        input_spec["attention_mask"] = TensorSpec(
            shape=(1, 1, sequence_length, context_length),
            dtype="float32",
        )

        input_spec["position_ids_cos"] = TensorSpec(
            shape=(1, 1, sequence_length, embed_dim),
            dtype="float32",
        )
        input_spec["position_ids_sin"] = TensorSpec(
            shape=(1, 1, sequence_length, embed_dim),
            dtype="float32",
        )

        # KV cache for each layer
        assert sequence_length < context_length, (
            "It is currently not supported to set input sequence length to the same "
            "as or longer than context length."
        )

        for layer in range(num_hidden_layers):
            past_k_name = f"past_key_{layer}_in"
            input_spec[past_k_name] = TensorSpec(
                shape=(
                    num_key_value_heads,
                    1,
                    head_dim,
                    context_length - sequence_length,
                ),
                dtype="float32",
            )

            past_v_name = f"past_value_{layer}_in"
            input_spec[past_v_name] = TensorSpec(
                shape=(
                    num_key_value_heads,
                    1,
                    context_length - sequence_length,
                    head_dim,
                ),
                dtype="float32",
            )

        # Deepstack: visual_pos_masks marks which sequence positions contain
        # vision tokens; deepstack_visual_embeds_i are per-layer visual features
        # injected at intermediate decoder layers.
        if num_deepstack_layers > 0:
            input_spec["visual_pos_masks"] = TensorSpec(
                shape=(1, sequence_length),
                dtype="bool",
            )
            for i in range(num_deepstack_layers):
                input_spec[f"deepstack_visual_embeds_{i}"] = TensorSpec(
                    shape=(num_visual_tokens, hidden_size),
                    dtype="float32",
                )

        return input_spec

    def _apply_monkey_patch(self) -> None:
        self.monkey_patch(skip_optimizations=self.skip_optimizations)

    def _load_tokenizer(self, checkpoint: str | os.PathLike | Path) -> Any:
        return get_tokenizer(checkpoint)

    def _load_image_processor(self, checkpoint: str | os.PathLike | Path) -> Any:
        # Cache HF image processor config for vision preprocessing metadata
        # (patch_size, merge_size, mean/std). A split/quantized checkpoint may
        # only contain tokenizer files (no preprocessor_config.json), in which
        # case AutoProcessor returns a bare tokenizer with no .image_processor;
        # fall back to the base HF repo, which carries the same static metadata.
        processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            hf_repo = getattr(self, "_hf_repo_name", None)
            assert hf_repo is not None, (
                "Checkpoint has no image processor and no _hf_repo_name fallback."
            )
            image_processor = AutoProcessor.from_pretrained(
                hf_repo, trust_remote_code=True
            ).image_processor
        return image_processor

    def _extract_embedding_weights(
        self, full_vlm: torch.nn.Module | None
    ) -> torch.Tensor | None:
        if full_vlm is None:
            return None
        return full_vlm.get_input_embeddings().weight.data.clone()  # type: ignore[operator]

    def _post_configure_tokenizer_and_config(self) -> None:
        return None

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        load_pretrained: bool = True,
        is_token_generator: bool = False,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
        _skip_optimizations: list[str] | None = None,
    ) -> None:
        """
        Initialize Qwen3-VL text model.

        Overrides parent to load from full VLM checkpoint and extract text model.
        """
        # Initialize nn.Module first to set up 'training' attribute
        torch.nn.Module.__init__(self)

        if host_device is None:
            host_device = torch.device("cpu")

        self.skip_optimizations = _skip_optimizations
        self.checkpoint = checkpoint

        has_recommended_memory(self.min_memory_recommended)

        self._apply_monkey_patch()
        llm_config = get_vlm_config(self.checkpoint)
        # Keep original config for full VLM operations
        self._original_llm_config = llm_config
        self.llm_config = self.edit_llm_config(llm_config)
        self._verify_ckpt()
        self.tokenizer = self._load_tokenizer(checkpoint)
        self._post_configure_tokenizer_and_config()
        self._image_processor = self._load_image_processor(checkpoint)

        # Load model using our custom loader
        model, full_vlm, lm_head = self.load_llm_from_checkpoint(
            checkpoint=self.checkpoint,
            llm_config=self.llm_config,
            load_pretrained=load_pretrained,
        )
        model.eval()

        # Extract and store embedding weights before discarding full VLM
        self._embedding_weights = self._extract_embedding_weights(full_vlm)

        # Create embedding (use original config for vocab_size)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length,
            config=cast(LlamaConfig, self.llm_config),
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for _, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()

        # Convert lm_head to Conv2d (not part of model.named_modules())
        from qai_hub_models.models.templates.llm.model_adaptations import (
            ConvInplaceLinear,
        )

        if isinstance(lm_head, torch.nn.Linear):
            lm_head = ConvInplaceLinear(lm_head)

        # Wrap text_model + lm_head into a single Module
        assert lm_head is not None
        wrapper = _VLMCausalLMWrapper(model, lm_head)
        wrapper.to(host_device).float()

        self.sequence_length: int = sequence_length
        self.context_length: int = context_length
        self.split_part = 1
        self.is_token_generator = is_token_generator
        self.model = wrapper
        self.attention_mask_min_clip = attention_mask_min_clip
        self.attention_mask_multiplier = attention_mask_multiplier

    @classmethod
    def load_llm_from_checkpoint(
        cls,
        checkpoint: str | os.PathLike | Path,
        llm_config: PretrainedConfig,
        load_pretrained: bool = True,
    ) -> tuple[torch.nn.Module, torch.nn.Module | None, torch.nn.Module | None]:
        """
        Load the text model from a Qwen3-VL checkpoint.

        Returns (text_model, full_vlm, lm_head) tuple. The full_vlm is kept for
        embedding table extraction. The lm_head is needed for logits computation.

        Qwen3-VL hierarchy: model.model (Qwen3VLModel) contains .language_model
        (Qwen3VLTextModel) and .visual (Qwen3VLVisionModel).
        """
        if load_pretrained:
            full_vlm = (
                modeling_qwen3_vl.Qwen3VLForConditionalGeneration.from_pretrained(
                    checkpoint,
                    attn_implementation="eager",
                )
            )
            # Extract the text model (language_model inside model)
            text_model = full_vlm.model.language_model
            lm_head = full_vlm.lm_head
            return text_model, full_vlm, lm_head
        # Create uninitialized text model
        text_model = modeling_qwen3_vl.Qwen3VLTextModel(llm_config)  # type: ignore[arg-type, unused-ignore]
        lm_head = torch.nn.Linear(
            llm_config.hidden_size, llm_config.vocab_size, bias=False
        )
        return text_model, None, lm_head

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    def get_embedding_weights(self) -> torch.Tensor:
        """Get embedding weights from the stored weights or text model."""
        if self._embedding_weights is not None:
            return self._embedding_weights
        text_model = self.model.model if hasattr(self.model, "model") else self.model
        return text_model.embed_tokens.weight.data  # type: ignore[union-attr, return-value]

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings using the embedding table."""
        embedding_weights = self.get_embedding_weights().to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, embedding_weights)

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Override to extract deepstack inputs from *args and pass to model."""
        from transformers.cache_utils import DynamicCache

        from qai_hub_models.models.templates.llm.model import (  # type: ignore[attr-defined]
            SHADynamicCacheNewValueOnly,
        )

        # args layout: (cos, sin, *kv_caches, [visual_pos_masks, *deepstack_embeds])
        position_ids = args[:2]
        num_kv_tensors = self.llm_config.num_hidden_layers * 2
        past_key_values_tensors = args[2 : 2 + num_kv_tensors]
        extra_args = args[2 + num_kv_tensors :]

        # Extract deepstack inputs if present
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if len(extra_args) > 0:
            visual_pos_masks = extra_args[0]
            if len(extra_args) > 1:
                deepstack_visual_embeds = list(extra_args[1:])

        # Build KV cache
        assert isinstance(self.llm_config.num_key_value_heads, int)
        if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
            kv_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(
                zip(
                    past_key_values_tensors[::2],
                    past_key_values_tensors[1::2],
                    strict=False,
                )
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                k = torch.cat(k_split, dim=1).permute(0, 1, 3, 2)
                v = torch.cat(v_split, dim=1)
                kv_cache.update(k, v, layer_idx, {})
        else:
            kv_cache = SHADynamicCacheNewValueOnly()
            for layer_idx, (k, v) in enumerate(
                zip(
                    past_key_values_tensors[::2],
                    past_key_values_tensors[1::2],
                    strict=False,
                )
            ):
                k_split = [
                    k[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                v_split = [
                    v[i : i + 1] for i in range(self.llm_config.num_key_value_heads)
                ]
                kv_cache.update(k_split, v_split, layer_idx, {})

        model_kwargs: dict[str, Any] = {
            self.main_input_name: input_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
        }
        if visual_pos_masks is not None:
            model_kwargs["visual_pos_masks"] = visual_pos_masks
        if deepstack_visual_embeds is not None:
            model_kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds

        out = self.model(**model_kwargs)

        out_cache = out["past_key_values"]
        flat_output_past_key_values = []
        for layer in range(len(out_cache)):
            if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
                if hasattr(out_cache, "key_cache"):
                    keys = out_cache.key_cache[layer]
                    values = out_cache.value_cache[layer]
                elif hasattr(out_cache.layers[layer], "keys"):
                    keys = out_cache.layers[layer].keys
                    values = out_cache.layers[layer].values
                else:
                    keys = out_cache.layers[layer][0]
                    values = out_cache.layers[layer][1]

                seq_len = input_tokens.shape[1]
                k = keys[:, :, -seq_len:, :].permute(1, 0, 3, 2)
                v = values[:, :, -seq_len:, :].permute(1, 0, 2, 3)

            elif hasattr(out_cache, "key_cache"):
                k = torch.cat(out_cache.key_cache[layer], dim=0)
                v = torch.cat(out_cache.value_cache[layer], dim=0)
            elif hasattr(out_cache.layers[layer], "keys"):
                k = torch.cat(out_cache.layers[layer].keys, dim=0)
                v = torch.cat(out_cache.layers[layer].values, dim=0)
            else:
                k = torch.cat(out_cache.layers[layer][0], dim=0)
                v = torch.cat(out_cache.layers[layer][1], dim=0)
            flat_output_past_key_values += [k, v]

        return [out["logits"], *flat_output_past_key_values]

    @staticmethod
    def get_input_prompt_with_tags(  # type: ignore[override]
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        include_image: bool | int = False,
        enable_thinking: bool = False,
        tokenizer: PreTrainedTokenizer | None = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Format a prompt with appropriate tags for Qwen3-VL.

        Overrides the base class to use Qwen3-VL's ChatML format and
        include vision placeholder tokens when processing images.
        Uses the tokenizer's chat template with structured content for images.

        Parameters
        ----------
        user_input_prompt
            The user's text prompt. Defaults to DEFAULT_USER_PROMPT.
        system_context_prompt
            System context/instructions. Defaults to DEFAULT_PROMPT_CONTEXT.
        include_image
            Whether to include vision placeholder tokens in the prompt.
            Pass ``True`` or ``1`` for a single image, an ``int > 1`` for
            multiple images, or ``False``/``0`` for text-only.
            Defaults to False.
        enable_thinking
            Whether to enable thinking mode.
            Defaults to False.
        tokenizer
            Required. The tokenizer to use for applying the chat template.
        add_generation_prompt
            Whether to append the assistant turn header.
            Defaults to True.
        continue_final_message
            Whether to continue the final message instead of starting a new one.
            Defaults to False.
        **kwargs
            Additional arguments (ignored, for compatibility with base class).

        Returns
        -------
        str
            Formatted prompt string with ChatML tags and optional
            vision placeholders.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required for get_input_prompt_with_tags")
        if user_input_prompt is None:
            user_input_prompt = DEFAULT_USER_PROMPT
        if system_context_prompt is None:
            system_context_prompt = DEFAULT_PROMPT_CONTEXT

        num_images = int(include_image) if isinstance(include_image, (bool, int)) else 0
        if num_images > 0:
            content: list[dict[str, str]] = [
                {"type": "image"} for _ in range(num_images)
            ]
            content.append({"type": "text", "text": user_input_prompt})
        else:
            content = [{"type": "text", "text": user_input_prompt}]

        messages: list[dict[str, Any]] = []
        if system_context_prompt:
            messages.append({"role": "system", "content": system_context_prompt})
        messages.append({"role": "user", "content": content})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking,
        )
        assert isinstance(prompt, str)
        return prompt

    @classmethod
    def get_image_placeholder_for_processor(
        cls, processor: VLMProcessorLike | None = None
    ) -> str:
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def _verify_ckpt(self) -> None:
        """Verify checkpoint is compatible with Qwen3-VL."""
        valid_model_types = {"qwen3_vl", "qwen3", "qwen3_vl_text"}
        architectures = getattr(self.llm_config, "architectures", None) or []
        if not (
            self.llm_config.model_type in valid_model_types
            or any("Qwen3" in arch for arch in architectures)
        ):
            raise ValueError(
                "Model config is not compatible with Qwen3-VL implementation. "
                f"Expected model_type in {valid_model_types}, got '{self.llm_config.model_type}'"
            )

    @staticmethod
    def monkey_patch(skip_optimizations: list[str] | None = None) -> None:
        """
        Apply monkey patches for Qwen3-VL ONNX export.

        Adaptations applied:
        - SHA (Split-Head Attention) for Qwen3VLTextAttention
        - MLP Conv2d (down_proj)
        - Bypass rotary embeddings (cos/sin pre-computed externally)
        - Export-friendly _deepstack_process (avoids boolean indexing)
        """
        from qai_hub_models.models.templates.qwen3.model import Qwen3_Optimizations
        from qai_hub_models.models.templates.qwen3_vl.model_adaptations import (
            QCQwen3VLTextMLP,
            SHAQwen3VLTextAttention,
        )

        # SHA attention
        if (
            skip_optimizations
            and Qwen3_Optimizations.SHA_ATTENTION in skip_optimizations
        ):
            print("Skip sha_attention optimization")
        else:
            modeling_qwen3_vl.Qwen3VLTextAttention = SHAQwen3VLTextAttention  # type: ignore[misc, unused-ignore]

        # Bypass rotary embedding module
        def bypass_RotaryEmbedding(
            self: torch.nn.Module,
            x: torch.Tensor,
            position_ids: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            return position_ids

        if not hasattr(
            modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding, "_original_forward"
        ):
            modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding.forward
            )
            modeling_qwen3_vl.Qwen3VLTextRotaryEmbedding.forward = (
                bypass_RotaryEmbedding  # type: ignore[assignment, unused-ignore]
            )

        # Patch Qwen3VLTextModel.forward to handle tuple position_ids
        # (pre-computed cos/sin from bypass_RotaryEmbedding) and to use
        # an export-friendly _deepstack_process.
        _original_text_forward = modeling_qwen3_vl.Qwen3VLTextModel.forward

        def _exportable_deepstack_process(
            hidden_states: torch.Tensor,
            visual_pos_masks: torch.Tensor,
            visual_embeds: torch.Tensor,
        ) -> torch.Tensor:
            """Export-friendly deepstack: avoids boolean indexing (dynamic shapes)."""
            visual_pos_masks = visual_pos_masks.to(hidden_states.device)
            visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
            # Use float mask broadcasting instead of boolean indexing
            mask_float = visual_pos_masks.unsqueeze(-1).float()
            # Scatter visual_embeds into the positions marked by the mask.
            # cumsum gives a 1-based index into visual_embeds for each True position.
            indices = (visual_pos_masks.long().cumsum(dim=-1) - 1).clamp(min=0)
            indices = indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            gathered = torch.gather(
                visual_embeds.unsqueeze(0).expand(hidden_states.shape[0], -1, -1),
                dim=1,
                index=indices,
            )
            return hidden_states + gathered * mask_float

        def _patched_text_forward(
            self: Any,
            input_ids: Any = None,
            attention_mask: Any = None,
            position_ids: Any = None,
            past_key_values: Any = None,
            inputs_embeds: Any = None,
            use_cache: Any = None,
            visual_pos_masks: Any = None,
            deepstack_visual_embeds: Any = None,
            **kwargs: Any,
        ) -> Any:
            if isinstance(position_ids, tuple):
                # Pre-computed (cos, sin) — skip HF's ndim processing.
                from transformers.modeling_outputs import BaseModelOutputWithPast

                use_cache = (
                    use_cache if use_cache is not None else self.config.use_cache
                )

                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)

                # position_embeddings = (cos, sin) from bypass_RotaryEmbedding
                position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

                hidden_states = inputs_embeds

                for layer_idx, decoder_layer in enumerate(self.layers):
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=None,
                        past_key_values=past_key_values,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs

                    # Deepstack: inject visual embeddings at early layers
                    if (
                        deepstack_visual_embeds is not None
                        and visual_pos_masks is not None
                        and layer_idx < len(deepstack_visual_embeds)
                    ):
                        hidden_states = _exportable_deepstack_process(
                            hidden_states,
                            visual_pos_masks,
                            deepstack_visual_embeds[layer_idx],
                        )

                hidden_states = self.norm(hidden_states)

                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                )

            return _original_text_forward(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        if not hasattr(modeling_qwen3_vl.Qwen3VLTextModel, "_original_forward"):
            modeling_qwen3_vl.Qwen3VLTextModel._original_forward = (  # type: ignore[attr-defined, unused-ignore]
                _original_text_forward
            )
            modeling_qwen3_vl.Qwen3VLTextModel.forward = _patched_text_forward

        # MLP Conv2d adaptation
        modeling_qwen3_vl.Qwen3VLTextMLP = QCQwen3VLTextMLP  # type: ignore[misc, unused-ignore]


class Qwen3VLTextBase_AIMETOnnx(Qwen3Base_AIMETOnnx):
    """
    AIMET-ONNX quantized version of Qwen3-VL text model.

    Uses inputs_embeds instead of input_ids.
    """

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Qwen3VLTextBase  # type: ignore[assignment]

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    get_input_prompt_with_tags = staticmethod(
        Qwen3VLTextBase.get_input_prompt_with_tags
    )

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

        # Full VLM config (needed by the generator for vision-related fields)
        hf_repo = getattr(self, "_hf_repo_name", None) or (
            str(checkpoint) if checkpoint else None
        )
        assert hf_repo is not None
        self._original_llm_config = get_vlm_config(hf_repo)

        # Load embedding weights from checkpoint for VLM models.
        self._embedding_weights = None
        if checkpoint is not None:
            embed_path = Path(checkpoint) / "embedding_weights.raw"
            if embed_path.exists():
                import numpy as np

                embed_np = np.fromfile(str(embed_path), dtype=np.float32)
                vocab_size = self.llm_config.vocab_size
                hidden_size = self.llm_config.hidden_size
                self._embedding_weights = torch.from_numpy(
                    embed_np.reshape(vocab_size, hidden_size)
                )
        self._post_init_vlm_aimet()

    def _post_init_vlm_aimet(self) -> None:
        """Family-specific AIMET init hook. Default no-op."""
        return

    def get_embedding_weights(self) -> torch.Tensor:
        """Get embedding weights from checkpoint (not from LM head)."""
        if self._embedding_weights is not None:
            return self._embedding_weights
        raise RuntimeError(
            "VLM embedding weights not loaded. Ensure checkpoint contains "
            "embedding_weights.raw or pass an FP model during from_pretrained."
        )

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings using the stored embedding table."""
        embedding_weights = self.get_embedding_weights().to(input_ids.device)
        return torch.nn.functional.embedding(input_ids, embedding_weights)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        """
        Adapt AIMET encodings for VLM model.

        VLM models use inputs_embeds instead of input_ids, so the embedding
        layer (embed_tokens) is not part of the exported ONNX model.
        """
        from qai_hub_models.utils.aimet.encodings import propagate_memory_encodings

        with open(src_encodings_path) as read_file:
            encodings = json.load(read_file)

        model = onnx.load(onnx_model_path)

        # Convert encodings to dictionaries for faster look-ups
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings["activation_encodings"]
        }
        encodings["param_encodings"] = {
            v["name"]: v for v in encodings["param_encodings"]
        }

        # Copy weight encodings to param encodings
        for key in encodings["activation_encodings"]:
            if "weight" in key:
                encodings["param_encodings"][key] = copy.deepcopy(
                    encodings["activation_encodings"][key]
                )

        propagate_memory_encodings(encodings, model)

        # convert back
        encodings["activation_encodings"] = list(
            encodings["activation_encodings"].values()
        )
        encodings["param_encodings"] = list(encodings["param_encodings"].values())

        with open(dst_encodings_path, "w") as write_file:
            json.dump(encodings, write_file, indent=4, sort_keys=True)

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
        if bundle.aimet_encodings_path is not None:
            encodings_path = str(bundle.aimet_encodings_path)
            self._adapt_aimet_encodings(
                encodings_path, encodings_path, str(bundle.onnx_graph_path)
            )
        return bundle


class Qwen3VLDynamic_AIMETOnnx(VLMDynamic_AIMETOnnx, Qwen3VLTextBase_AIMETOnnx):
    """Dynamic-shape variant of Qwen3VLTextBase_AIMETOnnx.

    Inherits the VLM calibration / weight-optimization data pipeline from
    VLMDynamic_AIMETOnnx; only model-specific config lives here.
    """

    FPModel = Qwen3VLTextBase

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return _vlm_eval_dataset_classes()


class Qwen3VLTextBase_QNN(Qwen3Base_QNN):
    """QNN version of Qwen3-VL text model."""

    llm_io_type: LLMIOType = LLMIOType.genie_input_embeds

    FPModel = Qwen3VLTextBase  # type: ignore[assignment]

    @property
    def main_input_name(self) -> str:
        """Override to use 'inputs_embeds' (HuggingFace naming with 's')."""
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "inputs_embeds"
        return "input_ids"

    get_input_prompt_with_tags = staticmethod(
        Qwen3VLTextBase.get_input_prompt_with_tags
    )


# Re-export position processor
Qwen3VLPositionProcessor = Qwen3PositionProcessor


# ---------------------------------------------------------------------------
# Qwen3VLPreSplitBase - FP PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Qwen3VLPreSplitBase(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Qwen3VLTextBase
):
    """FP PreSplit base for Qwen3-VL models.

    Manages the full torch model and ONNX splitting. Uses class-level cache
    keyed by checkpoint. VLM uses split_embedding=False since inputs_embeds
    bypasses the embedding layer.

    Concrete subclasses set the architecture constants below.
    """

    GeneratorClass: type[VLM_Generator] = HubCompatibleQwen3VLGenerator

    # --- per-model configuration (override in subclass) ---
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict = {}
    default_precision: Precision = Precision.w4a16
    min_memory_recommended: int = 0
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0
    split_embedding = False

    num_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    num_deepstack_layers: int = 0

    vision_patch_size: int = 0
    spatial_merge_size: int = 0
    default_num_visual_tokens: int = 0

    _hf_repo_name: str = ""

    @classmethod
    def num_visual_tokens_for_image_size(cls, image_size: tuple[int, int]) -> int:
        """Post-merge visual token count for an image: (W/patch)*(H/patch)/merge^2.

        ``image_size`` is ``(width, height)`` to match the dataset/eval convention
        (PIL ``Image.resize`` takes ``(width, height)``).
        """
        width, height = image_size
        return (
            (height // cls.vision_patch_size)
            * (width // cls.vision_patch_size)
            // (cls.spatial_merge_size * cls.spatial_merge_size)
        )

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision = Precision.w4a16,
    ) -> tuple[float | None, float]:
        return (-250.0, 1.0)

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint or self._hf_repo_name, **kwargs)

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        text_config = self.llm_config
        if hasattr(self.llm_config, "text_config"):
            text_config = self.llm_config.text_config
        if not (
            text_config.num_hidden_layers == self.num_layers
            and text_config.hidden_size == self.hidden_size
            and text_config.num_attention_heads == self.num_attention_heads
            and text_config.num_key_value_heads == self.num_key_value_heads
        ):
            raise ValueError("Model config is not compatible with our implementation.")

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        checkpoint: str | Path | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        host_device: torch.device | None = None,
        _skip_optimizations: list[str] | None = None,
    ) -> Self:
        checkpoint = checkpoint or cls._hf_repo_name
        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached

        attention_mask_min_clip, _ = cls.attention_mask_min_clip_and_multiplier()

        try:
            instance = cls(
                checkpoint=checkpoint,
                sequence_length=sequence_length,
                context_length=context_length,
                host_device=host_device,
                load_pretrained=True,
                attention_mask_min_clip=attention_mask_min_clip,
                _skip_optimizations=_skip_optimizations,
            )
        except Exception:
            cls.release()
            raise
        cls.cache_store(instance, cache_key)
        return instance

    def get_output_spec(self) -> OutputSpec:
        return Qwen3VLTextBase._get_output_spec(self.num_layers)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        return self.get_static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type, image_size
        )

    @classmethod
    def get_static_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        if llm_config is None:
            llm_config = {
                "num_hidden_layers": cls.num_layers,
                "hidden_size": cls.hidden_size,
                "num_key_value_heads": cls.num_key_value_heads,
                "num_attention_heads": cls.num_attention_heads,
            }
        return Qwen3VLTextBase._get_input_spec(
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
            head_dim=llm_config.get("head_dim", cls.head_dim),
            llm_io_type=llm_io_type,
            num_deepstack_layers=cls.num_deepstack_layers,
            num_visual_tokens=cls.num_visual_tokens_for_image_size(image_size),
        )

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Export full ONNX from PyTorch with dynamic shapes."""
        from torch.export import Dim

        seq_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]
        num_visual_tokens = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]

        extra_dynamic_shapes: dict[str, dict[int, Any]] = {
            "visual_pos_masks": {1: seq_len},
        }
        for i in range(self.num_deepstack_layers):
            extra_dynamic_shapes[f"deepstack_visual_embeds_{i}"] = {
                0: num_visual_tokens
            }

        onnx_dir = temp_path / "full_dynamic"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "model.onnx"
        get_onnx_model(
            fp_model=self,
            context_length=DEFAULT_CONTEXT_LENGTH,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            path=str(onnx_path),
            return_model=False,
            llm_io_type=self.llm_io_type,
            extra_dynamic_shapes=extra_dynamic_shapes,
        )
        return ONNXBundle.from_bundle_path(onnx_dir, "model")


Qwen3VLPreSplitT = TypeVar("Qwen3VLPreSplitT", bound=Qwen3VLPreSplitBase)


class VisionEncoderExportProtocol(Protocol):
    default_image_height: int
    default_image_width: int
    _pos_emb_cos: torch.Tensor
    _pos_emb_sin: torch.Tensor
    _window_attention_mask: torch.Tensor
    _full_attention_mask: torch.Tensor

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        device: torch.device | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        precision: Precision = Precision.float,
        **kwargs: Any,
    ) -> VisionEncoderExportProtocol: ...

    @classmethod
    def export_to_onnx(
        cls,
        veg_model: Any,
        host_device: torch.device,
    ) -> onnx.ModelProto: ...

    @classmethod
    def save_onnx(
        cls,
        onnx_model: onnx.ModelProto,
        output_dir: str | os.PathLike | Path,
        filename: str = "vision_encoder.onnx",
    ) -> Path: ...

    def eval(self) -> VisionEncoderExportProtocol: ...


class VisionEncoderCollectionProtocol(Protocol):
    default_image_height: int
    default_image_width: int
    _pos_emb_cos: torch.Tensor
    _pos_emb_sin: torch.Tensor
    _window_attention_mask: torch.Tensor
    _full_attention_mask: torch.Tensor

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        device: torch.device | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        precision: Precision = Precision.float,
        **kwargs: Any,
    ) -> VisionEncoderCollectionProtocol: ...

    def eval(self) -> VisionEncoderCollectionProtocol: ...


# ---------------------------------------------------------------------------
# Qwen3VLQuantizablePreSplitBase - Quantizable PreSplit with class-level cache
# ---------------------------------------------------------------------------


class Qwen3VLQuantizablePreSplitBase(  # type: ignore[misc]
    DynamicQuantizablePreSplitMixin[Qwen3VLPreSplitT],
    Qwen3VLDynamic_AIMETOnnx,
    Generic[Qwen3VLPreSplitT],
):
    """Quantizable PreSplit base for Qwen3-VL models.

    The S3 asset zip contains the FULL output of quantize.py (dynamic
    ONNX + weights + encodings + tokenizer + config + embedding_weights.raw).

    Concrete subclasses set ``FPModel`` and the config attributes below.
    """

    FPModel: type[Qwen3VLPreSplitT]
    _hf_repo_name: str = ""

    # DynamicQuantizablePreSplitMixin config
    model_id: str = ""
    model_asset_version: int = 0
    default_checkpoint: dict = {}
    supported_precisions: list[Precision] = []
    default_precision: Precision = Precision.w4a16

    # DynamicPreSplitOnnxMixin config
    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0
    split_embedding = False

    num_layers: int = 0

    # SHA produces per-head q_norm/k_norm nodes in the ONNX graph.
    # Between block starts (input_layernorm): num_attention_heads q_norms
    # + num_key_value_heads k_norms + 1 post_attention_layernorm intermediate ops
    ada_scale_num_rmsnorm_per_blk: int | None = None

    # VLM: vision encoder class (set by leaf classes)
    vision_encoder_cls: type[VisionEncoderExportProtocol] | None = None

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        return (-100, 1.0)

    def get_output_spec(self) -> OutputSpec:
        return Qwen3VLTextBase._get_output_spec(self.num_layers)

    def get_input_spec(
        self,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        return self.get_static_input_spec(
            llm_config, sequence_length, context_length, llm_io_type, image_size
        )

    @classmethod
    def get_static_input_spec(
        cls,
        llm_config: dict | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        llm_io_type: LLMIOType = LLMIOType.genie_input_embeds,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ) -> InputSpec:
        return cls.FPModel.get_static_input_spec(
            llm_config=llm_config,
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=llm_io_type,
            image_size=image_size,
        )

    def save_calibrated_checkpoint(  # type: ignore[override]
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: Qwen3VLPreSplitT | None = None,
    ) -> None:
        """Save calibrated checkpoint with ONNX, encodings, and embedding weights."""
        if fp_model is None:
            fp_model = self.FPModel.from_pretrained()
        super().save_calibrated_checkpoint(output_checkpoint, fp_model)

        # VLM-specific: embedding table is needed for on-device LUT encoder.
        # Skip if already present (e.g. pre-exported and SpinQuant-rotated).
        embedding_path = Path(output_checkpoint) / "embedding_weights.raw"
        if not embedding_path.exists():
            export_embedding_weights_from_tensor(
                fp_model.get_embedding_weights().float(), Path(output_checkpoint)
            )

    @classmethod
    def _has_onnx_on_disk(
        cls, ckpt: Path, sequence_length: int, context_length: int
    ) -> bool:
        """Require backbone + VEG ONNX + embedding for a complete VLM export.

        export_onnx / apply_pre_sim_transforms produce vision_encoder.onnx and
        embedding_weights.raw alongside the backbone. A run interrupted after
        the backbone is written but before (or during) the VEG/embedding write
        must not be treated as complete, or quantize() would skip re-export and
        SpinQuant, leaving those artifacts stale/missing and inconsistent with
        the rotated backbone.
        """
        if not super()._has_onnx_on_disk(ckpt, sequence_length, context_length):
            return False
        veg_complete = (ckpt / "vision_encoder.onnx").exists()
        embedding_complete = (ckpt / "embedding_weights.raw").exists()
        return veg_complete and embedding_complete

    @classmethod
    def export_onnx(
        cls,
        fp_model: Qwen3VLPreSplitT,  # type: ignore[override]
        output_dir: Path,
        **kwargs: Any,
    ) -> None:
        """Export backbone ONNX, VEG ONNX, and embedding weights."""
        # Backbone (delegate to parent)
        super().export_onnx(fp_model, output_dir)

        # Embedding weights (external to the ONNX graph for VLMs)
        export_embedding_weights_from_tensor(
            fp_model.get_embedding_weights().float(), output_dir
        )

        # VEG ONNX
        assert cls.vision_encoder_cls is not None
        host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_height = kwargs.get(
            "image_height", cls.vision_encoder_cls.default_image_height
        )
        image_width = kwargs.get(
            "image_width", cls.vision_encoder_cls.default_image_width
        )
        veg_model = cls.vision_encoder_cls.from_pretrained(
            device=host_device,
            image_height=image_height,
            image_width=image_width,
        )
        veg_model.eval()
        veg_onnx = cls.vision_encoder_cls.export_to_onnx(veg_model, host_device)
        cls.vision_encoder_cls.save_onnx(veg_onnx, str(output_dir))

        del veg_model, veg_onnx
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def apply_pre_sim_transforms(
        cls,
        output_dir: Path,
        spinquant_config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Apply SpinQuant co-rotating backbone + VEG + embedding."""
        if not spinquant_config:
            return

        # Load backbone
        backbone_path = output_dir / "model_dynamic.onnx"
        backbone_onnx = onnx.load(str(backbone_path), load_external_data=True)

        # Load VEG
        veg_onnx = onnx.load(
            str(output_dir / "vision_encoder.onnx"), load_external_data=True
        )

        # Load embedding
        hidden_size = cls.FPModel.hidden_size
        embedding_np = np.fromfile(
            str(output_dir / "embedding_weights.raw"), dtype=np.float32
        )
        embedding = torch.from_numpy(embedding_np.reshape(-1, hidden_size))

        # Co-rotate all three
        cls.apply_spinquant_to_onnx(
            backbone_onnx,
            spinquant_config,
            visual_model=veg_onnx,
            embedding=embedding,
        )

        # Save rotated backbone
        onnx.save_model(
            backbone_onnx,
            str(backbone_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

        # Save rotated VEG
        assert cls.vision_encoder_cls is not None
        cls.vision_encoder_cls.save_onnx(veg_onnx, str(output_dir))

        # Save rotated embedding
        export_embedding_weights_from_tensor(embedding, output_dir)

        del backbone_onnx, veg_onnx, embedding
        gc.collect()

        print("  SpinQuant complete.")


# ---------------------------------------------------------------------------
# Vision Encoder Component
# ---------------------------------------------------------------------------


class Qwen3VLVisionEncoderBase(Qwen3VLVisionEncoder):
    """Vision encoder base for Qwen3-VL (adapted VEG for on-device deployment).

    Returns multiple outputs: image_embeddings + deepstack features.
    Supports both FP inference and quantized inference (via AIMET-ONNX QuantSim).

    Concrete subclasses set ``_hf_repo_name``, ``vision_patch_size``,
    ``vision_hidden_size``, ``vision_num_heads``,
    ``quant_presplit_cls``, ``default_image_height``/``default_image_width`` and
    ``DEFAULT_IMAGE_SIZE``.
    """

    _hf_repo_name: str = ""
    vision_patch_size: int = 0
    # Vision-tower attention dims, used to derive the RoPE embedding width
    # (rope_dim = (vision_hidden_size // vision_num_heads) // 2). These differ
    # per model (e.g. 8B: 1152/16 -> head_dim 72 -> rope_dim 36; 4B: 1280/16 ->
    # head_dim 80 -> rope_dim 40), so they must be set per subclass rather than
    # relying on a hardcoded default.
    vision_hidden_size: int = 0
    vision_num_heads: int = 0
    default_image_height: int = 0
    default_image_width: int = 0
    # Set by subclass to the model's Quantizable PreSplit class.
    quant_presplit_cls: type[Qwen3VLQuantizablePreSplitBase]

    @classmethod
    def vision_rope_dim(cls) -> int:
        """RoPE embedding width for the vision tower = head_dim // 2.

        Derived from config-backed class attributes so each model variant gets
        the correct width (vs. a hardcoded default that only matched one model).
        """
        if not cls.vision_hidden_size or not cls.vision_num_heads:
            raise ValueError(
                f"{cls.__name__} must set vision_hidden_size and vision_num_heads "
                f"to derive the vision RoPE dim."
            )
        head_dim = cls.vision_hidden_size // cls.vision_num_heads
        return head_dim // 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._checkpoint: str | None = None
        self._precision: Precision = Precision.float
        self._quantized_session: Any | None = None

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        device: torch.device | None = None,
        image_height: int | None = None,
        image_width: int | None = None,
        precision: Precision = Precision.float,
        **kwargs: Any,
    ) -> Self:
        if image_height is None:
            image_height = cls.default_image_height
        if image_width is None:
            image_width = cls.default_image_width
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if precision != Precision.float and (
            isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
        ):
            checkpoint = cls.quant_presplit_cls.fetch_default_checkpoint(precision)

        load_device = device if precision == Precision.float else torch.device("cpu")
        instance: Self = super().from_pretrained(  # type: ignore[assignment]
            checkpoint=cls._hf_repo_name,
            device=load_device,
            image_height=image_height,
            image_width=image_width,
        )
        instance._checkpoint = str(checkpoint)
        instance._precision = precision

        if precision != Precision.float:
            instance._init_quantized_session(Path(str(checkpoint)), device)

        return instance

    def _init_quantized_session(
        self,
        ckpt_path: Path,
        device: torch.device,
    ) -> None:
        """Create an AIMET-ONNX QuantSim session for quantized inference."""
        veg_onnx = ckpt_path / "vision_encoder.onnx"
        veg_enc = ckpt_path / "vision_encoder.encodings"

        onnx_model = onnx.load(str(veg_onnx), load_external_data=True)

        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.insert(0, "CUDAExecutionProvider")

        quant_logger = logging.getLogger("Quant")
        prev_level = quant_logger.level
        quant_logger.setLevel(logging.WARNING)
        try:
            quant_sim = QuantizationSimModel(
                model=onnx_model,
                quant_scheme=QuantScheme.min_max,
                param_type="int8",
                activation_type="int16",
                providers=providers,
            )
            if veg_enc.exists():
                load_encodings_to_sim(quant_sim, str(veg_enc), strict=False)
        finally:
            quant_logger.setLevel(prev_level)

        self._quantized_session = quant_sim

    def component_precision(self) -> Precision:
        return self._precision

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    def forward(
        self,
        pixel_values: torch.Tensor,
        position_ids_cos: torch.Tensor | None = None,
        position_ids_sin: torch.Tensor | None = None,
        window_attention_mask: torch.Tensor | None = None,
        full_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self._is_quantized:
            return self._forward_quantized(pixel_values)
        return super().forward(
            pixel_values=pixel_values,
            position_ids_cos=position_ids_cos,
            position_ids_sin=position_ids_sin,
            window_attention_mask=window_attention_mask,
            full_attention_mask=full_attention_mask,
        )

    def _forward_quantized(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Run inference through the AIMET-ONNX QuantSim session."""
        assert self._quantized_session is not None
        results = mock_torch_onnx_inference(
            self._quantized_session.session,
            pixel_values,
            cast(torch.Tensor, self._pos_emb_cos),
            cast(torch.Tensor, self._pos_emb_sin),
            cast(torch.Tensor, self._window_attention_mask),
            cast(torch.Tensor, self._full_attention_mask),
        )
        if isinstance(results, torch.Tensor):
            return (results,)
        return tuple(results)

    def get_input_spec(
        self,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> InputSpec:
        if image_height is None:
            image_height = self.default_image_height
        if image_width is None:
            image_width = self.default_image_width
        return self.get_static_input_spec(image_height, image_width)

    @classmethod
    def get_static_input_spec(  # type: ignore[override]
        cls,
        image_height: int | None = None,
        image_width: int | None = None,
    ) -> InputSpec:
        if image_height is None:
            image_height = cls.default_image_height
        if image_width is None:
            image_width = cls.default_image_width
        return Qwen3VLVisionEncoder.get_static_input_spec(
            image_height=image_height,
            image_width=image_width,
            patch_size=cls.vision_patch_size,
            rope_dim=cls.vision_rope_dim(),
        )

    def _get_onnx_bundle(self) -> ONNXBundle:
        if self._checkpoint is None:
            raise ValueError("No checkpoint provided for VisionEncoder.")
        ckpt = Path(self._checkpoint)
        return ONNXBundle(
            bundle_path=ckpt,
            onnx_graph_name="vision_encoder.onnx",
            onnx_weights_name="vision_encoder.data"
            if (ckpt / "vision_encoder.data").exists()
            else None,
            aimet_encodings_name="vision_encoder.encodings"
            if (ckpt / "vision_encoder.encodings").exists()
            else None,
        )

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        out_dir = Path(output_dir) / f"{model_name}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )
        return out_dir

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        spec = input_spec or self.get_input_spec()
        result: SampleInputsType = {}
        for name, (shape, dtype_str) in spec.items():
            np_dtype = np.float32 if dtype_str == "float32" else np.int64
            result[name] = [np.zeros(shape, dtype=np_dtype)]
        return result


# ---------------------------------------------------------------------------
# Unified Part Base
# ---------------------------------------------------------------------------


class Qwen3VLPartBase(LLMPartBase, torch.nn.Module, MultiGraphWorkbenchModel):
    """Unified Part base: handles both FP and Quantizable modes based on precision.

    Spec derivation is inherited from ``LLMPartBase`` (head_dim attribute +
    ``_extra_graph_inputs`` hook); this class carries the family deploy/session
    plumbing (mirroring ``LlamaPartBase`` for text LLMs) plus the qwen3
    architecture constants and the deepstack graph-input override.

    Concrete subclasses set the architecture constants, the FP / Quantizable
    PreSplit classes, the export length lists, and ``part_id``.
    """

    # Architecture dims (LLMPartBase attribute names; head_dim is explicit
    # because hidden_size / num_attention_heads may not equal head_dim).
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    num_splits: int = 0
    num_deepstack_layers: int = 0
    default_precision: Precision = Precision.w4a16
    default_num_visual_tokens: int = 0
    part_id: int = 0

    # Set by subclass.
    fp_presplit_cls: type[Qwen3VLPreSplitBase]
    quant_presplit_cls: type[Qwen3VLQuantizablePreSplitBase]
    export_sequence_lengths: list[int] = []
    export_context_lengths: list[int] = []

    def __init__(
        self,
        presplit: Any,
        precision: Precision | None = None,
    ) -> None:
        super().__init__()
        if precision is None:
            precision = self.default_precision
        self._presplit = presplit
        self._precision = precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None
        self._graph_names: dict[str, tuple[int, int]] = {
            f"ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{self.num_splits}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(
                self.export_sequence_lengths, self.export_context_lengths
            )
        }

    @property
    def shared_source_model(self) -> bool:
        return True

    @property
    def graph_names(self) -> list[str]:
        return list(self._graph_names.keys())

    def component_precision(self) -> Precision:
        return self._precision

    @property
    def _is_quantized(self) -> bool:
        return self._precision != Precision.float

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        if not checkpoint_type.is_aimet_onnx():
            presplit: Any = cls.fp_presplit_cls.from_pretrained(
                host_device=host_device,
            )
            precision = Precision.float
        else:
            precision = checkpoint_type.precision(
                cls.default_precision, checkpoint=checkpoint
            )
            presplit = cls.quant_presplit_cls.from_pretrained(
                precision=precision,
                checkpoint=checkpoint,
                host_device=host_device,
                _skip_quantsim_creation=_skip_quantsim_creation,
            )
        return cls(presplit, precision=precision)

    def _extra_graph_inputs(
        self, name: str, sequence_length: int, context_length: int
    ) -> TensorSpec | None:
        # Deepstack-specific inputs (qwen3 VL only).
        if name == "visual_pos_masks":
            return TensorSpec(shape=(1, sequence_length), dtype="bool")
        if name.startswith("deepstack_visual_embeds_"):
            return TensorSpec(
                shape=(self.default_num_visual_tokens, self.hidden_size),
                dtype="float32",
            )
        return None

    def _get_onnx_input_names(self) -> list[str]:
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    def _get_quant_sim(self) -> QuantizationSimModel:
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )
        onnx_model.ir_version = min(onnx_model.ir_version, 11)

        assert isinstance(self._presplit, self.quant_presplit_cls)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        self._quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLMDynamic_AIMETOnnx._apply_precision_activations(
            self._quant_sim, self._precision
        )

        if onnx_bundle.aimet_encodings_path is not None:
            load_encodings_to_sim(
                self._quant_sim,
                str(onnx_bundle.aimet_encodings_path),
                strict=False,
            )

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()
        providers: list[str] = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        precision_suffix = f"_{self._precision}" if self._is_quantized else ""
        out_dir = Path(output_dir) / f"{model_name}{precision_suffix}{ext}"
        if (out_dir / f"{model_name}.onnx").exists():
            return out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        onnx_bundle = self._get_onnx_bundle()
        onnx_bundle.move(
            dst_folder=str(out_dir),
            dst_model_name=model_name,
            copy=True,
        )

        return out_dir


# ---------------------------------------------------------------------------
# Split-Forward Mixin (for ONNX-based evaluation)
# ---------------------------------------------------------------------------


class Qwen3VLSplitForwardMixin(SplitForwardMixin):
    """Split-forward mixin for Qwen3-VL eval via split Parts.

    Concrete wrappers set ``split_part_classes`` and ``default_num_visual_tokens``.
    """

    split_part_classes: list[type] = []
    default_num_visual_tokens: int = 0

    def get_split_part_classes(self) -> list[type]:
        return self.split_part_classes

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._exporting_onnx or torch.compiler.is_compiling():
            return super(SplitForwardMixin, self).forward(  # type: ignore[misc]
                input_tokens, attention_mask, *args
            )
        self._ensure_parts()
        assert self._parts is not None
        assert self._input_names_for_parts is not None

        full_names = list(
            self.get_input_spec(  # type: ignore[attr-defined]
                sequence_length=DEFAULT_SEQUENCE_LENGTH,
                context_length=DEFAULT_CONTEXT_LENGTH,
            ).keys()
        )
        # Total positional args = input_tokens + attention_mask + *args
        num_provided = 2 + len(args)
        num_expected = len(full_names)

        # Pad missing deepstack inputs with zeros using actual runtime shapes.
        # visual_pos_masks=0 means no visual tokens, so deepstack is a no-op.
        if num_provided < num_expected:
            actual_seq_len = input_tokens.shape[1]
            device = input_tokens.device
            extra = []
            for name in full_names[num_provided:]:
                if name == "visual_pos_masks":
                    extra.append(
                        torch.zeros(1, actual_seq_len, dtype=torch.bool, device=device)
                    )
                elif name.startswith("deepstack_visual_embeds_"):
                    extra.append(
                        torch.zeros(1, self.default_num_visual_tokens, device=device)
                    )
                else:
                    extra.append(torch.zeros(1, device=device))
            args = (*args, *extra)

        return self._split_forward(
            self._parts,
            self._input_names_for_parts,
            input_tokens,
            attention_mask,
            *args,
            output_names_for_parts=self._output_names_for_parts,
        )


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _cached_processor(hf_repo_name: str) -> Any:
    """Load an HF processor once per repo; bundle writing needs it several times."""
    return AutoProcessor.from_pretrained(hf_repo_name)


def _embed_chat_template(
    bundle_dir: Path, checkpoint_path: Path, hf_repo_name: str
) -> None:
    """Embed the chat template into the bundle's tokenizer_config.json.

    Qwen3-VL ships its template on the processor (chat_template.json), so the
    quantized checkpoint's tokenizer_config.json carries none and copying that
    file leaves the bundle with no template at all. Genie's minja engine and
    the host-side eval-prompt builders both read chat_template from
    tokenizer_config.json, so it has to land there rather than as a sidecar.
    """
    tok_cfg_path = bundle_dir / "tokenizer_config.json"
    # No tokenizer_config.json means the template has nowhere to live, which is the
    # template-less bundle this guards against -- fail rather than return quietly.
    if not tok_cfg_path.exists():
        raise ValueError(
            f"tokenizer_config.json is missing from the bundle for {hf_repo_name}; "
            "cannot embed a chat template."
        )
    with open(tok_cfg_path, encoding="utf-8") as f:
        tok_cfg = json.load(f)
    if tok_cfg.get("chat_template"):
        return

    template: str | None = None
    for name in ("chat_template.json", "chat_template.jinja"):
        src = checkpoint_path / name
        if not src.exists():
            continue
        raw = src.read_text(encoding="utf-8")
        if name.endswith(".json"):
            # A malformed sidecar must fall through to the next source, not raise.
            parsed = json.loads(raw)
            template = parsed.get("chat_template") if isinstance(parsed, dict) else None
        else:
            template = raw
        if template:
            break

    if not template:
        template = getattr(_cached_processor(hf_repo_name), "chat_template", None)

    # Shipping a template-less bundle is a hard load failure under minja and
    # silently collapses eval to empty responses, so fail the export instead.
    if not template:
        raise ValueError(
            f"No chat template found for {hf_repo_name} (checked the checkpoint "
            "and the processor); the Genie bundle could not render a chat turn."
        )

    tok_cfg["chat_template"] = template
    with open(tok_cfg_path, "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, indent=2, ensure_ascii=False)


class Qwen3VLCollectionBase(LLMCollectionBase):
    """Collection base for Qwen3-VL deployment.

    Combines N text parts + 1 vision encoder for full VLM deployment.

    Concrete subclasses set ``_hf_repo_name``, the PreSplit / Part / VisionEncoder
    classes, the image/deepstack/patch constants, ``sample_image``, and the
    ordered ``parts`` mapping.
    """

    _checkpoint: str

    # Set by subclass.
    _hf_repo_name: str = ""
    fp_presplit_cls: type[Qwen3VLPreSplitBase]
    quant_presplit_cls: type[Qwen3VLQuantizablePreSplitBase]
    part_base_cls: type[LLMPartBase]
    vision_encoder_cls: type[VisionEncoderCollectionProtocol]
    num_deepstack_layers: int = 0
    vision_patch_size: int = 0
    default_image_height: int = 0
    default_image_width: int = 0
    default_precision: Precision = Precision.w4a16
    sample_image: Any = None
    parts: dict[str, type] = {}

    def _get_collection_processor(self) -> Any:
        # Keep callsites polymorphic via hook while still using cached loading.
        return _cached_processor(self._hf_repo_name)

    def _get_text_config_from_llm_config(self, llm_config: Any) -> Any:
        if hasattr(llm_config, "text_config"):
            return llm_config.text_config
        return llm_config

    def _build_vlm_rope_config(
        self, text_config: Any, image_processor: Any
    ) -> dict[str, Any] | None:
        # Build VLM MRoPE config from the HF config. transformers 5.x nests
        # rope settings (incl. mrope_section) under rope_parameters; get_rope_scaling
        # reads either layout.
        rope_scaling = get_rope_scaling(text_config)
        # Qwen3-VL uses *interleaved* MRoPE (mrope_interleaved=True), which Genie
        # implements only under "qwen3vl-mrope" (nsp-model.cpp). "qwen2vl-mrope"
        # applies a different, contiguous sectioning and would corrupt positions.
        vlm_rope_config: dict[str, Any] = {
            "rope-type": "qwen3vl-mrope",
            "time-step": 50,
        }
        vlm_rope_config["spatial-merge-size"] = image_processor.merge_size
        if rope_scaling is not None and "mrope_section" in rope_scaling:
            vlm_rope_config["mrope-section"] = rope_scaling["mrope_section"]
        return vlm_rope_config

    def _get_chat_template_spec(self) -> dict[str, str]:
        return Qwen3VLTextBase.get_chat_template()

    def _resolve_vision_patch_size(self, image_processor: Any, llm_config: Any) -> int:
        return image_processor.patch_size

    def _build_vision_preprocessing(
        self, image_processor: Any, llm_config: Any
    ) -> GenieVisionPreprocessing:
        return GenieVisionPreprocessing(
            image_width=self.default_image_width,
            image_height=self.default_image_height,
            patch_size=image_processor.patch_size,
            temporal_patch_size=image_processor.temporal_patch_size,
            spatial_merge_size=image_processor.merge_size,
            normalize_mean=image_processor.image_mean,
            normalize_std=image_processor.image_std,
        )

    def _get_veg_prompt_tokenizer(
        self, processor: VLMProcessorLike, tokenizer: Any
    ) -> Any:
        return tokenizer

    def _adapt_dummy_veg_prompt(
        self,
        formatted_text: str,
        processor: VLMProcessorLike,
        num_images: int,
    ) -> str:
        return formatted_text

    def _build_sample_veg_prompt_files(
        self,
        _processor: VLMProcessorLike,
    ) -> tuple[str, str]:
        prompt_prefix = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|>"
        )
        prompt_suffix = (
            "<|vision_end|>Describe the image.<|im_end|>\n<|im_start|>assistant\n"
        )
        return prompt_prefix, prompt_suffix

    def __init__(
        self,
        vision_encoder: Any,
        *parts: Any,
    ) -> None:
        components: dict[str, Any] = {
            "vision_encoder": vision_encoder,
            **dict(zip(self.parts.keys(), parts, strict=True)),
        }
        super().__init__(components)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        **kwargs: Any,
    ) -> Self:
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        precision = (
            checkpoint_type.precision(cls.default_precision, checkpoint=checkpoint)
            if checkpoint_type.is_aimet_onnx()
            else Precision.float
        )

        part_kwargs: dict[str, Any] = dict(
            checkpoint=checkpoint,
            host_device=host_device,
        )
        parts: list[
            VisionEncoderCollectionProtocol | BaseModel | MultiGraphWorkbenchModel
        ] = []
        for part_cls in [cls.vision_encoder_cls, *cls.parts.values()]:
            if issubclass(part_cls, cls.vision_encoder_cls):
                parts.append(
                    part_cls.from_pretrained(
                        checkpoint=checkpoint,
                        device=host_device,
                        precision=precision,
                    )
                )
            else:
                parts.append(part_cls.from_pretrained(**part_kwargs))  # type: ignore[attr-defined]
        instance = cls(*parts)
        resolved_checkpoint: str | Path = checkpoint
        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            for comp in parts:
                presplit = getattr(comp, "_presplit", None)
                ckpt = getattr(presplit, "checkpoint", None)
                if ckpt is not None:
                    resolved_checkpoint = ckpt
                    break
        instance._checkpoint = str(resolved_checkpoint)
        return instance

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """Write genie-app assets: genie config, embedding table, tokenizer, HTP config, app script."""
        output_dir = Path(output_dir)
        checkpoint_path = Path(self._checkpoint)

        # --- Embedding weights ---
        embed_src = checkpoint_path / "embedding_weights.raw"
        if embed_src.exists():
            shutil.copy(embed_src, output_dir / "embedding_weights.raw")
            print("Copied embedding table from checkpoint")
        else:
            fp_model = self.fp_presplit_cls.from_pretrained()
            export_embedding_weights_from_tensor(
                fp_model.get_embedding_weights().float(), output_dir
            )
        metadata.supplementary_files["embedding_weights.raw"] = (
            "Embedding table (float32) for token-to-embedding conversion."
        )

        # --- Tokenizer files ---
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
            src = checkpoint_path / name
            if src.exists():
                shutil.copy(src, output_dir / name)
                metadata.supplementary_files[name] = f"Model {name} from checkpoint."
        _embed_chat_template(output_dir, checkpoint_path, self._hf_repo_name)

        # --- Sample prompt (text-only; vision prompt is assembled at runtime) ---
        tokenizer = get_tokenizer(self._hf_repo_name)
        sample_prompt = self.fp_presplit_cls.get_input_prompt_with_tags(
            include_image=False,
            tokenizer=cast(PreTrainedTokenizer, tokenizer),
        )
        with open(output_dir / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)
        metadata.supplementary_files["sample_prompt.txt"] = (
            "Sample text-only prompt for standalone genie-t2t-run."
        )

        # --- HTP backend extension config ---
        device_info: dict[str, str] = {}
        if metadata.chipset_attributes:
            ca = metadata.chipset_attributes
            if ca.htp_version is not None:
                device_info["hexagon"] = f"v{ca.htp_version}"
            if ca.soc_model is not None:
                device_info["soc-model"] = str(ca.soc_model)
        if save_htp_config_for_genie_bundle(device_info, output_dir):
            metadata.supplementary_files["htp_backend_ext_config.json"] = (
                "HTP backend extension config for Genie."
            )

        # --- Genie config (text-dec-htp.json equivalent) ---
        # Linking collapses all context-length graphs into one context binary
        # with a single reported attention_mask shape, so read the part's own
        # export config instead of scanning metadata.model_files.
        context_lengths: list[int] = []
        image_processor = None
        llm_config = None
        for comp in self.components.values():
            if isinstance(comp, self.part_base_cls):
                context_lengths = list(getattr(comp, "export_context_lengths", []))
                presplit = comp._presplit
                image_processor = getattr(presplit, "_image_processor", None)
                llm_config = getattr(
                    presplit, "_original_llm_config", presplit.llm_config
                )
                break
        context_length: int = max(context_lengths, default=0)

        if image_processor is None:
            image_processor = self._get_collection_processor().image_processor
        vision_patch_size = self._resolve_vision_patch_size(image_processor, llm_config)
        assert vision_patch_size == self.vision_patch_size, (
            f"HF vision patch_size ({vision_patch_size}) "
            f"!= vision_patch_size ({self.vision_patch_size})"
        )
        assert vision_patch_size > 0, "Resolved vision patch_size must be > 0."

        # Build model_list from downloaded text part .bin files (exclude vision encoder)
        model_list = sorted(
            fn
            for fn in metadata.model_files
            if fn.startswith("part") and fn.endswith(".bin")
        )

        # Get text_config from the full VLM config
        assert llm_config is not None, "Could not retrieve llm_config from presplit"
        text_config = self._get_text_config_from_llm_config(llm_config)
        vlm_rope_config = self._build_vlm_rope_config(text_config, image_processor)

        # text-generator.json: used by genie-app-script.txt (genie-app VLM pipeline)
        genie_config = create_genie_config(
            context_length=context_length,
            llm_config=text_config,
            embedding_type="rope",
            model_list=model_list,
            embedding_size=text_config.hidden_size,
            top_level_key="text-generator",
            embedding_lut_path="embedding_weights.raw",
            vlm_rope_config=vlm_rope_config,
        )
        with open(output_dir / "text-generator.json", "w") as f:
            json.dump(genie_config, f, indent=4)
        metadata.supplementary_files["text-generator.json"] = (
            "Genie SDK config for text decoder (VLM pipeline)."
        )

        # genie_config.json: same content with "dialog" key for genie-t2t-run
        dialog_config = create_genie_config(
            context_length=context_length,
            llm_config=text_config,
            embedding_type="rope",
            model_list=model_list,
            embedding_size=text_config.hidden_size,
            top_level_key="dialog",
            embedding_lut_path="embedding_weights.raw",
            vlm_rope_config=vlm_rope_config,
        )
        with open(output_dir / "genie_config.json", "w") as f:
            json.dump(dialog_config, f, indent=4)
        metadata.supplementary_files["genie_config.json"] = (
            "Genie SDK config for genie-t2t-run (text-only LLM testing)."
        )

        # --- Image encoder config (img-enc-htp.json) ---
        veg_bins = sorted(
            fn
            for fn in metadata.model_files
            if fn.startswith("vision_encoder") and fn.endswith(".bin")
        )
        img_enc_config: dict[str, Any] = {
            "image-encoder": {
                "version": 1,
                "engine": {
                    "version": 1,
                    "mode": "image",
                    "backend": {
                        "version": 1,
                        "type": "QnnHtp",
                        "QnnHtp": {
                            "version": 1,
                            "spill-fill-bufsize": 0,
                            "use-mmap": False,
                            "allow-async-init": False,
                        },
                        "extensions": "htp_backend_ext_config.json",
                    },
                    "model": {
                        "version": 1,
                        "type": "binary",
                        "binary": {
                            "version": 1,
                            "ctx-bins": veg_bins,
                        },
                        "vision-param": {
                            "height": self.default_image_height // vision_patch_size,
                            "width": self.default_image_width // vision_patch_size,
                        },
                    },
                },
            }
        }
        with open(output_dir / "img-enc-htp.json", "w") as f:
            json.dump(img_enc_config, f, indent=4)
        metadata.supplementary_files["img-enc-htp.json"] = (
            "Genie SDK config for vision encoder."
        )

        # --- Text encoder config (LUT embedding lookup) ---
        text_enc_config = {
            "text-encoder": {
                "version": 1,
                "type": "lut",
                "lut": {
                    "version": 1,
                    "lut-path": "embedding_weights.raw",
                    "size": text_config.hidden_size,
                    "datatype": "float32",
                },
                "tokenizer": {"version": 1, "path": "tokenizer.json"},
            }
        }
        with open(output_dir / "text-encoder.json", "w") as f:
            json.dump(text_enc_config, f, indent=4)
        metadata.supplementary_files["text-encoder.json"] = (
            "Genie SDK config for text encoder (LUT embedding)."
        )

        # --- Genie metadata & genie-app-script.txt ---
        chat_spec = self._get_chat_template_spec()

        pipeline_nodes = {
            "imageEncoder": "img-enc-htp.json",
            "lutEncoder": "text-encoder.json",
            "textGenerator": "text-generator.json",
        }

        pipeline_connections = [
            GeniePipelineConnection(
                producer_node="imageEncoder",
                producer_node_io="GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT",
                consumer_node="textGenerator",
                consumer_node_io="GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
            ),
            GeniePipelineConnection(
                producer_node="lutEncoder",
                producer_node_io="GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT",
                consumer_node="textGenerator",
                consumer_node_io="GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT",
            ),
        ]

        # Deepstack connection: a single GENIE_NODE_WILDCARD <-> GENIE_NODE_WILDCARD
        # connection. Genie has no dedicated deepstack node-IO enums; instead its
        # InjectiveConnector auto-routes every tensor whose name appears in BOTH
        # the producer's outputs and the consumer's inputs. The VEG outputs
        # ``deepstack_visual_embeds_{0..N-1}`` (+ ``visual_pos_masks``) and the
        # text generator consumes the same names, so one wildcard connection
        # carries all deepstack features by name. This must come AFTER the primary
        # EMBEDDING_OUTPUT->EMBEDDING_INPUT connection above (Genie requires a
        # primary connection before a wildcard), and only one wildcard per node.
        if self.num_deepstack_layers > 0:
            pipeline_connections.append(
                GeniePipelineConnection(
                    producer_node="imageEncoder",
                    producer_node_io="GENIE_NODE_WILDCARD",
                    consumer_node="textGenerator",
                    consumer_node_io="GENIE_NODE_WILDCARD",
                )
            )

        sample_inputs = [
            GenieSampleInput(
                node="lutEncoder",
                node_io="GENIE_NODE_TEXT_ENCODER_TEXT_INPUT",
                file="sample_inputs/prompt_prefix.txt",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT",
                file="sample_inputs/pixel_values.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS",
                file="sample_inputs/position_ids_cos.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN",
                file="sample_inputs/position_ids_sin.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK",
                file="sample_inputs/window_attention_mask.raw",
            ),
            GenieSampleInput(
                node="imageEncoder",
                node_io="GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK",
                file="sample_inputs/full_attention_mask.raw",
            ),
            GenieSampleInput(
                node="lutEncoder",
                node_io="GENIE_NODE_TEXT_ENCODER_TEXT_INPUT",
                file="sample_inputs/prompt_suffix.txt",
            ),
        ]

        metadata.genie = GenieMetadata(
            chat_template=GenieChatTemplate(**cast(dict[str, Any], chat_spec)),
            context_lengths=sorted(context_lengths),
            supports_streaming=True,
            supports_vision=True,
            supports_thinking=False,
            pipeline=GeniePipeline(
                nodes=pipeline_nodes,
                connections=pipeline_connections,
            ),
            sample_inputs=sample_inputs,
            vision_preprocessing=self._build_vision_preprocessing(
                image_processor, llm_config
            ),
        )

        # Generate genie-app-script.txt from the same pipeline data.
        genie_script = generate_genie_app_script(
            pipeline_nodes, pipeline_connections, sample_inputs
        )
        with open(output_dir / "genie-app-script.txt", "w") as f:
            f.write(genie_script)
        metadata.supplementary_files["genie-app-script.txt"] = (
            "Genie-app pipeline script for VLM inference."
        )

        # --- Sample VEG inputs (sample_inputs/ directory) ---
        self._write_sample_veg_inputs(output_dir)

    def _write_sample_veg_inputs(self, output_dir: str | os.PathLike) -> None:
        """Generate sample VEG input .raw files in sample_inputs/ for genie-app."""
        inputs_dir = Path(output_dir) / "sample_inputs"
        inputs_dir.mkdir(exist_ok=True)

        img = load_image(self.sample_image)
        img_resized = img.resize((self.default_image_width, self.default_image_height))

        # Patchify + normalize via HF processor
        proc = self._get_collection_processor()
        processor_tokenizer = getattr(proc, "tokenizer", None)
        if processor_tokenizer is None:
            processor_tokenizer = get_tokenizer(self._hf_repo_name)
        prompt_tokenizer = self._get_veg_prompt_tokenizer(proc, processor_tokenizer)
        dummy_text = self.fp_presplit_cls.get_input_prompt_with_tags(
            user_input_prompt="",
            include_image=True,
            tokenizer=prompt_tokenizer,
        )
        dummy_text = self._adapt_dummy_veg_prompt(dummy_text, proc, 1)
        processed = proc(text=[dummy_text], images=[img_resized], return_tensors="pt")

        # Instantiate VEG to get pre-computed position/attention buffers
        veg = self.vision_encoder_cls.from_pretrained(device=torch.device("cpu"))
        veg.eval()

        raw_files = {
            "pixel_values.raw": processed["pixel_values"],
            "position_ids_cos.raw": veg._pos_emb_cos.cpu().float(),
            "position_ids_sin.raw": veg._pos_emb_sin.cpu().float(),
            "window_attention_mask.raw": veg._window_attention_mask.cpu().float(),
            "full_attention_mask.raw": veg._full_attention_mask.cpu().float(),
        }
        for name, tensor in raw_files.items():
            tensor.detach().numpy().astype(np.float32).tofile(inputs_dir / name)
        del veg

        # Prompt text files
        prompt_prefix, prompt_suffix = self._build_sample_veg_prompt_files(proc)
        (inputs_dir / "prompt_prefix.txt").write_text(prompt_prefix)
        (inputs_dir / "prompt_suffix.txt").write_text(prompt_suffix)

        print(f"Wrote VEG sample inputs to {inputs_dir}/")

    @classmethod
    def prepare_genie_assets(cls, **kwargs: Any) -> None:
        # All genie assets are produced by write_supplementary_files above.
        # The parent class would overwrite genie_config.json with "dialog"
        # key, but VLM pipeline requires "text-generator" key.
        pass
