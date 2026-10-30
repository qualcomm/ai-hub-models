# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

# isort: off
try:
    from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin
    from aimet_onnx.common.defs import QuantizationDataType
    from aimet_onnx.common.utils import AimetLogger
except (ImportError, ModuleNotFoundError):
    print(
        "Some quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Quantized model can be exported without this requirement."
    )
# isort: on

import atexit
import functools
import gc
import glob
import inspect
import itertools
import json
import logging
import math
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from enum import Enum, unique
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, NoReturn, TypeVar, cast

import numpy as np
import onnx
import onnxruntime
import qai_hub as hub
import torch
from onnx.external_data_helper import load_external_data_for_model
from packaging.version import Version
from qai_hub.client import Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models.configs.model_metadata import (
    GenieChatTemplate,
    GenieMetadata,
    GeniePipeline,
    GenieSampleInput,
    ModelFileMetadata,
    ModelMetadata,
)
from qai_hub_models.configs.tensor_spec import (
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.onnx.torch_wrapper import (
    OnnxModelTorchWrapper,
    _verify_onnxruntime_qnn_installed,
)
from qai_hub_models.utils.runtime_torch_wrapper import ModelIODetails
from qai_hub_models.utils.version_helpers import ensure_supported_version

try:
    from transformers import AutoConfig, PretrainedConfig
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    from transformers.models.llama import LlamaConfig
except ImportError:

    class DynamicCache:  # type: ignore[no-redef, unused-ignore]
        pass


from qai_hub.public_rest_api import DatasetEntries
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.datasets.device_eval import is_prompt_task
from qai_hub_models.models.templates.llm.common import (
    TORCH_DYNAMIC_SHAPE_BELOW_VERSION,
    TORCH_DYNAMIC_SHAPE_MIN_VERSION,
    LLMIOType,
    cleanup,
    is_kv_key_name,
    is_kv_value_name,
    is_native_kv,
)
from qai_hub_models.models.templates.llm.llm_helpers import (
    create_genie_config,
    save_htp_config_for_genie_bundle,
)
from qai_hub_models.models.templates.llm.onnx_optimize import optimize_onnx_model
from qai_hub_models.models.templates.llm.sha_dynamic_kvcache import (
    SHADynamicCacheNewValueOnly,
)
from qai_hub_models.models.templates.llm.split_onnx_utils.utils import split_onnx
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, CachedWebModelAsset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphWorkbenchModelCollection,
)
from qai_hub_models.utils.base_multi_graph_model import (
    MultiGraphWorkbenchModel,
)
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.envvars import CheckpointEnvvar
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec
from qai_hub_models.utils.onnx.helpers import (
    ONNXBundle,
    generate_wrapper_onnx_file,
    mock_torch_onnx_inference,
    safe_torch_onnx_export,
)
from qai_hub_models.utils.private_asset_loaders import CachedPrivateAsset
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries
from qai_hub_models.utils.system_info import has_recommended_memory

AIMET_ONNX_INSTALLED = False
try:
    import aimet_onnx.common.quantsim as qs
    from aimet_onnx import quantsim
    from aimet_onnx.quantsim import (
        QuantizationSimModel,
        QuantScheme,
        load_encodings_to_sim,
    )

    from qai_hub_models.models.templates.llm._utils import (
        _apply_int8_kv_cache_tying_and_lm_head,
        _get_kv_io_map,
        _get_lm_head_weights,
        _set_lm_head_to_8b,
    )
    from qai_hub_models.utils.quantization_aimet_onnx import (
        ensure_min_aimet_onnx_version,
    )

    AIMET_ONNX_INSTALLED = True
except (ImportError, ModuleNotFoundError):
    print(
        "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
        "Install qai-hub-models on a Linux machine to use quantized models."
    )

if TYPE_CHECKING:
    from qai_hub_models.models.templates.lm_driver.generator import Generator
    from qai_hub_models.models.templates.lm_schema import DatasetSpec, TechniqueSpec
    from qai_hub_models.utils.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

MIN_TRANSFORMER_VERSION = "4.45.0"
MIN_AIMET_ONNX_VERSION = "2.8.0"
# isort: off

DEFAULT_SEQUENCE_LENGTH = 128
DEFAULT_CONTEXT_LENGTH = 4096

DEFAULT_EXPORT_SEQUENCE_LENGTHS: list[int] = [DEFAULT_SEQUENCE_LENGTH, 1]
DEFAULT_EXPORT_CONTEXT_LENGTHS: list[int] = [DEFAULT_CONTEXT_LENGTH]

DEFAULT_CALIBRATION_SEQ_LEN = 2048

if AIMET_ONNX_INSTALLED:
    ensure_min_aimet_onnx_version(MIN_AIMET_ONNX_VERSION)


try:
    from transformers import (
        AutoTokenizer,
        PreTrainedTokenizerBase,
    )

    # TODO: 10761 remove transformer version check once AIMET
    # transformer restriction is uplifted.
    ensure_supported_version("transformers", min_version=MIN_TRANSFORMER_VERSION)
except ImportError:
    pass


def determine_precision_from_checkpoint(checkpoint: str) -> Precision | None:
    if checkpoint.startswith("DEFAULT_"):
        return Precision.parse(checkpoint[len("DEFAULT_") :].lower())
    return None


def sample_input(
    input_spec: InputSpec,
    input_prompt_processed: str,
    context_length: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizerBase,
    llm_config: PretrainedConfig,
    embedding: Embedding,
) -> dict[str, list[np.ndarray]]:
    input_tokens = tokenizer(
        input_prompt_processed,
        return_tensors="pt",
        padding="max_length",
        max_length=context_length,
    )
    num_tokens = int(
        min(
            torch.sum(input_tokens["attention_mask"]).item(),
            sequence_length,
        )
    )
    use_native_kv = "cache_index" in input_spec

    tokenized_input_ids = input_tokens["input_ids"].type(torch.int32)
    if use_native_kv:
        input_ids = torch.full(
            (tokenized_input_ids.shape[0], sequence_length),
            tokenizer.pad_token_id,
            dtype=torch.int32,
        )
        if num_tokens > 0:
            input_ids[:, :num_tokens] = tokenized_input_ids[:, -num_tokens:]
    else:
        input_ids = tokenized_input_ids[:, -sequence_length:]

    padding_size = sequence_length - num_tokens
    if use_native_kv:
        position_ids_list = list(range(num_tokens)) + [0] * padding_size
    else:
        position_ids_list = [0] * padding_size + list(range(num_tokens))
    position_ids = (
        torch.Tensor(position_ids_list).type(torch.long).reshape(1, sequence_length)
    )
    position_ids_cos, position_ids_sin = embedding.get_embedding(position_ids)

    attention_mask_shape = input_spec["attention_mask"][0]
    key_value_length = attention_mask_shape[3]

    attention_mask = torch.zeros((1, key_value_length))
    if use_native_kv:
        attention_mask[:, :num_tokens] = 1.0
    else:
        attention_mask[:, -num_tokens:] = 1.0

    attention_mask_converter = AttentionMaskConverter(True)
    cm_attention_mask = attention_mask_converter.to_4d(
        attention_mask,
        query_length=sequence_length,
        key_value_length=key_value_length,
        dtype=torch.float32,
    )
    cm_attention_masks = cm_attention_mask.clip(-50, 0)

    if "input_ids" in input_spec:
        input_dict = {
            "input_ids": [input_ids.detach().cpu().numpy()],
        }
    else:
        inputs_embeds = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], llm_config.hidden_size)
        )
        input_dict = {"inputs_embeds": [inputs_embeds.detach().numpy()]}

    input_dict = input_dict | {
        "attention_mask": [cm_attention_masks.detach().numpy()],
        "position_ids_cos": [position_ids_cos.detach().numpy()],
        "position_ids_sin": [position_ids_sin.detach().numpy()],
    }

    # Populate the rest with zeros (KV cache input)
    for k, (shape, _) in input_spec.items():
        if k.startswith("past_"):
            input_dict[k] = [np.zeros(shape, dtype=np.float32)]

    if "cache_index" in input_spec:
        input_dict["cache_index"] = [np.array([0], dtype=np.int32)]

    return input_dict


def get_tokenizer(
    model_ckpt: str | os.PathLike | Path | None,
) -> PreTrainedTokenizerBase:
    """Tokenizer to use for LLMs"""
    assert model_ckpt is not None
    print()
    print(f"Loading tokenizer from {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, is_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = "left"

    return tokenizer


def get_llm_config(model_ckpt: str | os.PathLike | Path | None) -> LlamaConfig:
    """Construct and return a HuggingFace LLM config."""
    assert model_ckpt is not None
    print()
    print(f"Loading model config from {model_ckpt}")
    llm_config = AutoConfig.from_pretrained(model_ckpt, trust_remote_code=True)
    # If it's a multi-modal model, extract only the language model
    if hasattr(llm_config, "text_config"):
        llm_config = llm_config.text_config
    llm_config._attn_implementation = "eager"
    llm_config._attn_implementation_internal = "eager"

    # Force use_cache=true for all LLMs
    llm_config.use_cache = True

    return llm_config


def get_onnx_model(
    fp_model: LLMBase,
    context_length: int,
    sequence_length: int,
    path: str,
    return_model: bool = False,
    llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    quiet: bool = False,
    extra_dynamic_shapes: dict[str, dict[int, Any]] | None = None,
) -> onnx.ModelProto | None:
    """Export an LLM to ONNX.

    extra_dynamic_shapes: dynamic dims keyed by `get_input_spec` name, each
        overriding what the builder below would infer, so a model with extra or
        differently-named inputs (Qwen3-VL's deepstack embeds, Gemma4's
        per_layer_inputs / swa_attention_mask / dual RoPE) declares only the
        difference. Use a named `Dim` where a dim must be shared or derived --
        Gemma4's SWA mask width is `seq + sliding_window`, and `Dim.DYNAMIC`
        cannot do arithmetic.
    """
    ensure_supported_version(
        "torch",
        min_version=TORCH_DYNAMIC_SHAPE_MIN_VERSION,
        below_version=TORCH_DYNAMIC_SHAPE_BELOW_VERSION,
    )

    # Create the checkpoint directory if it does not exist.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # The GPU memory of the model passed into torch.onnx.export cannot
    # subsequently be released due to what looks like a PyTorch bug. We export
    # on the CPU as a workaround.
    assert fp_model.model is not None
    assert isinstance(fp_model.model, torch.nn.Module)
    # Note: torch.nn.Module.device does not exist according to PyTorch
    # documentation and mypy.
    old_device: torch.device = next(iter(fp_model.model.parameters())).device
    device = torch.device("cpu")
    fp_model.to(device)

    input_specs = fp_model.get_input_spec(
        llm_config=fp_model.llm_config.to_dict(),
        context_length=context_length,
        sequence_length=sequence_length,
        llm_io_type=llm_io_type,
    )
    if not quiet:
        print()
        print(
            "Exporting ONNX model with dynamic sequence length and dynamic context length."
        )

    example_input = [
        torch.zeros(
            input_specs[name][0], dtype=getattr(torch, input_specs[name][1])
        ).to(device)
        for name in input_specs
    ]

    data_full_path = os.path.join(os.path.dirname(path), "model.data")
    data_backup_full_path = os.path.join(os.path.dirname(path), ".model.data")

    # If the data file already exists, we need to preserve it. This may have
    # quantization-modified weights (from algorithms like AdaScale), so if we
    # re-export the model from PyTorch, we will revert the weights. To prevent
    # this, we rename the original data file so it is not overwritten and
    # restore it after the export.
    already_has_data = os.path.isfile(data_full_path)
    if already_has_data:
        os.rename(data_full_path, data_backup_full_path)

    # Build dynamic_shapes for dynamo export
    from torch.export import Dim

    # Define dimension constraints
    # seq_len, kv_seq_len, and ctx_len are all dynamic.
    # ctx_len = seq_len + kv_seq_len, but we declare them independently
    # and let the tracer infer the relationship from the model computation.
    seq_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]
    kv_seq_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]
    ctx_len = Dim.DYNAMIC  # type: ignore[attr-defined, unused-ignore]

    input_names = list(input_specs.keys())

    # Per-input dims, in get_input_spec order. extra_dynamic_shapes wins for
    # every name including the leading two, so a model that breaks the
    # (input_tokens, attention_mask) convention (Gemma4 has per_layer_inputs
    # second and two masks) declares just those names.
    _extra = extra_dynamic_shapes or {}
    per_input_dynamic_shapes: list[dict[int, Any] | None] = []
    for i, name in enumerate(input_names):
        if name in _extra:
            per_input_dynamic_shapes.append(_extra[name])
        elif i == 0:
            # input_tokens (input_ids or input_embeds): (1, seq_len, ...)
            per_input_dynamic_shapes.append({1: seq_len})
        elif i == 1:
            # attention_mask: (1, 1, seq_len, context_length)
            per_input_dynamic_shapes.append({2: seq_len, 3: ctx_len})
        elif name == "position_ids":
            # Shape: (1, seq_len)
            per_input_dynamic_shapes.append({1: seq_len})
        elif name in ("position_ids_cos", "position_ids_sin"):
            # Shape: (1, 1, seq_len, embed_dim)
            per_input_dynamic_shapes.append({2: seq_len})
        elif is_kv_key_name(name):
            # past_key shape: (num_kv_heads, 1, embed_dim*2, kv_seq_len)
            per_input_dynamic_shapes.append({3: kv_seq_len})
        elif is_kv_value_name(name):
            # past_value shape: (num_kv_heads, 1, kv_seq_len, embed_dim*2)
            per_input_dynamic_shapes.append({2: kv_seq_len})
        elif name == "cache_index":
            # No dynamic dims for cache_index (shape [1])
            per_input_dynamic_shapes.append(None)
        else:
            # No dynamic dims for other inputs
            per_input_dynamic_shapes.append(None)

    # Nest to match the forward signature: forward(self, *inputs) (Gemma4) takes
    # one tuple wrapping every input, while the conventional
    # forward(self, input_tokens, attention_mask, *args) takes the 3-element form.
    forward_params = list(
        inspect.signature(type(fp_model).forward).parameters.values()
    )[1:]  # drop self
    only_varargs = len(forward_params) == 1 and (
        forward_params[0].kind is inspect.Parameter.VAR_POSITIONAL
    )
    dynamic_shapes: tuple[Any, ...]
    if only_varargs:
        dynamic_shapes = (tuple(per_input_dynamic_shapes),)
    else:
        dynamic_shapes = (
            per_input_dynamic_shapes[0],
            per_input_dynamic_shapes[1],
            tuple(per_input_dynamic_shapes[2:]),
        )

    try:
        extra = {
            "opset_version": 18,
            "dynamo": True,
            "optimize": False,
            "dynamic_shapes": dynamic_shapes,
        }

        with torch.no_grad():
            safe_torch_onnx_export(
                fp_model,
                tuple(example_input),
                path,
                input_names=list(input_specs.keys()),
                output_names=list(
                    fp_model._get_output_spec(
                        fp_model.llm_config.num_hidden_layers,
                    )
                ),
                **extra,
            )

        fp_model.to(old_device)

        onnx_model = onnx.load(path)
        # Clean up multiple weights files
        for file in glob.glob(os.path.join(os.path.dirname(path), "*.weight")):
            os.remove(file)
        for file in glob.glob(os.path.join(os.path.dirname(path), "onnx__*")):
            os.remove(file)

        onnx_model = optimize_onnx_model(onnx_model)

        onnx.save_model(
            onnx_model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )

    finally:
        if already_has_data:
            os.replace(data_backup_full_path, data_full_path)

    load_external_data_for_model(onnx_model, os.path.dirname(path))

    if not return_model:
        del onnx_model
        gc.collect()
    return onnx_model if return_model else None


def _get_evaluator(
    task: str,
    context_length: int,
    sequence_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    **task_kwargs: Any,
) -> BaseEvaluator:
    from qai_hub_models.models.templates.llm.kldiv_evaluator import KLDivEvaluator
    from qai_hub_models.models.templates.llm.llm_response_evaluator import (
        LLMResponseEvaluator,
    )
    from qai_hub_models.models.templates.llm.mmlu_evaluator import MMLUEvaluator
    from qai_hub_models.models.templates.llm.mmmu_evaluator import MMMUEvaluator
    from qai_hub_models.models.templates.llm.ppl_evaluator import PerplexityEvaluator

    if "wikitext" in task:
        return PerplexityEvaluator(context_length, device, tokenizer)
    if "tricky_llm_prompts" in task:
        return KLDivEvaluator(context_length, device, tokenizer, verbose=True)
    if "mmmu" in task:
        return MMMUEvaluator(context_length, device, tokenizer)
    if is_prompt_task(task):
        if "output_dir" not in task_kwargs:
            raise ValueError(
                f"Task '{task}' requires 'output_dir' to be passed via "
                f"get_evaluator(..., output_dir=...)."
            )
        return LLMResponseEvaluator(
            context_length=context_length,
            device=device,
            tokenizer=tokenizer,
            output_dir=task_kwargs["output_dir"],
            max_new_tokens=task_kwargs.get("max_new_tokens", 2048),
            end_tokens=task_kwargs.get("end_tokens"),
            grader_venv=task_kwargs.get("grader_venv"),
        )
    return MMLUEvaluator(context_length, device, tokenizer)


class Embedding(ABC):
    def __init__(  # noqa: B027
        self,
        head_dim: int = 128,
        max_length: int = 2048,
        config: Any = None,
    ) -> None:
        pass

    @abstractmethod
    def get_embedding(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


class PositionProcessorBase(torch.nn.Module):
    """Prepares positions (Embedding and attention mask preparation); used by ORT GenAI."""

    def __init__(self, context_length: int, config: PretrainedConfig) -> None:
        super().__init__()

    def forward(
        self, attention_mask_before_processor: torch.Tensor, position_ids: torch.Tensor
    ) -> NoReturn:
        raise NotImplementedError("Must be implemented by subclass")


class LLMConfigEditor:
    def edit_llm_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        return llm_config  # no change by default


FPModelT = TypeVar("FPModelT", bound="LLMBase")


class SingleSlotCacheMixin:
    """Mixin that maintains a single per-class cached instance keyed by a string key.

    On cache hit (same key), returns the existing instance.
    On cache miss (different key), evicts and frees the old instance
    before the caller creates a new one.

    The slot is keyed by the exact class, never inherited through the MRO.
    That matters because sibling/derived model classes share a checkpoint
    (and therefore a cache key): e.g. ``FPSplitModelWrapper`` derives from
    ``<Model>_PreSplit``, and both use the HF repo name as their key. With
    a plain class attribute, the write from whichever class is constructed
    first lands in that class's ``__dict__`` while lookups on the other
    resolve to it through inheritance -- so ``FPSplitModelWrapper.from_pretrained()``
    would silently hand back a ``PreSplit`` instance (making
    ``isinstance(model, SplitForwardMixin)`` False and running the monolithic
    forward under the "split" label), and ``FPSplitModelWrapper.release()``
    would free the ``PreSplit``'s weights while leaving the ``PreSplit`` slot
    pointing at a ``model=None`` husk.

    Subclasses must provide a ``free_memory()`` method on instances.
    """

    # class -> (instance, cache_key). Keyed by the exact class object so a
    # subclass never sees (or evicts) an ancestor's cached instance.
    _cache_slots: ClassVar[dict[type, tuple[Any, str]]] = {}

    @classmethod
    def cache_lookup(cls, cache_key: str) -> Any:
        """Return cached instance if key matches, or None on miss.

        On miss with an existing cached instance, the old one is evicted.
        """
        slot = SingleSlotCacheMixin._cache_slots.get(cls)
        if slot is None:
            return None
        instance, key = slot
        if key == cache_key:
            return instance

        # Different key -- evict old instance
        del SingleSlotCacheMixin._cache_slots[cls]
        instance.free_memory()
        return None

    @classmethod
    def cache_store(cls, instance: Any, cache_key: str) -> None:
        """Evict any existing cached instance, then store the new one."""
        slot = SingleSlotCacheMixin._cache_slots.get(cls)
        if slot is not None and slot[0] is not instance:
            cls.release()
        SingleSlotCacheMixin._cache_slots[cls] = (instance, cache_key)

    @classmethod
    def release(cls) -> None:
        """Evict this class's cached instance, free it, and run global cleanup.

        Doubles as the instance ``release()`` for cached models: Python
        binds this classmethod to ``type(self)`` when called as
        ``instance.release()``, so the instance's own cache slot is evicted.

        Releasing a class that holds no cached instance only runs the global
        cleanup; it never touches a base or derived class's slot.
        """
        slot = SingleSlotCacheMixin._cache_slots.pop(cls, None)
        if slot is not None:
            slot[0].free_memory()
        cleanup()


class DynamicPreSplitOnnxMixin:
    """Mixin that manages ONNX splitting for PreSplit models.

    Provides a cached convert_to_onnx_and_split() workflow using
    the template method pattern. Subclasses implement
    get_full_onnx_bundle() to supply the full (unsplit) ONNXBundle.

    Class-level configuration (override in subclasses):
        split_model_name: str  -- basename for split files
        num_splits: int        -- number of ONNX splits
        num_layers_per_split: int -- layers per split
        split_lm_head: bool    -- separate LM head into its own part
        reuse_emitted_graph_outputs: bool -- later splits may read an earlier
            split's graph outputs as inputs
    """

    split_model_name: str = ""
    num_splits: int = 0
    num_layers_per_split: int = 0
    # VLM models feed inputs_embeds directly and have no embedding split.
    split_embedding: bool = True
    split_lm_head: bool = False
    # KV-shared models (Gemma4) have a later split consume the earlier split's
    # KV-cache graph outputs. Off by default: it changes partition boundaries,
    # and the no-op claim is unverified on the other split LLMs.
    reuse_emitted_graph_outputs: bool = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.onnx_splits: dict[int, ONNXBundle] = {}
        # TemporaryDirectory removes itself when collected. No __del__ here:
        # torch.export's AttrProxy copies share this __dict__ and would run it mid-export.
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def convert_to_onnx_and_split(self, part_id: int = 1) -> ONNXBundle:
        """Convert to ONNX and split into parts. Results are cached.

        Parameters
        ----------
        part_id
            Part ID (1-indexed) to return.

        Returns
        -------
        ONNXBundle
            The requested split as an ONNXBundle.
        """
        if part_id in self.onnx_splits:
            return self.onnx_splits[part_id]

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self._temp_dir.name)

        full_bundle = self.get_full_onnx_bundle(temp_path)
        full_bundle = self._postprocess_full_onnx_bundle(full_bundle)

        split_output_dir = temp_path / "splits_dynamic"
        split_output_dir.mkdir(parents=True, exist_ok=True)

        split_bundles = split_onnx(
            onnxfile=full_bundle,
            modelname=self.split_model_name,
            num_splits=self.num_splits,
            num_layers_per_split=self.num_layers_per_split,
            output_dir=str(split_output_dir),
            split_embedding=self.split_embedding,
            split_lm_head=self.split_lm_head,
            splitting_points=getattr(self, "splitting_points", None),
            reuse_emitted_graph_outputs=self.reuse_emitted_graph_outputs,
        )

        for i, bundle in enumerate(split_bundles):
            self.onnx_splits[i + 1] = bundle

        return self.onnx_splits[part_id]

    def get_full_onnx_bundle(self, temp_path: Path) -> ONNXBundle:
        """Return the full (unsplit) ONNXBundle.

        Default implementation copies model_dynamic.onnx, model.data, and
        model.encodings from the checkpoint directory into a temp bundle.
        This works for quantized PreSplit models whose checkpoint already
        contains these files (checkpoint must be resolved before calling).

        If the checkpoint is not a local directory with model_dynamic.onnx,
        delegates to super() (e.g. Llama3DynamicBase which exports from
        PyTorch).
        """
        checkpoint = getattr(self, "checkpoint", None)
        if checkpoint is None or (
            isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT")
        ):
            return super().get_full_onnx_bundle(temp_path)  # type: ignore[misc]

        checkpoint_path = Path(checkpoint)
        src_onnx = checkpoint_path / "model_dynamic.onnx"
        if not src_onnx.exists():
            return super().get_full_onnx_bundle(temp_path)  # type: ignore[misc]

        bundle_dir = temp_path / "bundle_dynamic"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(src_onnx, bundle_dir / "model.onnx")

        src_data = checkpoint_path / "model.data"
        if src_data.exists():
            shutil.copy(src_data, bundle_dir / "model.data")

        src_enc = checkpoint_path / "model.encodings"
        if src_enc.exists():
            shutil.copy(src_enc, bundle_dir / "model.encodings")

        return ONNXBundle.from_bundle_path(bundle_dir, "model")

    def _postprocess_full_onnx_bundle(self, bundle: ONNXBundle) -> ONNXBundle:
        """Hook for subclasses to adjust the full ONNX bundle before splitting.

        Cooperative: forwards to later classes in the MRO so model-family
        classes (which sit after this mixin in MRO) can reshape the shipped
        ONNX/encodings to match what the split/compile step expects.
        """
        postprocess = getattr(super(), "_postprocess_full_onnx_bundle", None)
        if postprocess is not None:
            bundle = postprocess(bundle)
        return bundle

    def free_memory(self) -> None:
        """Free memory by clearing caches and releasing resources."""
        # Clear torch model (FP path)
        model = getattr(self, "model", None)
        if model is not None:
            model.to("cpu")
            del model
            setattr(self, "model", None)  # noqa: B010

        # Clear QuantSim (quantized path)
        quant_sim = getattr(self, "quant_sim", None)
        if quant_sim is not None:
            del quant_sim
            setattr(self, "quant_sim", None)  # noqa: B010

        # Clear ONNX split cache and temp directory
        self.onnx_splits.clear()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

        cleanup()


class DynamicQuantizablePreSplitMixin(
    SingleSlotCacheMixin, DynamicPreSplitOnnxMixin, Generic[FPModelT]
):
    """Mixin for quantized PreSplit models with dynamic ONNX shapes.

    Combines SingleSlotCacheMixin (class-level cache) and DynamicPreSplitOnnxMixin
    (ONNX splitting) with DEFAULT-checkpoint resolution and
    save_calibrated_checkpoint.

    Must be mixed with a subclass of LLM_AIMETOnnx which provides
    FPModel, create_onnx_models, save_tokenizer_and_config,
    from_pretrained, and save_calibrated_checkpoint.

    Override ``resolve_default_checkpoint`` to change how DEFAULT
    checkpoints are fetched. The default downloads and extracts a full
    .zip archive (dynamic ONNX + encodings + weights + tokenizer + config).
    """

    # Declared for type-checking; provided by LLMDynamicBase / LLMDynamic_AIMETOnnx at runtime.
    model_id: str
    model_asset_version: int
    default_checkpoint: dict[Precision, str]
    supported_precisions: list[Precision]
    default_precision: Precision
    FPModel: type[FPModelT]
    sequence_length: int
    context_length: int

    # Used unless ``_skip_quantsim_creation`` is not None.
    skip_quantsim_creation_default: bool = False

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, checkpoint=checkpoint, **kwargs)
        self.precision: Precision = kwargs.get("precision", self.default_precision)

    @classmethod
    def resolve_default_checkpoint(
        cls,
        precision: Precision,
        host_device: torch.device,
        fp_model: FPModelT | None,
    ) -> tuple[str, FPModelT | None]:
        """Resolve a DEFAULT checkpoint string to a local directory path.

        Downloads and extracts the full .zip archive. Override in subclasses
        for different strategies (e.g. downloading only encodings and
        exporting ONNX from the FP torch model).

        The default implementation downloads a full .zip archive that
        contains ONNX + encodings + data. If ``fp_model`` is provided,
        it additionally exports ONNX models and saves tokenizer/config.

        Override in subclasses for different fetching strategies (e.g.
        downloading only encodings and exporting ONNX from a torch model).

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
            (resolved_checkpoint_path, fp_model). The fp_model may be
            the same as input or a newly created one.
        """
        return cls.fetch_default_checkpoint(precision), fp_model

    @classmethod
    def fetch_default_checkpoint(cls, precision: Precision) -> str:
        """Download and extract the DEFAULT asset zip for *precision*.

        QAIHM_TEST_CHECKPOINT redirects this to a zip in the private bucket, so a
        quantization can be evaluated before it is published.

        Returns the local path to the extracted checkpoint directory.
        """
        private_key = CheckpointEnvvar.get()
        if private_key:
            return str(
                CachedPrivateAsset(
                    private_key,
                    # Distinct version segment: fetch() short-circuits on an
                    # extracted dir, so sharing one would serve public weights.
                    ASSET_CONFIG.get_local_store_model_path(
                        cls.model_id,
                        f"{cls.model_asset_version}-private",
                        Path(private_key).name,
                    ),
                ).fetch(extract=True)
            )

        precision_checkpoint = cls.default_checkpoint[precision]
        return str(
            CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                precision_checkpoint + ".zip",
            ).fetch(extract=True)
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | os.PathLike | Path = "DEFAULT",
        host_device: torch.device | None = None,
        precision: Precision | None = None,
        fp_model: FPModelT | None = None,
        _skip_quantsim_creation: bool | None = None,
    ) -> Self:
        """
        Load or return a cached Quantizable PreSplit.

        If a cached instance exists for the same checkpoint, returns it
        with updated sequence/context lengths (dynamic shapes). If a
        different checkpoint is requested, the old instance is freed first.

        Parameters
        ----------
        checkpoint
            Controls whether evaluation/inference runs floating point or
            quantized (QuantSim):
              - "DEFAULT": quantized simulation, using the default
                published quantized checkpoint.
              - "DEFAULT_UNQUANTIZED": floating-point baseline, using
                the default published FP weights.
              - Local folder: quantized (model*.onnx + model.encodings)
                or floating point (config.yaml + model.safetensor),
                inferred from folder contents.
            Use DEFAULT_UNQUANTIZED vs. DEFAULT to compare FP
            accuracy against quantized accuracy.
        host_device
            Device for computation.
        precision
            Quantization precision. Defaults to cls.default_precision.
        fp_model
            Optional FP model, passed by the evaluate framework to avoid
            creating a duplicate. If not provided and checkpoint starts
            with "DEFAULT", one is created automatically.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing). Only relevant when
            checkpoint resolves to a quantized checkpoint; ignored
            for floating-point evaluation. When None (the default),
            falls back to cls.skip_quantsim_creation_default.

        Returns
        -------
        Self
            The cached or newly created instance.
        """
        if _skip_quantsim_creation is None:
            _skip_quantsim_creation = cls.skip_quantsim_creation_default
        if precision is None:
            precision = cls.default_precision

        cache_key = str(checkpoint)
        cached = cls.cache_lookup(cache_key)
        if cached is not None:
            return cached

        if host_device is None:
            host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(checkpoint, str) and checkpoint.startswith("DEFAULT"):
            precision = determine_precision_from_checkpoint(checkpoint) or precision
            if precision not in cls.supported_precisions:
                available = [str(p) for p in cls.supported_precisions]
                raise ValueError(
                    f"This model is not supported for {precision!s} precision. "
                    f"Models are available in following precisions: {','.join(available)}."
                )
            if precision not in cls.default_checkpoint:
                available = [str(p) for p in cls.default_checkpoint]
                raise ValueError(
                    f"No checkpoint is available for this model in {precision!s} precision. "
                    f"Please generate a local quantized checkpoint. "
                    f"Checkpoints are available in the following precisions: {','.join(available)}."
                )
            checkpoint, fp_model = cls.resolve_default_checkpoint(
                precision=precision,
                host_device=host_device,
                fp_model=fp_model,
            )

        instance = super().from_pretrained(  # type: ignore[misc]
            checkpoint=checkpoint,
            host_device=host_device,
            precision=precision,
            fp_model=fp_model,
            _skip_quantsim_creation=_skip_quantsim_creation,
        )

        instance.precision = precision

        cls.cache_store(instance, cache_key)
        return instance

    def save_calibrated_checkpoint(
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: FPModelT | None = None,
    ) -> None:
        """Save calibrated checkpoint with dynamic shapes."""
        if fp_model is None:
            fp_model = self.FPModel.from_pretrained()
        super().save_calibrated_checkpoint(  # type: ignore[misc]
            output_checkpoint,
            fp_model,
        )


class SplitForwardMixin:
    """Mixin that overrides forward() to chain inference through split Parts.

    Lazily creates Part instances on first forward call and maps inputs
    from the full model's input spec to each Part's ONNX input names.
    Intermediate hidden states flow from one Part to the next.

    Mix in *before* the PreSplit base so that this forward() wins in MRO::

        class MyWrapper(SplitForwardMixin, SomePreSplit): ...

    Subclasses must override ``_get_split_part_classes()`` to return
    the concrete Part classes.
    """

    # Used unless ``_skip_quantsim_creation`` is not None.
    skip_quantsim_creation_default: bool = True

    _parts: list | None
    _input_names_for_parts: list[list] | None
    _output_names_for_parts: list[list[str]] | None
    _exporting_onnx: bool
    # Optional GPU pin for eval: when the torch weights live on CPU (see
    # evaluate.py) the generator reads this to place KV/compute on the GPU.
    host_device: torch.device | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._parts = None
        self._input_names_for_parts = None
        self._output_names_for_parts = None
        self._exporting_onnx = False
        super().__init__(*args, **kwargs)

    # -- overridable hooks ------------------------------------------------

    def get_split_part_classes(self) -> list[type]:
        """Return the Part classes to instantiate, in order.

        Each class must accept ``(presplit, precision=...)`` and provide
        ``_get_onnx_input_names() -> list[str]``.
        """
        raise NotImplementedError("Subclasses must override get_split_part_classes()")

    def get_split_precision(self) -> Precision:
        """Return the precision to use when creating Parts."""
        return getattr(self, "precision", Precision.float)

    # -- implementation ---------------------------------------------------

    @staticmethod
    def _build_input_name_for_parts(
        parts: list, full_input_names: list[str]
    ) -> tuple[list[list], list[list[str]]]:
        """Build input mapping table for split parts.

        For each Part, maps its ONNX input names to positional indices in the
        full model's args.  Names not found in the full model (intermediate
        hidden states) are tagged ``"prev"`` and sourced from the previous
        Part's first output.

        A Part input may instead name a graph output *emitted by an earlier
        Part* -- Gemma4's KV-shared layers take the earlier split's KV ``_out``
        tensors as leaf inputs (see ``reuse_emitted_graph_outputs``). Those are
        tagged ``("emitted", name)``; falling through to ``"prev"`` would hand
        them the incoming hidden state, whose rank differs (ORT: "Invalid rank
        for input: swa_value_22_h1_out Got: 3 Expected: 4").

        Also returns each Part's ONNX output names, so ``_split_forward`` can
        index emitted tensors by name without re-parsing the graphs per token.
        """
        name_to_idx = {n: i for i, n in enumerate(full_input_names)}
        input_names_for_parts: list[list] = []
        output_names_for_parts: list[list[str]] = []
        emitted: set[str] = set()
        for part in parts:
            sources: list = []
            for n in part._get_onnx_input_names():
                if n in name_to_idx:
                    sources.append(name_to_idx[n])
                elif n in emitted:
                    sources.append(("emitted", n))
                elif is_kv_key_name(n) or is_kv_value_name(n):
                    # A KV input must resolve to a full-model input or an
                    # earlier Part's output; "prev" would silently bind the
                    # rank-3 hidden state to a rank-4 cache.
                    raise ValueError(
                        f"Part {len(input_names_for_parts)} KV input {n!r} matches no "
                        f"full-model input and no earlier Part output. "
                        f"Full-model KV inputs: "
                        f"{[m for m in full_input_names if is_kv_key_name(m) or is_kv_value_name(m)]}"
                    )
                else:
                    sources.append("prev")
            input_names_for_parts.append(sources)
            part_outputs = part._get_onnx_output_names()
            output_names_for_parts.append(part_outputs)
            emitted.update(part_outputs)
        return input_names_for_parts, output_names_for_parts

    @staticmethod
    def _split_forward(
        parts: list,
        input_names_for_parts: list[list],
        *args: torch.Tensor,
        output_names_for_parts: list[list[str]] | None = None,
    ) -> list[torch.Tensor]:
        """Chain Part1 -> Part2 -> ... with input mapping.

        Works identically for FP and quantized Parts -- each Part's own
        ``forward()`` handles the FP / quantized branching internally.
        """
        prev_output: torch.Tensor | None = None
        kv_outputs: list[torch.Tensor] = []
        # Graph outputs emitted so far, for Parts that take an earlier Part's
        # output as a leaf input (Gemma4 KV-shared layers).
        emitted: dict[str, torch.Tensor] = {}

        for i, (part, input_names) in enumerate(
            zip(parts, input_names_for_parts, strict=False)
        ):
            part_args: list[torch.Tensor] = []
            for s in input_names:
                if isinstance(s, tuple) and s[0] == "emitted":
                    part_args.append(emitted[s[1]])
                elif s == "prev":
                    # Part1's inputs all come from the full model's input spec,
                    # so "prev" should never appear in its mapping.
                    assert prev_output is not None, (
                        f"Part {i} consumes the previous Part's output, but it is the first Part."
                    )
                    part_args.append(prev_output)
                else:
                    part_args.append(args[s])
            outputs = part(*part_args)
            outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)

            if output_names_for_parts is not None:
                emitted.update(zip(output_names_for_parts[i], outputs, strict=False))

            prev_output = outputs[0]  # hidden state or logits
            kv_outputs.extend(outputs[1:])  # KV cache updates (empty for Part1)

        # prev_output is logits from the final Part
        assert prev_output is not None
        return [prev_output, *kv_outputs]

    def _ensure_parts(self) -> None:
        """Lazily create split Parts on first forward call."""
        if self._parts is not None:
            return
        precision = self.get_split_precision()
        self._parts = [
            cls(self, precision=precision) for cls in self.get_split_part_classes()
        ]
        full_names = list(
            self.get_input_spec(  # type: ignore[attr-defined]
                sequence_length=self.sequence_length,  # type: ignore[attr-defined]
                context_length=self.context_length,  # type: ignore[attr-defined]
                llm_io_type=self.llm_io_type,  # type: ignore[attr-defined]
            ).keys()
        )
        (
            self._input_names_for_parts,
            self._output_names_for_parts,
        ) = self._build_input_name_for_parts(self._parts, full_names)

    def convert_to_onnx_and_split(self, part_id: int = 1) -> Any:
        """Wrap ONNX export so forward() falls back to LLMBase.forward.

        During ``get_full_onnx_bundle`` -> ``get_onnx_model`` ->
        ``torch.onnx.export(dynamo=True)``, the tracer calls ``forward()``.
        The flag makes ``forward()`` delegate to the base-class torch forward
        (``LLMBase.forward``) instead of the ORT-based split forward, which
        is not traceable and would recursively re-enter this method.
        """
        self._exporting_onnx = True
        try:
            return super().convert_to_onnx_and_split(part_id=part_id)  # type: ignore[misc]
        finally:
            self._exporting_onnx = False

    def free_memory(self) -> None:
        """Release the split Parts' GPU sessions, then the base's resources.

        Each Part's ORT/QuantSim session lives outside torch's allocator (see
        ``LLMPartBase.release``), so dropping ``_parts`` is what frees that GPU
        memory before the grader runs.
        """
        if self._parts is not None:
            for part in self._parts:
                part.release()
            self._parts = None
            self._input_names_for_parts = None
            self._output_names_for_parts = None
        base_free = getattr(super(), "free_memory", None)
        if base_free is not None:
            base_free()

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self._exporting_onnx or torch.compiler.is_compiling():
            return super().forward(input_tokens, attention_mask, *args)  # type: ignore[misc]
        self._ensure_parts()
        assert self._parts is not None
        assert self._input_names_for_parts is not None
        return self._split_forward(
            self._parts,
            self._input_names_for_parts,
            input_tokens,
            attention_mask,
            *args,
            output_names_for_parts=self._output_names_for_parts,
        )


class LLMPartBase:
    """Base for one split of a dynamic-shape ONNX LLM exported for deployment.

    A "Part" is one split of the full model. Both text LLMs (e.g. Llama) and
    VLMs (e.g. Qwen2.5-VL) split the model into Parts whose ONNX input names
    follow the same conventions (``past_key_*`` / ``past_value_*`` /
    ``attention_mask`` / ``position_ids_cos`` / ``position_ids_sin`` / the
    token-or-embedding input / intermediate hidden states). This base derives
    the per-graph input spec and output names from those conventions, reading
    the architecture dimensions from class attributes.

    Subclasses (the model-family Part base) must provide:

    - ``hidden_size``, ``num_attention_heads``, ``num_key_value_heads`` (int)
    - ``part_id`` (1-indexed int)
    - ``_presplit``: the full-model PreSplit instance (an ``LLMBase``), whose
      ``llm_io_type`` determines the first-input convention (see below)
    - ``_graph_names``: mapping of graph name -> (sequence_length, context_length)
    - ``_get_onnx_input_names()`` / ``_get_onnx_output_names()``

    ``_presplit.llm_io_type`` gates the first-input convention:

    - ``*_input_ids`` (text LLMs): the embedding is its own split, so Part 1's
      only input is the int32 ``input_ids``.
    - ``genie_input_embeds`` (VLMs): there is no embedding split; the token
      input arrives pre-embedded as float32 ``inputs_embeds``.
    """

    # Provided by the concrete Part base.
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    # Optional explicit head dim. When None, derived as
    # ``hidden_size // num_attention_heads``.
    head_dim: int | None = None
    part_id: int
    # The full-model PreSplit (FP or quantized). Typed Any because the FP and
    # quantized PreSplits live in separate hierarchies (LLMBase vs
    # LLM_AIMETOnnx); the only attribute used here is ``llm_io_type``.
    _presplit: Any
    _graph_names: dict[str, tuple[int, int]]
    # GPU-resident inference sessions, set in each subclass's __init__.
    _fp_session: onnxruntime.InferenceSession | None
    _quant_sim: QuantizationSimModel | None

    def release(self) -> None:
        """Drop this Part's ORT/QuantSim GPU sessions. Idempotent.

        These sessions allocate GPU memory outside torch's allocator, so they
        must be dropped explicitly -- ``torch.cuda.empty_cache()`` can't reclaim
        them. Shared by all Part hierarchies (text LLMs and VLMs).
        """
        self._fp_session = None
        self._quant_sim = None

    def _get_onnx_input_names(self) -> list[str]:
        raise NotImplementedError

    def _get_onnx_output_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def _splits_embedding(self) -> bool:
        """Whether the embedding is its own split (Part 1 takes input_ids).

        Derived from the full model's IO signature: models fed integer token
        ids split the embedding into Part 1; models fed pre-Gather embeddings
        (VLMs) do not.
        """
        return self._presplit.llm_io_type in (
            LLMIOType.genie_input_ids,
            LLMIOType.huggingface_input_ids,
        )

    def get_graph_input_spec(self, graph_name: str) -> InputSpec:
        """Build the input spec for one graph from ONNX input names.

        Shapes are derived from the architecture dimensions and the
        (sequence_length, context_length) recorded for ``graph_name``.
        """
        sequence_length, context_length = self._graph_names[graph_name]

        # Embedding split (text LLMs): Part 1's only input is input_ids.
        if self._splits_embedding and self.part_id == 1:
            return {"input_ids": TensorSpec(shape=(1, sequence_length), dtype="int32")}

        head_dim = self.head_dim or (self.hidden_size // self.num_attention_heads)
        embed_dim = head_dim // 2

        # Read actual input names from the split ONNX model.
        onnx_input_names = self._get_onnx_input_names()
        use_native_kv = any("nativekvcache" in n for n in onnx_input_names)
        kv_seq_len = (
            context_length if use_native_kv else context_length - sequence_length
        )

        spec: InputSpec = {}
        for name in onnx_input_names:
            if is_kv_key_name(name):
                spec[name] = TensorSpec(
                    shape=(
                        self.num_key_value_heads,
                        1,
                        head_dim,
                        kv_seq_len,
                    ),
                    dtype="float32",
                )
            elif is_kv_value_name(name):
                spec[name] = TensorSpec(
                    shape=(
                        self.num_key_value_heads,
                        1,
                        kv_seq_len,
                        head_dim,
                    ),
                    dtype="float32",
                )
            elif name == "cache_index":
                spec[name] = TensorSpec(shape=(1,), dtype="int32")
            elif name == "input_ids":
                spec[name] = TensorSpec(shape=(1, sequence_length), dtype="int32")
            elif name == "attention_mask":
                spec[name] = TensorSpec(
                    shape=(1, 1, sequence_length, context_length),
                    dtype="float32",
                )
            elif "position_ids_cos" in name or "position_ids_sin" in name:
                spec[name] = TensorSpec(
                    shape=(1, 1, sequence_length, embed_dim),
                    dtype="float32",
                )
            else:
                # Family-specific inputs (e.g. VLM deepstack tensors) get first
                # refusal via this hook; the base returns None for names it does
                # not recognize.
                extra = self._extra_graph_inputs(name, sequence_length, context_length)
                if extra is not None:
                    spec[name] = extra
                else:
                    # Either the pre-embedded token input ("inputs_embeds" for
                    # VLMs) or an intermediate hidden state from the previous
                    # part (by elimination). Both are (1, seq, hidden).
                    spec[name] = TensorSpec(
                        shape=(1, sequence_length, self.hidden_size),
                        dtype="float32",
                    )

        return spec

    def _extra_graph_inputs(
        self, name: str, sequence_length: int, context_length: int
    ) -> TensorSpec | None:
        """Spec for a family-specific graph input, or None if not recognized.

        Base returns None for everything (text LLMs have no extra inputs).
        Subclasses (e.g. VLMs with deepstack tensors) override this.
        """
        return None

    def get_graph_output_spec(self, graph_name: str) -> OutputSpec:
        """Output names for this Part, sanitized for the on-device runtime.

        All graphs of a Part share the same outputs, read from the split ONNX
        model at runtime.
        """
        return {
            name.replace("/", "_").replace(".", "_"): TensorSpec()
            for name in self._get_onnx_output_names()
        }


class DynamicSplitPartBase(LLMPartBase, torch.nn.Module, MultiGraphWorkbenchModel):
    """
    Unified Part base for dynamic-shape (dynamo) split LLMs.

    Handles both FP and Quantizable modes based on precision. Each Part
    represents one split of the ONNX model for deployment. When precision is
    float, uses the FP PreSplit (ONNX ModelProto inference). When precision is
    quantized, uses the Quantizable PreSplit (ONNXBundle + encodings).

    This base is model-family agnostic. Concrete models (or their family base)
    supply the architecture constants (``part_id``, ``num_splits``,
    ``hidden_size``, ``num_attention_heads``, ``num_key_value_heads``, and
    optionally ``head_dim`` when it differs from
    ``hidden_size // num_attention_heads``) and the concrete FP / Quantizable
    PreSplit classes (``fp_presplit_cls`` / ``quant_presplit_cls``) via class
    attributes.

    Models needing to adjust quantizers/encodings before loading (e.g.
    tied-embedding ``Gather``-fed weights) override
    ``transform_split_encodings``.
    """

    # --- per-model configuration (override in subclass) ---
    part_id: int = 0  # 1-indexed
    num_splits: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    default_precision: Precision = Precision.w4a16
    # The concrete FP / Quantizable PreSplit classes. Typed Any because the FP
    # and quantized PreSplits live in separate hierarchies.
    fp_presplit_cls: Any
    quant_presplit_cls: Any

    def __init__(
        self,
        presplit: Any,
        precision: Precision | None = None,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
    ) -> None:
        super().__init__()
        self._presplit = presplit
        self._precision = precision or self.default_precision
        self._quant_sim: QuantizationSimModel | None = None
        self._fp_session: onnxruntime.InferenceSession | None = None
        self._sequence_lengths = sequence_lengths
        self._context_lengths = context_lengths
        self._graph_names: dict[str, tuple[int, int]] = self._build_graph_names(
            self._sequence_lengths, self._context_lengths
        )

    def _build_graph_names(
        self, sequence_lengths: list[int], context_lengths: list[int]
    ) -> dict[str, tuple[int, int]]:
        return {
            f"{'token' if seq_len == 1 else 'prompt'}_ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{self.num_splits}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(sequence_lengths, context_lengths)
        }

    def restrict_to_single_instantiation(self) -> None:
        """Reduce this Part to a single (seq_len, ctx_len) graph."""
        self._context_lengths = [max(self._context_lengths)]
        self._sequence_lengths = [max(self._sequence_lengths)]
        self._graph_names = self._build_graph_names(
            self._sequence_lengths, self._context_lengths
        )

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
        context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS,
        sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS,
        **kwargs: Any,
    ) -> Self:
        """Create Part by getting or creating the appropriate PreSplit (cached)."""
        checkpoint_type = CheckpointType.from_checkpoint(checkpoint)
        if not checkpoint_type.is_aimet_onnx():
            presplit: Any = cls.fp_presplit_cls.from_pretrained(host_device=host_device)
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
        return cls(
            presplit,
            precision=precision,
            context_lengths=context_lengths,
            sequence_lengths=sequence_lengths,
        )

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        """Get sample inputs for this specific part only.

        Uses actual ONNX input names read from the split model at runtime.
        When called from the multi-graph sample_inputs path, input_spec
        carries the per-graph shapes so we derive seq_len from it.
        """
        # Derive seq_len from input_spec when available (multi-graph path).
        if input_spec is not None and "input_ids" in input_spec:
            seq_len = input_spec["input_ids"][0][1]  # shape (1, seq_len)
        else:
            seq_len, _ = self._graph_names[graph_name]

        full_inputs = self._presplit._sample_inputs_impl()

        if self.part_id == 1:
            # Embedding split: only input_ids
            return {"input_ids": [np.zeros((1, seq_len), dtype=np.int32)]}

        # Parts 2+: read actual input names from ONNX and match them.
        # Use the graph input spec to generate correctly-shaped samples for
        # inputs not found in full_inputs (e.g. native KV with local layer
        # indices that differ from the full model's global indices).
        graph_spec = self.get_graph_input_spec(graph_name)
        result: SampleInputsType = {}
        onnx_input_names = self._get_onnx_input_names()

        for name in onnx_input_names:
            if name in full_inputs:
                result[name] = full_inputs[name]
            elif name in graph_spec:
                spec = graph_spec[name]
                dtype = np.dtype(spec.dtype) if spec.dtype else np.float32
                result[name] = [np.zeros(spec.shape, dtype=dtype)]
            else:
                result[name] = [
                    np.zeros((1, seq_len, self.hidden_size), dtype=np.float32)
                ]

        return result

    # -------------------------------------------------------------------
    # Methods that branch on self._is_quantized
    # -------------------------------------------------------------------

    def _get_onnx_input_names(self) -> list[str]:
        """Read actual input names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [i.name for i in onnx_model.graph.input]

    def _get_onnx_output_names(self) -> list[str]:
        """Read actual output names from split ONNX model."""
        onnx_bundle = self._get_onnx_bundle()
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=False
        )
        return [o.name for o in onnx_model.graph.output]

    def _get_onnx_bundle(self) -> ONNXBundle:
        """Get ONNXBundle for this Part (works for both FP and quantized)."""
        return self._presplit.convert_to_onnx_and_split(part_id=self.part_id)

    def transform_split_encodings(
        self,
        onnx_model: onnx.ModelProto,
        quant_sim: QuantizationSimModel,
        encodings_path: str,
    ) -> str:
        """Adjust QuantSim quantizers and/or rewrite encodings before loading.

        Called by :meth:`_get_quant_sim` immediately before
        ``load_encodings_to_sim``. The default is a no-op that returns the
        encodings path unchanged (the historical plain-load behavior).

        Override per-model for special cases -- e.g. tied-embedding models
        whose single shared initializer feeds an embedding ``Gather`` (no
        per-channel axis) need to relax that quantizer to fp16 and drop its
        per-channel param entry before loading.

        Returns the path to the encodings file to load (may be a rewritten
        copy or the original ``encodings_path``).
        """
        return encodings_path

    def _get_quant_sim(self) -> QuantizationSimModel:
        """Get or create QuantSim for this specific part from its ONNXBundle."""
        if self._quant_sim is not None:
            return self._quant_sim

        onnx_bundle = self._get_onnx_bundle()

        # Load ONNX model
        onnx_model = onnx.load(
            str(onnx_bundle.onnx_graph_path), load_external_data=True
        )

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Clamp to keep QuantSim compatible.
        onnx_model.ir_version = min(onnx_model.ir_version, 10)

        assert isinstance(self._presplit, self.quant_presplit_cls)
        _hd = self._presplit.host_device
        host_device = _hd if isinstance(_hd, torch.device) else torch.device("cpu")
        providers = self._presplit.get_ort_providers(host_device)

        # Use shared construction + activation config. We skip the
        # full-model _configure_quant_sim (lm_head, KV tying) since
        # those heuristics misfire on split models.
        self._quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
        LLMDynamic_AIMETOnnx._apply_precision_activations(
            self._quant_sim, self._precision
        )

        # Load encodings if available. Models with special encoding needs
        # (e.g. tied embeddings) override transform_split_encodings to adjust
        # the quantizers / rewrite the encodings file first; the default is a
        # no-op plain load.
        if onnx_bundle.aimet_encodings_path is not None:
            encodings_path = self.transform_split_encodings(
                onnx_model,
                self._quant_sim,
                str(onnx_bundle.aimet_encodings_path),
            )
            load_encodings_to_sim(self._quant_sim, encodings_path, strict=False)

        return self._quant_sim

    def _get_fp_session(self) -> onnxruntime.InferenceSession:
        """Get or create an ORT session for FP inference (cached)."""
        if self._fp_session is not None:
            return self._fp_session

        onnx_bundle = self._get_onnx_bundle()

        providers: list[str | tuple[str, dict[str, str | int]]] = [
            "CPUExecutionProvider"
        ]
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.insert(
                0,
                (
                    "CUDAExecutionProvider",
                    {"arena_extend_strategy": "kSameAsRequested"},
                ),
            )

        # Dynamo export (opset 18) produces IR version 11, but ORT 1.x
        # only supports up to 10.  Patch the file in-place (graph-only
        # load keeps external weight references intact).
        onnx_path = str(onnx_bundle.onnx_graph_path)
        onnx_model = onnx.load(onnx_path, load_external_data=False)
        if onnx_model.ir_version > 10:
            onnx_model.ir_version = 10
            onnx.save(onnx_model, onnx_path)

        self._fp_session = onnxruntime.InferenceSession(onnx_path, providers=providers)
        return self._fp_session

    def forward(
        self, *args: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor | Collection[torch.Tensor]:
        """Forward pass for this Part (FP or quantized based on precision)."""
        if self._is_quantized:
            quant_sim = self._get_quant_sim()
            return mock_torch_onnx_inference(quant_sim.session, *args, **kwargs)
        session = self._get_fp_session()
        return mock_torch_onnx_inference(session, *args, **kwargs)

    @property
    def shared_source_model(self) -> bool:
        return True

    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Export ONNX model for this Part."""
        model_name = self.__class__.__name__

        ext = ".aimet" if self._is_quantized else ".onnx"
        # Include precision in directory name to avoid cache collisions
        # between different precisions sharing the same output_path.
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

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return super().get_graph_hub_compile_options(
            graph_name,
            target_runtime,
            precision,
            other_compile_options + " --quantize_full_type w8a16 --quantize_io",
            device,
        )

    def get_graph_hub_inference_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> str:
        if self._is_quantized:
            return self._presplit.get_hub_inference_options(
                target_runtime=target_runtime,
                other_inference_options=other_inference_options,
                context_graph_name=graph_name,
            )
        return super().get_graph_hub_inference_options(
            graph_name,
            target_runtime=target_runtime,
            other_inference_options=other_inference_options,
        )


class LLMCollectionBase(MultiGraphWorkbenchModelCollection):
    """Collection base shared by every LLM / VLM; sets LLM-wide compile options."""

    def get_component_graph_hub_compile_options(
        self,
        component_name: str,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        return super().get_component_graph_hub_compile_options(
            component_name,
            graph_name,
            target_runtime,
            precision,
            other_compile_options + " -O0",
            device,
        )


class DynamicSplitCollectionBase(LLMCollectionBase):
    """
    Unified Collection base with N Parts for a dynamic-shape split LLM.

    Supports both FP and Quantizable modes based on precision parameter.
    All Parts share the same PreSplit via class-level cache for memory
    efficiency.

    Concrete models subclass this and set the config attributes:
    ``hf_repo_name`` (HuggingFace repo for tokenizer/config),
    ``fp_presplit_cls`` (the FP PreSplit class, for chat template / sample
    prompt), ``part_base_cls`` (the model's Part base, for the isinstance
    check), ``supports_thinking`` (whether the model supports a thinking mode,
    propagated to ``GenieMetadata``), ``think_start`` / ``think_end`` (tags
    delimiting the reasoning span, attached to the emitted
    ``GenieChatTemplate``), and ``parts`` (an ordered mapping from
    component name to Part class).
    """

    hf_repo_name: str = ""
    fp_presplit_cls: Any
    part_base_cls: Any
    supports_thinking: bool = False
    think_start: str = ""
    think_end: str = ""
    think_header: str = ""
    parts: dict[str, Any]
    # Default (seq_len, ctx_len) graphs to emit. Override in subclasses that
    # need non-standard sequence lengths (e.g. SSD uses an AR-32 token graph).
    default_sequence_lengths: list[int] = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    default_context_lengths: list[int] = DEFAULT_EXPORT_CONTEXT_LENGTHS

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = "DEFAULT",
        host_device: torch.device | None = None,
        _skip_quantsim_creation: bool = True,
        # Empty lists are a read-only "use cls defaults" sentinel (never mutated).
        sequence_lengths: list[int] = [],  # noqa: B006
        context_lengths: list[int] = [],  # noqa: B006
    ) -> Self:
        """
        Create Collection with all Parts.

        Parameters
        ----------
        checkpoint
            Path to checkpoint with ONNX + encodings, or ``"DEFAULT"``
            to create from HuggingFace.
        host_device
            Device for computation.
        _skip_quantsim_creation
            Skip QuantSim creation (for testing).
        sequence_lengths
            Sequence lengths to compile for. Empty means use
            ``cls.default_sequence_lengths``.
        context_lengths
            Context lengths to compile for. Empty means use
            ``cls.default_context_lengths``.

        Returns
        -------
        Self
            The Collection with all Parts.
        """
        sequence_lengths = sequence_lengths or cls.default_sequence_lengths
        context_lengths = context_lengths or cls.default_context_lengths
        return cls(
            {
                part_name: part_cls.from_pretrained(
                    checkpoint=checkpoint,
                    host_device=host_device,
                    _skip_quantsim_creation=_skip_quantsim_creation,
                    sequence_lengths=sequence_lengths,
                    context_lengths=context_lengths,
                )
                for part_name, part_cls in cls.parts.items()
            }
        )

    def restrict_to_single_instantiation(self) -> None:
        """Reduce every Part to a single instantiation."""
        for component in self.components.values():
            assert isinstance(component, self.part_base_cls)
            component.restrict_to_single_instantiation()

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        output_path = Path(output_dir)

        # Tokenizer/config source, in preference order: output_path (idempotent
        # re-run), then the first Part's local checkpoint, then hf_repo_name.
        # The checkpoint comes second because Gemma4's QAT checkpoint ships
        # tokenizer.json / config.json but its hf_repo_name is a private repo
        # that may be inaccessible at export time.
        first_part = next(iter(self.components.values()))
        _presplit = getattr(first_part, "_presplit", None)
        _ckpt_dir = str(getattr(_presplit, "checkpoint", None) or "")
        _ckpt_has_tokenizer = bool(
            _ckpt_dir and (Path(_ckpt_dir) / "tokenizer.json").exists()
        )
        _ckpt_has_config = bool(
            _ckpt_dir and (Path(_ckpt_dir) / "config.json").exists()
        )

        # Write sample_prompt.txt for on-device tests
        if (output_path / "tokenizer.json").exists():
            _tok_src = str(output_path)
        elif _ckpt_has_tokenizer:
            _tok_src = _ckpt_dir
        else:
            _tok_src = self.hf_repo_name
        tokenizer = AutoTokenizer.from_pretrained(_tok_src)
        # Thinking off, as for eval prompts: an unbounded trace can exhaust the context
        # window mid-sweep. Non-thinking templates ignore the unused kwarg.
        sample_prompt = self.fp_presplit_cls.get_input_prompt_with_tags(
            tokenizer=tokenizer, enable_thinking=False
        )
        # Save tokenizer and config from HuggingFace (skip if already present)
        if not (output_path / "tokenizer.json").exists():
            # GenieX reads the chat_template field embedded inside tokenizer_config.json.
            # save_jinja_files=False keeps it embedded instead of writing a separate chat_template.jinja file.
            tokenizer.save_pretrained(output_path, save_jinja_files=False)
        if not (output_path / "config.json").exists():
            # Prefer the Part's already-loaded config when the checkpoint dir
            # ships config.json, for the private-repo reason above.
            llm_config = (
                getattr(_presplit, "llm_config", None) if _ckpt_has_config else None
            ) or AutoConfig.from_pretrained(self.hf_repo_name)
            llm_config.save_pretrained(output_path)
        else:
            llm_config = AutoConfig.from_pretrained(str(output_path))

        # Multimodal configs (Gemma4Config, Qwen3VLConfig) nest the text
        # hyperparameters under `.text_config`, but create_genie_config reads
        # hidden_size / num_attention_heads / vocab_size / rope off the top
        # level. No-op for text-only LLMs.
        llm_config = getattr(llm_config, "text_config", llm_config)

        # Derive context_lengths from the first part
        first_part = next(iter(self.components.values()))
        assert isinstance(first_part, self.part_base_cls)
        context_lengths: list[int] = first_part._context_lengths
        context_length: int = max(context_lengths)

        # Build genie_config.json
        model_list = list(metadata.model_files.keys())
        config = create_genie_config(context_length, llm_config, "rope", model_list)
        with open(output_path / "genie_config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Build htp_backend_ext_config.json from chipset attributes
        device_info: dict[str, str] = {}
        if metadata.chipset_attributes:
            ca = metadata.chipset_attributes
            if ca.htp_version is not None:
                device_info["hexagon"] = f"v{ca.htp_version}"
            if ca.soc_model is not None:
                device_info["soc-model"] = str(ca.soc_model)
        if save_htp_config_for_genie_bundle(device_info, output_path):
            metadata.supplementary_files["htp_backend_ext_config.json"] = (
                "HTP backend configuration for the target device."
            )

        with open(output_path / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)

        chat_spec = self.fp_presplit_cls.get_chat_template()
        chat_template = (
            GenieChatTemplate(**cast(dict[str, Any], chat_spec))
            if chat_spec
            else GenieChatTemplate()
        )
        if self.supports_thinking:
            chat_template.think_start = self.think_start
            chat_template.think_end = self.think_end
            chat_template.think_header = self.think_header
        metadata.genie = GenieMetadata(
            chat_template=chat_template,
            context_lengths=context_lengths,
            supports_streaming=True,
            supports_vision=False,
            supports_thinking=self.supports_thinking,
        )

        metadata.supplementary_files["genie_config.json"] = (
            "Genie SDK configuration for on-device LLM inference."
        )
        metadata.supplementary_files["sample_prompt.txt"] = (
            "Sample prompt for on-device inference."
        )
        metadata.supplementary_files["tokenizer.json"] = (
            "Tokenizer for encoding/decoding text."
        )


class LLMBase(BaseModel, LLMConfigEditor, ABC):
    # The Hugging Face LLM class (e.g., LlamaForCausalLM)
    LMClass: Any | None = None

    # Embedding subclass
    EmbeddingClass: type[Embedding] | None = None

    # IO signature
    llm_io_type: LLMIOType = LLMIOType.genie_input_ids

    # Minimum recommended memory for exporting (in GB)
    min_memory_recommended: int = 0

    # Default prompts for demos (override in subclasses)
    default_user_prompt: str = "What is gravity?"
    default_system_prompt: str = "You are a helpful AI assistant."

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        """Return chat template tokens as a dict suitable for GenieChatTemplate(**d).

        Subclasses that define explicit chat template tokens should override
        this.  The base implementation returns an empty dict (template is only
        known to the HuggingFace tokenizer).

        Returns
        -------
        dict[str, str]
            Keys: system_prefix, system_suffix, user_prefix, user_suffix,
            assistant_prefix, assistant_suffix, default_system_prompt.
            Optionally: vision_start, vision_end.
        """
        return {}

    @classmethod
    def get_input_prompt_with_tags(
        cls,
        user_input_prompt: str | None = None,
        system_context_prompt: str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        include_image: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Format a prompt using the tokenizer's chat template.

        Parameters
        ----------
        user_input_prompt
            The user's input prompt. Defaults to cls.default_user_prompt.
        system_context_prompt
            System context/instructions. Defaults to cls.default_system_prompt.
        tokenizer
            Tokenizer for applying chat template. Required for base implementation.
        include_image
            For VLM models, whether to include vision placeholder tokens.
            VLM subclasses should override to handle this parameter.
            Ignored by default (non-VLM models).
        **kwargs
            Additional arguments passed to tokenizer.apply_chat_template.

        Returns
        -------
        str
            Formatted prompt string.
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required for get_input_prompt_with_tags")
        if user_input_prompt is None:
            user_input_prompt = cls.default_user_prompt
        if system_context_prompt is None:
            system_context_prompt = cls.default_system_prompt
        messages = []
        if system_context_prompt:
            messages.append({"role": "system", "content": system_context_prompt})
        messages.append({"role": "user", "content": user_input_prompt})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        assert isinstance(prompt, str)
        return prompt

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        sequence_length: int,
        context_length: int,
        is_token_generator: bool = False,
        load_pretrained: bool = True,
        host_device: torch.device | None = None,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
        _skip_optimizations: list[str] | None = None,
    ) -> None:
        """
        This is an abstract base class of all LLM models.

        Parameters
        ----------
        checkpoint
            Can be local folder or Hugging Face repo name.
        sequence_length
            Input sequence length (in tokens).
        context_length
            Total context length (in tokens).
        is_token_generator
            Whether this is a token generator model.
        load_pretrained
            Load a pre-trained model as opposed to a randomly initialized.
        host_device
            Device to use: GPU/CPU.
        attention_mask_min_clip
            Min clip the attention mask by this value if not None.
        attention_mask_multiplier
            Bake in a multiplier for the attention mask into the network.
            This is useful to conform to Genie's unconfigurable -1000
            "infinity" for FP16 activations, when a network may require an even
            larger (in magnitude) value.
        _skip_optimizations
            Turn off one or more of {sha_attention, rank4_rms_norm}.
        """
        super().__init__()
        self.skip_optimizations = _skip_optimizations
        self.checkpoint = checkpoint

        # Ensure User has recommended memory,
        # otherwise, provide warning to user and recommend to increase swap-space as a work-around.
        has_recommended_memory(self.min_memory_recommended)

        # TODO: Make this into a context manager
        self.monkey_patch(skip_optimizations=self.skip_optimizations)
        llm_config = get_llm_config(self.checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        self._verify_ckpt()
        self.tokenizer = get_tokenizer(checkpoint)
        assert self.LMClass is not None
        if load_pretrained:
            model = self.LMClass.from_pretrained(
                self.checkpoint,
                config=self.llm_config,
                ignore_mismatched_sizes=False,
            )
        else:
            model = self.LMClass(self.llm_config)
        model.eval().float()

        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "0"

        for _, module in model.named_modules():
            if hasattr(module, "prepare_conv"):
                module.prepare_conv()
            if hasattr(module, "prepare_sha"):
                module.prepare_sha()
            if hasattr(module, "use_native_kv"):
                module.use_native_kv = is_native_kv(self.llm_io_type)

        model.to(host_device)

        self.sequence_length: int = sequence_length
        self.context_length: int = context_length
        self.split_part = 1
        self.is_token_generator = is_token_generator
        self.model = model
        self.attention_mask_min_clip = attention_mask_min_clip
        self.attention_mask_multiplier = attention_mask_multiplier

    def get_input_spec(
        self,
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Model configuration dictionary.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        llm_io_type
            Input/output type for the LLM.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        raise NotImplementedError

    @staticmethod
    def monkey_patch(
        skip_optimizations: list[str] | None = None,
    ) -> None:
        pass

    def _verify_ckpt(self) -> None:
        # Override in baseclass to verify compatibility with config
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> Self:
        pass

    @staticmethod
    def _get_output_spec(num_hidden_layers: int) -> OutputSpec:
        output_spec: OutputSpec = {"logits": TensorSpec()}
        for layer in range(num_hidden_layers):
            output_spec[f"past_key_{layer}_out"] = TensorSpec()
            output_spec[f"past_value_{layer}_out"] = TensorSpec()
        return output_spec

    # Must be defined by transformers generator class
    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "input_embeds"
        return "input_ids"

    def forward(
        self,
        input_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.llm_io_type == LLMIOType.huggingface_input_ids:
            position_ids = args[0]
            past_key_values = args[1:]
        else:
            # contains (position_ids_cos, position_ids_sin)
            position_ids = args[:2]  # type: ignore[assignment, unused-ignore]
            past_key_values = args[2:]

        # Native KV: cache_index is the last element after all KV pairs
        cache_index: torch.Tensor | None = None
        if is_native_kv(self.llm_io_type):
            cache_index = past_key_values[-1]
            past_key_values = past_key_values[:-1]

        assert isinstance(self.llm_config.num_key_value_heads, int)
        num_kv_heads = self.llm_config.num_key_value_heads
        if self.skip_optimizations and "sha_attention" in self.skip_optimizations:
            kv_cache = DynamicCache()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2], strict=False)
            ):
                k_split = [k[i : i + 1] for i in range(num_kv_heads)]
                v_split = [v[i : i + 1] for i in range(num_kv_heads)]
                k = torch.cat(k_split, dim=1).permute(0, 1, 3, 2)
                v = torch.cat(v_split, dim=1)

                kv_cache.update(k, v, layer_idx, {})
        elif is_native_kv(self.llm_io_type):
            kv_cache = SHADynamicCacheNewValueOnly()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2], strict=False)
            ):
                k_heads = list(k.chunk(num_kv_heads, dim=0))
                v_heads = list(v.chunk(num_kv_heads, dim=0))
                kv_cache.update(k_heads, v_heads, layer_idx, {})
        else:
            kv_cache = SHADynamicCacheNewValueOnly()
            for layer_idx, (k, v) in enumerate(
                zip(past_key_values[::2], past_key_values[1::2], strict=False)
            ):
                k_split = [k[i : i + 1] for i in range(num_kv_heads)]
                v_split = [v[i : i + 1] for i in range(num_kv_heads)]

                # kv_cache doesn't report supporting lists of tensors, but it seems to work
                kv_cache.update(k_split, v_split, layer_idx, {})

        if (
            cache_index is not None
            and hasattr(kv_cache, "set_valid_length")
            and not torch.compiler.is_compiling()
        ):
            kv_cache.set_valid_length(int(cache_index.item()))

        model_kwargs: dict[str, Any] = {
            self.main_input_name: input_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": kv_cache,
        }

        # Compute cache_position from cache_index via tensor ops so the
        # ONNX optimizer keeps cache_index as a dynamic input.
        if cache_index is not None:
            seq_len = input_tokens.shape[1]
            cache_position = cache_index + torch.arange(
                seq_len, device=cache_index.device, dtype=torch.int32
            )
            model_kwargs["cache_position"] = cache_position

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

            elif is_native_kv(self.llm_io_type):
                if hasattr(out_cache, "key_cache"):
                    k_heads = out_cache.key_cache[layer]
                    v_heads = out_cache.value_cache[layer]
                elif hasattr(out_cache.layers[layer], "keys"):
                    k_heads = out_cache.layers[layer].keys
                    v_heads = out_cache.layers[layer].values
                else:
                    k_heads = out_cache.layers[layer][0]
                    v_heads = out_cache.layers[layer][1]
                # The cache holds only the new slices; gathering them back out of
                # the scattered buffer would break HTP's native-KV pattern.
                k = torch.cat(k_heads, dim=0)
                v = torch.cat(v_heads, dim=0)

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

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)  # type: ignore[operator, unused-ignore]

    @staticmethod
    def _get_input_spec(
        num_hidden_layers: int,
        sequence_length: int,
        context_length: int,
        hidden_size: int,
        num_key_value_heads: int,
        num_attention_heads: int,
        head_dim: int | None = None,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        # Use explicit head_dim if provided, otherwise derive from hidden_size
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        embed_dim = head_dim // 2
        input_spec: InputSpec = {}

        if llm_io_type == LLMIOType.genie_input_embeds:
            input_spec |= {
                "input_embeds": TensorSpec(
                    shape=(1, sequence_length, hidden_size),
                    dtype="float32",
                )
            }
        else:
            input_spec |= {
                "input_ids": TensorSpec(shape=(1, sequence_length), dtype="int32")
            }

        input_spec |= {
            "attention_mask": TensorSpec(
                shape=(1, 1, sequence_length, context_length),
                dtype="float32",
            ),
        }

        if llm_io_type == LLMIOType.huggingface_input_ids:
            input_spec |= {
                "position_ids": TensorSpec(
                    shape=(1, sequence_length),
                    dtype="int32",
                ),
            }
        else:
            input_spec |= {
                # each cos/sin are applied to a half-sliced copy of the hidden size
                # and then concatenated.
                "position_ids_cos": TensorSpec(
                    shape=(1, 1, sequence_length, embed_dim),
                    dtype="float32",
                ),
                "position_ids_sin": TensorSpec(
                    shape=(1, 1, sequence_length, embed_dim),
                    dtype="float32",
                ),
            }

        # TODO: We could support sequence_length == CONTEXT_LENGTH, but the
        # KV cache input needs to be removed.
        assert sequence_length < context_length, (
            "It is currently not supported to set input sequence length to the same as or longer than context length. There should be no KV cache input at all in such case."
        )

        use_native_kv = is_native_kv(llm_io_type)
        prefix = "past_nativekvcache_" if use_native_kv else "past"
        kv_seq_len = (
            context_length if use_native_kv else context_length - sequence_length
        )

        for layer in range(num_hidden_layers):
            input_spec[f"{prefix}_key_{layer}_in"] = TensorSpec(
                shape=(num_key_value_heads, 1, head_dim, kv_seq_len),
                dtype="float32",
            )
            input_spec[f"{prefix}_value_{layer}_in"] = TensorSpec(
                shape=(num_key_value_heads, 1, kv_seq_len, head_dim),
                dtype="float32",
            )

        if use_native_kv:
            input_spec["cache_index"] = TensorSpec(shape=(1,), dtype="int32")

        return input_spec

    def get_calibration_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
    ) -> DatasetEntries | None:
        # No calibration data needed
        return None

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_io_type=self.llm_io_type,
                llm_config=self.llm_config.to_dict(),
            )
        return sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )

    def get_evaluator(
        self,
        task: str = "wikitext",
        device: torch.device = torch.device("cpu"),
        **task_kwargs: Any,
    ) -> BaseEvaluator:
        return _get_evaluator(
            task,
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            device,
            **task_kwargs,
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        from qai_hub_models.datasets.grace import Grace2
        from qai_hub_models.datasets.mmlu import mmmlu_dataset_classes

        return [*mmmlu_dataset_classes(), Grace2]

    def release(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.to("cpu")
            del self.model

    def __del__(self) -> None:
        # Clean up since it is prone to hang onto GPU memory otherwise
        self.release()
        # Python can be in a weird state when __del__ gets called, so we
        # have to make sure these still exist.
        if "gc" in globals() and gc is not None:
            gc.collect()
        if "torch" in globals() and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


@unique
class LLMInstantiationType(Enum):
    """
    Types of an LLM "Instantiation"
    Export instantiates 2 copies of the LLM, each with a different sequence length.
    """

    PROMPT_PROCESSOR = "prompt"
    TOKEN_GENERATOR = "token"


LLM_AIMETOnnxT = TypeVar("LLM_AIMETOnnxT", bound="LLM_AIMETOnnx")


class LLM_AIMETOnnx(AIMETOnnxQuantizableMixin, LLMConfigEditor, BaseModel, ABC):
    # Embedding subclass
    EmbeddingClass: type[Embedding] | None = None

    # PyTorch equivalent of this class
    FPModel: type[LLMBase] | None = None

    # Whether the model supports thinking mode (e.g., Qwen3 models).
    supports_thinking: bool = False

    split_lm_head: bool = False

    supported_precisions: list[Precision]

    # Param tensors held at int8 instead of the precision's default weight
    # bitwidth: their int4 matmul saturates the HTP accumulator (HEXNNVVV-9471),
    # so on-target output diverges from QuantSim. Re-quantization only -- a
    # published asset must also ship these at bw 8 in its model.encodings.
    # QAIRT 2.49's "precision_compensation" should remove the need for these
    # lists; see https://github.com/qcom-ai-hub/tetracode/issues/20616.
    int8_param_names: tuple[str, ...] = ()
    # Parameter tensors held at fp16 while the rest of the model stays at the
    # precision default (currently used with w4a16).
    fp16_param_names: tuple[str, ...] = ()

    @classmethod
    def get_chat_template(cls) -> dict[str, str]:
        """Delegate to FPModel's get_chat_template."""
        assert cls.FPModel is not None
        return cls.FPModel.get_chat_template()

    @classmethod
    def get_input_prompt_with_tags(cls, **kwargs: Any) -> str:
        """Delegate to FPModel's get_input_prompt_with_tags."""
        assert cls.FPModel is not None
        return cls.FPModel.get_input_prompt_with_tags(**kwargs)

    def __init__(
        self,
        quant_sim: QuantizationSimModel | None,
        checkpoint: str | os.PathLike | Path | None,
        sequence_length: int,
        context_length: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
        llm_config: PretrainedConfig | None = None,
        host_device: torch.device | None = None,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        BaseModel.__init__(self)
        AIMETOnnxQuantizableMixin.__init__(self, quant_sim)
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.host_device = host_device

        assert (
            tokenizer is not None and llm_config is not None
        ) or checkpoint is not None, (
            f"{self.__class__.__name__} is unable to instantiate tokenizer/config. Must pass either checkpoint or tokenizer/config explicitly."
        )

        self.tokenizer = tokenizer or get_tokenizer(checkpoint)
        llm_config = llm_config or get_llm_config(checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )
        self.checkpoint = checkpoint
        self.attention_mask_min_clip = attention_mask_min_clip
        self.attention_mask_multiplier = attention_mask_multiplier

    def __del__(self) -> None:
        if hasattr(self, "quant_sim") and self.quant_sim is not None:
            del self.quant_sim
            # Python can be in a weird state when __del__ gets called, so we
            # have to make sure these still exist.
            if "gc" in globals() and gc is not None:
                gc.collect()
            if "torch" in globals() and torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def release(self) -> None:
        free_memory = getattr(self, "free_memory", None)
        if callable(free_memory):
            free_memory()
        self.to("cpu")
        if hasattr(self, "quant_sim") and self.quant_sim is not None:
            del self.quant_sim
            self.quant_sim = None
        self._released = True

    def get_input_spec(
        self,
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Model configuration dictionary.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        llm_io_type
            Input/output type for the LLM.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        raise NotImplementedError

    @property
    def llm_io_type(self) -> LLMIOType:
        assert self.FPModel is not None
        return self.FPModel.llm_io_type

    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "input_embeds"
        return "input_ids"

    def _sample_inputs_impl(  # type: ignore[override]
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_config=self.llm_config.to_dict(),
            )
        assert self.FPModel is not None
        return sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:  # type: ignore[override]
        # This must be defined by the HubModelProtocol protocol via BaseModel
        return self._sample_inputs_impl(input_spec)

    # ------------------------------------------------------------------
    # Phased quantize protocol
    # ------------------------------------------------------------------

    @classmethod
    def export_onnx(
        cls,
        fp_model: LLMBase,
        output_dir: Path,
        **kwargs: Any,
    ) -> None:
        """Export backbone ONNX to output_dir/model_dynamic.onnx + model.data.

        VLM subclasses override this to also export VEG ONNX and embedding.
        """
        # Every deployable LLM/VLM FP PreSplit is a DynamicPreSplitOnnxMixin;
        # the dynamic-shape ONNX bundle is the only supported export path.
        assert isinstance(fp_model, DynamicPreSplitOnnxMixin), (
            f"{type(fp_model).__name__} is not a DynamicPreSplitOnnxMixin; "
            "only dynamic-shape ONNX export is supported."
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            bundle = fp_model.get_full_onnx_bundle(Path(tmp))
            shutil.copy(bundle.onnx_graph_path, output_dir / "model_dynamic.onnx")
            bundle_data = bundle.onnx_graph_path.parent / "model.data"
            if bundle_data.exists():
                shutil.copy(bundle_data, output_dir / "model.data")

        # Save tokenizer + config so the checkpoint is self-contained
        fp_model.tokenizer.save_pretrained(str(output_dir))
        fp_model.llm_config.save_pretrained(str(output_dir))

    @classmethod
    def apply_pre_sim_transforms(
        cls,
        output_dir: Path,
        spinquant_config: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Apply pre-QuantSim transforms (e.g. SpinQuant) to ONNX on disk.

        Mutates files in output_dir in-place. No-op if no transforms configured.
        VLM subclasses override to co-rotate backbone + VEG + embedding.
        """
        if not spinquant_config:
            return

        backbone_path = output_dir / "model_dynamic.onnx"
        backbone_onnx = onnx.load(str(backbone_path), load_external_data=True)

        cls.apply_spinquant_to_onnx(backbone_onnx, spinquant_config)

        onnx.save_model(
            backbone_onnx,
            str(backbone_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.data",
        )
        del backbone_onnx
        gc.collect()

    @classmethod
    def _load_onnx_and_build_quantsim(
        cls,
        output_dir: Path,
        host_device: torch.device,
        precision: Precision,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> QuantizationSimModel:
        """Load ONNX from disk, create QuantSim, and load encodings.

        This handles the mechanical steps of discovering the ONNX file,
        building a QuantizationSimModel, and loading any saved encodings.
        Instance construction is handled by the caller (from_pretrained).
        """
        if AIMET_ONNX_INSTALLED:
            AimetLogger.set_level_for_all_areas(logging.WARNING)

        # Discover ONNX file
        onnx_path = output_dir / "model_dynamic.onnx"
        if not onnx_path.exists():
            # Fall back to legacy fixed-shape format
            onnx_path = (
                output_dir / f"model_seqlen{sequence_length}_cl{context_length}.onnx"
            )
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"No ONNX model found in {output_dir}. Run export_onnx first."
            )

        print()
        print(f"Loading onnx model from {onnx_path}")
        onnx_model = onnx.load(str(onnx_path), load_external_data=False)
        load_external_data_for_model(onnx_model, str(output_dir))

        print()
        print("Creating a QuantSim model using AIMET ONNX.")
        quant_sim = cls.create_quantsim(onnx_model, host_device, precision)
        del onnx_model
        gc.collect()

        # Load encodings if present (re-quantize / resume / inference)
        aimet_encodings = output_dir / "model.encodings"
        if aimet_encodings.exists():
            print()
            print(f"Loading the encodings from {aimet_encodings}")
            load_encodings_to_sim(quant_sim, str(aimet_encodings), strict=False)

        return quant_sim

    @classmethod
    def _has_onnx_on_disk(
        cls, ckpt: Path, sequence_length: int, context_length: int
    ) -> bool:
        """Check whether a checkpoint directory holds a complete export.

        For the base backbone this means the ONNX graph + its external data.
        VLM subclasses override to additionally require their extra export
        artifacts (vision encoder ONNX, embedding weights) so that a run
        interrupted mid-export is not mistaken for a complete checkpoint --
        which would otherwise cause quantize() to skip re-export and leave
        stale/missing VEG/embedding artifacts inconsistent with the backbone.
        """
        external_data_exists = (ckpt / "model.onnx.data").exists() or (
            ckpt / "model.data"
        ).exists()
        if not external_data_exists:
            return False
        if (ckpt / "model_dynamic.onnx").exists():
            return True
        legacy_path = ckpt / f"model_seqlen{sequence_length}_cl{context_length}.onnx"
        return legacy_path.exists()

    # ------------------------------------------------------------------
    # from_pretrained (delegates to phased protocol)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device,
        sequence_length: int,
        context_length: int,
        precision: Precision,
        fp_model: LLMBase | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Self:
        """
        Load from checkpoint or export+quantize an FP model.

        When checkpoint contains an ONNX model on disk, loads directly.
        When no ONNX exists and fp_model is provided, exports it before
        creating QuantSim. SpinQuant and other pre-sim transforms are applied
        by the quantize() flow (which pre-exports the rotated ONNX to the
        checkpoint dir), not here.

        Parameters
        ----------
        host_device
            Device to use: GPU/CPU.
        sequence_length
            Sequence Length for the model.
        context_length
            Context Length for the model.
        precision
            Precision to use for the model.
        fp_model
            Floating point version of this model.
        checkpoint
            Path to previously calibrated AIMET encodings and ONNX models.
        _skip_quantsim_creation
            If True, skip creating the QuantSim model.

        Returns
        -------
        Self
            Instance of the model loaded from the checkpoint.
        """
        if host_device is None:
            host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Capture fp_model attributes before it may be freed.
        _fp_tokenizer = fp_model.tokenizer if fp_model is not None else None
        _fp_llm_config = fp_model.llm_config if fp_model is not None else None
        _fp_embedding_weights = None
        if fp_model is not None and hasattr(fp_model, "_embedding_weights"):
            _fp_embedding_weights = fp_model._embedding_weights
        _fp_original_llm_config = None
        if fp_model is not None and hasattr(fp_model, "_original_llm_config"):
            _fp_original_llm_config = fp_model._original_llm_config

        quant_sim: QuantizationSimModel | None = None
        output_dir: Path | None = None
        tmp_dir: tempfile.TemporaryDirectory | None = None

        if _skip_quantsim_creation:
            # Lightweight instance (split-forward demo/export path)
            pass
        elif checkpoint is not None and cls._has_onnx_on_disk(
            Path(str(checkpoint)), sequence_length, context_length
        ):
            # ONNX on disk -- load directly (inference / pre-exported path)
            output_dir = Path(str(checkpoint))
            quant_sim = cls._load_onnx_and_build_quantsim(
                output_dir=output_dir,
                host_device=host_device,
                precision=precision,
                sequence_length=sequence_length,
                context_length=context_length,
            )
            if fp_model is not None:
                fp_model.to("cpu")
        elif checkpoint is not None and not cls._has_onnx_on_disk(
            Path(str(checkpoint)), sequence_length, context_length
        ):
            # Checkpoint provided, but ONNX not found - fail loudly
            raise ValueError(
                "The quantized checkpoint (with custom weights) must have "
                "an ONNX model."
            )
        else:
            # No ONNX on disk -- export, then load
            if fp_model is None:
                raise ValueError(
                    "The quantized checkpoint (with custom weights) must have "
                    "an ONNX model."
                )

            tmp_dir = tempfile.TemporaryDirectory()
            output_dir = Path(tmp_dir.name)

            cls.export_onnx(fp_model, output_dir)

            # Free fp_model memory before QuantSim creation to avoid OOM for
            # large models (e.g. 7B) where both PyTorch weights and the
            # AIMET-ONNX CUDA session don't fit on a single GPU.
            fp_model.to("cpu")
            del fp_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            quant_sim = cls._load_onnx_and_build_quantsim(
                output_dir=output_dir,
                host_device=host_device,
                precision=precision,
                sequence_length=sequence_length,
                context_length=context_length,
            )

        # -- Unified instance construction --
        attention_mask_min_clip, attention_mask_multiplier = (
            cls.attention_mask_min_clip_and_multiplier(precision)
        )
        init_kwargs: dict[str, Any] = dict(
            quant_sim=quant_sim,
            host_device=host_device,
            checkpoint=str(output_dir) if output_dir else checkpoint,
            tokenizer=_fp_tokenizer,
            llm_config=_fp_llm_config,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )
        if not issubclass(cls, LLMDynamic_AIMETOnnx):
            init_kwargs["sequence_length"] = sequence_length
            init_kwargs["context_length"] = context_length

        instance = cls(**init_kwargs)

        # Re-attach the embedding weights captured from fp_model, but ONLY if
        # the instance did not already load them from the checkpoint.
        if (
            _fp_embedding_weights is not None
            and getattr(instance, "_embedding_weights", None) is None
        ):
            instance._embedding_weights = _fp_embedding_weights
        if _fp_original_llm_config is not None:
            instance._original_llm_config = _fp_original_llm_config
        if tmp_dir is not None:
            instance._tmp_dir = tmp_dir

        return instance

    @classmethod
    def attention_mask_min_clip_and_multiplier(
        cls,
        precision: Precision,
    ) -> tuple[float | None, float]:
        if precision in {Precision.w4, Precision.float}:
            # Align with Genie (Genie/src/qualla/src/src/engines/qnn-htp/nsp-model.cpp)
            attention_mask_min_clip = -1000.0
        else:
            attention_mask_min_clip = -50.0
        return (attention_mask_min_clip, 1.0)

    def _adapt_aimet_encodings(
        self, src_encodings_path: str, dst_encodings_path: str, onnx_model_path: str
    ) -> None:
        pass

    def _use_zip_file(self) -> bool:
        return False

    @staticmethod
    def _build_quantsim(
        onnx_model: onnx.ModelProto,
        providers: list[str | tuple[str, dict]],
    ) -> QuantizationSimModel:
        """Create a QuantizationSimModel with standard LLM parameters.

        Sets module-level AIMET globals and constructs the QuantSim with
        int4 weights / int16 activations. No precision-specific configuration
        is applied -- call ``_apply_precision_activations`` or
        ``_configure_quant_sim`` afterwards.
        """
        if not AIMET_ONNX_INSTALLED:
            raise ImportError(
                "Quantized models require the AIMET-ONNX package, which is only supported on Linux. "
                "Install qai-hub-models on a Linux machine to use quantized models."
            )

        AimetLogger.set_level_for_all_areas(logging.WARNING)
        # Tie Quantizers for Concat Op
        quantsim.op_types_to_tie_qtzrs = ["Concat"]
        quantsim._tie_qtzrs = True
        # Ignore Slice and Constant outputs
        quantsim.op_outputs_to_ignore.append("Slice")
        quantsim.op_outputs_to_ignore.append("Constant")
        qs.encoding_version = "1.0.0"

        # aimet-onnx 2.33.0+ rejects an initializer shared across consumers of
        # different op types (e.g. Qwen3's tied lm_head.weight feeding both the
        # embedding Gather and the lm_head MatMul). Give each consumer a unique
        # copy. No-op on older aimet (function absent) and on models without
        # such conflicts.
        try:
            from aimet_onnx.utils import duplicate_shared_initializers

            duplicate_shared_initializers(onnx_model.graph)
        except ImportError:
            pass

        quant_sim = QuantizationSimModel(
            model=onnx_model,
            param_type="int4",
            activation_type="int16",
            quant_scheme=QuantScheme.min_max,
            config_file="htp_v73",
            providers=providers,
        )
        print(f"QuantSim session providers: {quant_sim.session.get_providers()}")
        return quant_sim

    @staticmethod
    def _apply_precision_activations(
        quant_sim: QuantizationSimModel, precision: Precision
    ) -> None:
        """Apply precision-dependent activation quantizer settings.

        Safe for both full models and split models (no structural
        heuristics). For w4: sets all activation quantizers to float16.
        """
        if precision == Precision.w4:
            for op_name, qc_op in quant_sim.qc_quantize_op_dict.items():
                if op_name in quant_sim.activation_names:
                    qc_op.reset_encoding_stats()
                    qc_op.data_type = QuantizationDataType.float
                    qc_op.bitwidth = 16

    @classmethod
    def create_quantsim(
        cls,
        onnx_model: onnx.ModelProto,
        host_device: torch.device,
        precision: Precision,
    ) -> QuantizationSimModel:
        """
        onnx_model: ONNX Model to create QuantSim model.
        host_device: Device that the QuantSim model must be placed on.
        """
        quant_sim = cls._build_quantsim(onnx_model, cls.get_ort_providers(host_device))
        return cls._configure_quant_sim(quant_sim, precision)

    @classmethod
    def _configure_quant_sim(
        cls,
        quant_sim: QuantizationSimModel,
        precision: Precision,
    ) -> QuantizationSimModel:
        if precision == Precision.w4a16:
            kv_io_map = _get_kv_io_map(quant_sim)
            quant_sim = _apply_int8_kv_cache_tying_and_lm_head(
                quant_sim, kv_io_map, use_16x8_matmuls=True
            )
        elif precision == Precision.w4:
            _set_lm_head_to_8b(quant_sim)
            cls._apply_precision_activations(quant_sim, precision)

        cls._hold_params_at_int8(quant_sim)
        cls._hold_params_at_fp16(quant_sim, precision)

        return quant_sim

    @classmethod
    def _hold_params_at_int8(cls, quant_sim: QuantizationSimModel) -> None:
        """Widen ``int8_param_names`` quantizers to 8-bit symmetric per-channel.

        Done before calibration so the range is computed for the int8 grid
        rather than reusing an int4-calibrated one.
        """
        for name in cls.int8_param_names:
            quantizer = quant_sim.qc_quantize_op_dict.get(name)
            if quantizer is None:
                raise KeyError(
                    f"No quantizer for {name}; {cls.__name__}.int8_param_names "
                    "is out of date with the exported graph."
                )
            quantizer.set_bitwidth(8)
            quantizer.use_symmetric_encodings = True
            quantizer.enable_per_channel_quantization()
            logger.info("Holding %s at int8 (per-channel, symmetric).", name)

    @classmethod
    def _hold_params_at_fp16(
        cls,
        quant_sim: QuantizationSimModel,
        precision: Precision,
    ) -> None:
        """Set ``fp16_param_names`` quantizers to fp16."""
        if precision != Precision.w4a16:
            return

        for name in cls.fp16_param_names:
            quantizer = quant_sim.qc_quantize_op_dict.get(name)
            if quantizer is None:
                raise KeyError(
                    f"No quantizer for {name}; {cls.__name__}.fp16_param_names "
                    "is out of date with the exported graph."
                )
            quantizer.quant_info.usePerChannelMode = False
            quantizer.reset_encoding_stats()
            quantizer.data_type = QuantizationDataType.float
            quantizer.bitwidth = 16
            logger.info("Holding %s at fp16.", name)

    def save_calibrated_checkpoint(  # type: ignore[override]
        self,
        output_checkpoint: str | os.PathLike | Path,
        fp_model: LLMBase,
    ) -> None:
        """
        output_checkpoint: Path to the directory which must store the checkpoint.

        Produces a single model_dynamic.onnx with symbolic dimensions.
        """
        os.makedirs(output_checkpoint, exist_ok=True)
        print(f"Creating a checkpoint of quantized model at {output_checkpoint}.")

        # Remove existing .data file if it exists since QuantSim export will create a new one
        if os.path.exists(os.path.join(output_checkpoint, "model.data")):
            os.remove(os.path.join(output_checkpoint, "model.data"))

        # Export the QuantSim model (produces model.onnx + model.data + model.encodings).
        # The input ONNX has dynamic shapes, which QuantSim preserves.
        assert self.quant_sim is not None
        self.quant_sim.export(str(output_checkpoint), "model")
        del self.quant_sim

        onnx_path = os.path.join(output_checkpoint, "model.onnx")
        shutil.move(onnx_path, os.path.join(output_checkpoint, "model_dynamic.onnx"))

        self.llm_config.save_pretrained(output_checkpoint)
        # save_jinja_files=False embeds chat_template inside tokenizer_config.json
        # so downstream Genie/geniex bundle staging picks it up.
        self.tokenizer.save_pretrained(output_checkpoint, save_jinja_files=False)

    @classmethod
    def create_onnx_models(
        cls,
        checkpoint: str | os.PathLike | Path,
        fp_model: LLMBase,
        context_length: int,
        host_device: torch.device = torch.device("cpu"),
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> None:
        """
        Export ONNX models for the checkpoint.

        Creates a single model_dynamic.onnx with symbolic dimensions.
        """
        external_weights_file = os.path.join(checkpoint, "model.data")
        onnx_file = os.path.join(checkpoint, "model.onnx")

        dynamic_onnx_model = os.path.join(checkpoint, "model_dynamic.onnx")
        if not os.path.exists(dynamic_onnx_model) or not os.path.exists(
            external_weights_file
        ):
            get_onnx_model(
                fp_model=fp_model,
                context_length=context_length,
                sequence_length=DEFAULT_SEQUENCE_LENGTH,
                path=onnx_file,
                llm_io_type=llm_io_type,
            )
            shutil.move(onnx_file, dynamic_onnx_model)

    @classmethod
    def save_tokenizer_and_config(
        cls, checkpoint: str | os.PathLike | Path, fp_model: LLMBase
    ) -> None:
        # Make sure tokenizer/config exist in the checkpoint.
        # save_jinja_files=False embeds chat_template inside tokenizer_config.json
        # so downstream Genie/geniex bundle staging picks it up.
        if not os.path.isfile(os.path.join(checkpoint, "tokenizer.json")):
            fp_model.tokenizer.save_pretrained(checkpoint, save_jinja_files=False)
        if not os.path.isfile(os.path.join(checkpoint, "config.json")):
            fp_model.llm_config.save_pretrained(checkpoint)

    def convert_to_onnx_and_aimet_encodings(  # type: ignore[override]
        self,
        output_dir: str | os.PathLike | Path,
        input_spec: InputSpec | None = None,
        model_name: str | None = None,
        external_weights: bool = False,
        bundle_external_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str:
        if model_name is None:
            model_name = self.__class__.__name__

        base_path = os.path.join(output_dir, f"{model_name}.aimet")
        os.makedirs(base_path, exist_ok=True)
        assert self.checkpoint is not None

        src_onnx_filepath = os.path.join(self.checkpoint, "model_dynamic.onnx")
        if not os.path.exists(src_onnx_filepath):
            # Fall back to fixed-shape model
            src_onnx_filepath = os.path.join(
                self.checkpoint,
                f"model_seqlen{self.sequence_length}_cl{self.context_length}.onnx",
            )
        # Find the external data file (always "model.data" in our pipelines)
        src_external_weights_filepath = None
        for name in ["model.data", "model.onnx.data"]:
            candidate = os.path.join(self.checkpoint, name)
            if os.path.exists(candidate):
                src_external_weights_filepath = candidate
                break
        assert src_external_weights_filepath is not None, (
            f"External data file not found in {self.checkpoint}"
        )

        src_encodings_filepath = os.path.join(self.checkpoint, "model.encodings")

        dst_onnx_filepath = os.path.join(base_path, "model.onnx")
        dst_external_weights_filepath = os.path.join(
            base_path, os.path.basename(src_external_weights_filepath)
        )
        dst_encodings_filepath = os.path.join(base_path, "model.encodings")

        shutil.copy(src_onnx_filepath, dst_onnx_filepath)
        shutil.copy(src_external_weights_filepath, dst_external_weights_filepath)

        self._adapt_aimet_encodings(
            src_encodings_filepath, dst_encodings_filepath, dst_onnx_filepath
        )

        return base_path

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime not in [TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT]:
            raise RuntimeError(
                f"Unsupported target_runtime provided: {target_runtime}."
                " Only Genie runtime is supported."
            )

        if precision not in {Precision.w4a16, Precision.w4}:
            raise RuntimeError("Only w4a16 and w4 precisions are supported")

        other_compile_options += " --quantize_full_type w8a16 --quantize_io -O0"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        return super().get_hub_link_options(
            target_runtime,
            other_link_options,
        )

    def get_qnn_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """
        Get the name of the QAIRT Context Graph applicable for the given sub-component.

        Sequence length (ar...) and context length (cl...) in graph name
        are semantically important to Genie
        """
        if self.sequence_length == 1:
            instantiation_type = LLMInstantiationType.TOKEN_GENERATOR
        else:
            instantiation_type = LLMInstantiationType.PROMPT_PROCESSOR
        return f"{instantiation_type.value}_ar{self.sequence_length}_cl{self.context_length}_{split_index + 1}_of_{num_splits}"

    def get_hub_inference_options(
        self,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        inference_options = super().get_hub_inference_options(
            target_runtime,
            other_inference_options,
            context_graph_name=context_graph_name,
        )
        inference_options += " --max_profiler_iterations 50"
        return inference_options

    def get_evaluator(
        self,
        task: str = "wikitext",
        device: torch.device = torch.device("cpu"),
        **task_kwargs: Any,
    ) -> BaseEvaluator:
        return _get_evaluator(
            task,
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            device,
            **task_kwargs,
        )

    get_eval_dataset_classes = LLMBase.get_eval_dataset_classes

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
        """Prepare assets to run the model end to end on-device using Genie SDK."""
        context_length = max(context_lengths)
        # Copy necessary config files
        for name in ["tokenizer.json", "tokenizer_config.json", "config.json"]:
            if (Path(checkpoint) / name).exists():
                shutil.copy(
                    Path(checkpoint) / name,
                    output_path / name,
                )
        # Save the HTP config
        device_info: dict[str, str] = {}
        attrs = hub_device.attributes
        for attr in [attrs] if isinstance(attrs, str) else attrs:
            if ":" in attr:
                key, value = attr.split(":", 1)
                device_info[key] = value
        save_htp_config_for_genie_bundle(device_info, output_path)
        # Save the genie config
        config = create_genie_config(context_length, llm_config, "rope", model_list)
        with open(output_path / "genie_config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save metadata needed for on-device run (via LLM_QNN)
        metadata = cls._create_model_metadata(
            model_id,
            model_name,
            precision,
            runtime,
            encodings_path,
            input_specs,
            output_specs,
        )

        # --- Genie supplementary files ---
        # text-generator.json: same as genie_config but with "text-generator" key
        text_gen_config = create_genie_config(
            context_length,
            llm_config,
            "rope",
            model_list,
            top_level_key="text-generator",
        )
        with open(output_path / "text-generator.json", "w") as f:
            json.dump(text_gen_config, f, indent=4)

        # sample_prompt.txt
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        sample_prompt = cls.get_input_prompt_with_tags(tokenizer=tokenizer)
        with open(output_path / "sample_prompt.txt", "w") as f:
            f.write(sample_prompt)

        # Populate supplementary_files
        metadata.supplementary_files = {
            "tokenizer.json": "Model tokenizer.json from checkpoint.",
            "tokenizer_config.json": "Model tokenizer_config.json from checkpoint.",
            "config.json": "Model config.json from checkpoint.",
            "htp_backend_ext_config.json": "HTP backend extension config for Genie.",
            "text-generator.json": "Genie SDK config for text generator.",
            "genie_config.json": "Config for use in genie-t2t-run.",
            "genie-app-script.txt": "Genie-app pipeline script for LLM inference.",
        }

        # Populate genie metadata
        chat_spec = cls.get_chat_template()
        pipeline_nodes = {"textGenerator": "text-generator.json"}
        sample_inputs = [
            GenieSampleInput(
                node="textGenerator",
                node_io="GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT",
                file="sample_prompt.txt",
            ),
        ]
        metadata.genie = GenieMetadata(
            chat_template=GenieChatTemplate(**cast(dict[str, Any], chat_spec))
            if chat_spec
            else GenieChatTemplate(),
            context_lengths=context_lengths,
            supports_streaming=True,
            supports_vision=False,
            supports_thinking=cls.supports_thinking,
            pipeline=GeniePipeline(
                nodes=pipeline_nodes,
                connections=[],
            ),
            sample_inputs=sample_inputs,
        )

        # genie-app-script.txt
        from qai_hub_models.models.templates.llm.llm_helpers import (
            generate_genie_app_script,
        )

        genie_script = generate_genie_app_script(pipeline_nodes, [], sample_inputs)
        with open(output_path / "genie-app-script.txt", "w") as f:
            f.write(genie_script)

        metadata.to_json(output_path / "metadata.json")

    @staticmethod
    def _create_model_metadata(
        model_id: str,
        model_name: str,
        precision: Precision,
        target_runtime: TargetRuntime,
        input_encodings_path: str | os.PathLike | Path,
        input_specs: dict[str, Any],
        output_specs: dict[str, Any],
    ) -> ModelMetadata:
        def make_io_metadata(
            specs: dict[str, Any], encodings: dict[str, Any]
        ) -> dict[str, TensorSpec]:
            if "encodings" in encodings:
                # 2.0.0: y_scale / y_zero_point, with the zero point already negated.
                all_encodings = {
                    v["name"]: {
                        "scale": np.atleast_1d(v["y_scale"]).tolist(),
                        "offset": (-np.atleast_1d(v.get("y_zero_point", 0))).tolist(),
                    }
                    for v in encodings["encodings"]
                    if "y_scale" in v
                }
            elif Version(encodings["version"]) >= Version("1.0.0"):
                all_encodings = {
                    v["name"]: v for v in encodings["activation_encodings"]
                }
            else:
                all_encodings = encodings["activation_encodings"]

            entries: dict[str, TensorSpec] = {}
            for name, (shape, dtype_str) in specs.items():
                entry = TensorSpec(shape=tuple(shape), dtype=dtype_str)

                fixed_name = name
                if name == "logits":
                    # The logits are missing. We need to start correcting this when
                    # making the encodings. For legacy encodings, we try to look it up.
                    for key in all_encodings:
                        if "lm_head" in key and "Conv_output_0" in key:
                            fixed_name = key
                            break

                if node_encodings := all_encodings.get(fixed_name):
                    if isinstance(node_encodings, list):
                        scale = node_encodings[0].get("scale")
                        offset = node_encodings[0].get("offset")

                        if scale is not None and offset is not None:
                            entry.quantization_parameters = QuantizationParameters(
                                scale=scale, zero_point=-offset
                            )
                    else:
                        scale = node_encodings.get("scale")
                        offset = node_encodings.get("offset")

                        if scale is not None and offset is not None:
                            entry.quantization_parameters = QuantizationParameters(
                                scale=scale[0], zero_point=-offset[0]
                            )

                entries[name] = entry
            return entries

        with open(input_encodings_path) as f:
            encodings = json.load(f)

        input_metadata = {
            k: make_io_metadata(v, encodings) for k, v in input_specs.items()
        }
        output_metadata = {
            k: make_io_metadata(v, encodings) for k, v in output_specs.items()
        }

        model_files: dict[str, ModelFileMetadata] = {}
        for k, v in input_metadata.items():
            model_files[k] = ModelFileMetadata(inputs=v, outputs=output_metadata[k])

        return ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            precision=precision,
            runtime=target_runtime,
            tool_versions=ToolVersions(),
            model_files=model_files,
        )

    @functools.cache
    def _get_embedding_table(self) -> torch.nn.Embedding:
        if self.quant_sim is None:
            raise RuntimeError(
                "Cannot get embedding table from LLM object created with _skip_quantsim_creation=True"
            )
        assert isinstance(self.quant_sim, QuantizationSimModel)

        lm_head_weights = list(_get_lm_head_weights(self.quant_sim.model.model))
        if len(lm_head_weights) != 1:
            raise RuntimeError("Unable to isolate LM Head weights from ONNX Model.")

        embedding_table = torch.from_numpy(
            onnx.numpy_helper.to_array(lm_head_weights[0]).copy()
        )
        return torch.nn.Embedding(
            self.llm_config.vocab_size,
            self.llm_config.hidden_size,
            self.llm_config.pad_token_id,
            _weight=embedding_table.T,
        )

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_table()(input_ids)


class LLM_QNN(LLMConfigEditor, BaseModel, ABC):
    # Embedding subclass
    FPModel: type[LLMBase] | None = None
    EmbeddingClass: type[Embedding] | None = None
    num_layers_per_split: int
    llm_io_type: LLMIOType = LLMIOType.genie_input_ids

    @classmethod
    def get_input_prompt_with_tags(cls, **kwargs: Any) -> str:
        """Delegate to FPModel's get_input_prompt_with_tags."""
        assert cls.FPModel is not None
        return cls.FPModel.get_input_prompt_with_tags(**kwargs)

    def __init__(
        self,
        part_models: dict[tuple[str, int], OnnxModelTorchWrapper],
        checkpoint: str | os.PathLike | Path | None,
        metadata: ModelMetadata,
        sequence_length: int,
        context_length: int,
        precision: Precision,
        tokenizer: PreTrainedTokenizerBase | None = None,
        llm_config: PretrainedConfig | None = None,
        host_device: torch.device | None = None,
        _temporary_paths: list[Path] | None = None,
    ) -> None:
        BaseModel.__init__(self)
        self.part_models = part_models
        self.context_length = context_length
        self.sequence_length = sequence_length
        self.host_device = host_device
        self.metadata = metadata
        self.precision = precision
        self._temporary_paths = _temporary_paths

        assert (
            tokenizer is not None and llm_config is not None
        ) or checkpoint is not None, (
            f"{self.__class__.__name__} is unable to instantiate tokenizer/config. Must pass either checkpoint or tokenizer/config explicitly."
        )

        self.tokenizer = tokenizer or get_tokenizer(checkpoint)
        llm_config = llm_config or get_llm_config(checkpoint)
        self.llm_config = self.edit_llm_config(llm_config)
        assert self.EmbeddingClass is not None
        self.embedding = self.EmbeddingClass(
            max_length=context_length, config=llm_config
        )
        self.checkpoint = checkpoint

    def release(self) -> None:
        self.part_models.clear()

    def get_input_spec(
        self,
        llm_config: dict,
        sequence_length: int,
        context_length: int,
        llm_io_type: LLMIOType = LLMIOType.genie_input_ids,
    ) -> InputSpec:
        """
        Parameters
        ----------
        llm_config
            Model configuration dictionary.
        sequence_length
            Sequence length for the model.
        context_length
            Context length for the model.
        llm_io_type
            Input/output type for the LLM.

        Returns
        -------
        InputSpec
            Input specification for the model.
        """
        raise NotImplementedError

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if not input_spec:
            input_spec = self.get_input_spec(
                sequence_length=self.sequence_length,
                context_length=self.context_length,
                llm_config=self.llm_config.to_dict(),
            )
        assert self.FPModel is not None
        return sample_input(
            input_spec,
            self.get_input_prompt_with_tags(tokenizer=self.tokenizer),
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            self.llm_config,
            self.embedding,
        )

    def sample_inputs(self, input_spec: InputSpec | None = None) -> SampleInputsType:  # type: ignore[override]
        # This must be defined by the HubModelProtocol protocol via BaseModel
        return self._sample_inputs_impl(input_spec)

    @classmethod
    def from_pretrained(
        cls,
        host_device: torch.device,
        sequence_length: int,
        context_length: int,
        fp_model: LLMBase | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Self:
        """
        Load weight from local checkpoint of Huggingface and create Aimet-ONNX QuantSim.
        Optionally load onnx model and AIMET encodings from a checkpoint.

        Parameters
        ----------
        host_device
            Device to use: GPU/CPU.
        sequence_length
            Sequence Length for the model.
        context_length
            Context Length for the model.
        fp_model
            Floating point version of this model.
            This is quantized as part of this class and QuantSim model is created.
        checkpoint
            Path to previously calibrated AIMET encodings and
            ONNX models. Note that encodings are sensitive to AIMET ONNX versions.
        _skip_quantsim_creation
            If True, skip creating the QuantSim model.

        Returns
        -------
        LLM_QNN : Self
            Instance of the model loaded from the checkpoint.
        """
        _verify_onnxruntime_qnn_installed()

        if host_device is None:
            host_device = torch.device("cpu")

        inst = "prompt" if sequence_length > 1 else "token"

        assert checkpoint is not None
        checkpoint_path = Path(checkpoint)

        # Construct ONNX model wrapping the context binaries
        context_bin_paths = sorted(checkpoint_path.glob("*.bin"))

        metadata = ModelMetadata.from_json(checkpoint_path / "metadata.json")

        tool_versions = ToolVersions.from_yaml(checkpoint_path / "tool-versions.yaml")
        assert tool_versions.qairt is not None
        qairt_version = tool_versions.qairt.full_version

        part_models: dict[tuple[str, int], OnnxModelTorchWrapper] = {}
        temporary_paths: list[Path] = []

        num_splits = len(context_bin_paths)
        for part_i, context_bin_path in enumerate(context_bin_paths):
            graph_name = cls.construct_qnn_context_graph_name(
                part_i, len(context_bin_paths), sequence_length, context_length
            )
            onnx_path = checkpoint_path / (
                os.path.splitext(os.path.basename(context_bin_path))[0]
                + f"_{inst}.onnx"
            )

            io_metadata = metadata.model_files[f"{inst}_{part_i + 1}_of_{num_splits}"]

            def _io_metadata_to_specs(
                entries: dict[str, TensorSpec],
            ) -> dict[str, ModelIODetails]:
                onnx_specs: dict[str, Any] = {}
                for k, v in entries.items():
                    qdq_params = None
                    if v.quantization_parameters is not None:
                        qdq_params = ModelIODetails.QDQParams(
                            scale=v.quantization_parameters.scale,
                            zero_point=v.quantization_parameters.zero_point,
                        )

                    onnx_specs[k] = ModelIODetails(
                        shape=v.shape,
                        dtype=np.dtype(v.dtype),
                        qdq_params=qdq_params,
                    )
                return onnx_specs

            onnx_input_specs = _io_metadata_to_specs(io_metadata.inputs)
            onnx_output_specs = _io_metadata_to_specs(io_metadata.outputs)

            rel_context_bin_path = os.path.relpath(context_bin_path, checkpoint)
            generate_wrapper_onnx_file(
                graph_name,
                onnx_path,
                onnx_input_specs,
                onnx_output_specs,
                rel_context_bin_path,
                qairt_version,
            )
            # Quantization parameters for intermediates are not available and
            # will raise warnings. Leaving them quantized is fine.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                part_models[(inst, part_i)] = OnnxModelTorchWrapper.OnNPU(onnx_path)
            temporary_paths.append(onnx_path)

        return cls(
            part_models=part_models,
            checkpoint=checkpoint,
            sequence_length=sequence_length,
            metadata=metadata,
            context_length=context_length,
            host_device=host_device,
            precision=metadata.precision,
            _temporary_paths=temporary_paths,
        )

    def __del__(self) -> None:
        if hasattr(self, "_temporary_paths") and self._temporary_paths:
            for path in self._temporary_paths:
                os.remove(path)

    def forward(self, *args: torch.Tensor) -> list[torch.Tensor]:
        input_ids = args[0]
        sequence_length = input_ids.shape[1]
        num_splits = len(self.part_models)
        inst = "prompt" if sequence_length > 1 else "token"

        part1_model = self.part_models[(inst, 0)]
        part1_out = part1_model(input_ids)

        last_intermediate = part1_out

        # Shared between between parts
        attn_mask = args[1]
        position_ids_cos = args[2]
        position_ids_sin = args[3]

        kv_output: list[torch.Tensor] = []

        for part_i in range(1, num_splits):
            torch_part_i_inputs = [
                last_intermediate,
                attn_mask,
                position_ids_cos,
                position_ids_sin,
            ]

            kv_indices = range(
                (part_i - 1) * self.num_layers_per_split,
                min(
                    part_i * self.num_layers_per_split,
                    self.llm_config.num_hidden_layers,
                ),
            )
            for idx in kv_indices:
                torch_part_i_inputs.append(args[4 + idx * 2])
                torch_part_i_inputs.append(args[4 + idx * 2 + 1])

            part_i_model = self.part_models[(inst, part_i)]
            part_i_out = part_i_model(*torch_part_i_inputs)

            last_intermediate = part_i_out[0]
            kv_output += part_i_out[1:]

        logits = last_intermediate
        return [cast(torch.Tensor, logits), *kv_output]

    @classmethod
    def construct_qnn_context_graph_name(
        cls,
        split_index: int,
        num_splits: int,
        sequence_length: int,
        context_length: int,
    ) -> str:
        """Similar to get_qnn_context_graph_name, but can be run without instance."""
        if sequence_length == 1:
            instantiation_type = LLMInstantiationType.TOKEN_GENERATOR
        else:
            instantiation_type = LLMInstantiationType.PROMPT_PROCESSOR
        return f"{instantiation_type.value}_ar{sequence_length}_cl{context_length}_{split_index + 1}_of_{num_splits}"

    def get_qnn_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """
        Get the name of the QNN Context Graph applicable for the given sub-component.

        Sequence length (ar...) and context length (cl...) in graph name
        are semantically important to Genie.
        """
        return self.construct_qnn_context_graph_name(
            split_index, num_splits, self.sequence_length, self.context_length
        )

    def get_evaluator(
        self,
        task: str = "wikitext",
        device: torch.device = torch.device("cpu"),
        **task_kwargs: Any,
    ) -> BaseEvaluator:
        return _get_evaluator(
            task,
            self.context_length,
            self.sequence_length,
            self.tokenizer,
            device,
            **task_kwargs,
        )

    get_eval_dataset_classes = LLMBase.get_eval_dataset_classes

    @property
    def main_input_name(self) -> str:
        if self.llm_io_type == LLMIOType.genie_input_embeds:
            return "input_embeds"
        return "input_ids"

    @functools.cache
    def _get_embedding_table(self) -> torch.nn.Embedding:
        if self.quant_sim is None:
            raise RuntimeError(
                "Cannot get embedding table from LLM object created with _skip_quantsim_creation=True"
            )
        assert isinstance(self.quant_sim, QuantizationSimModel)

        lm_head_weights = list(_get_lm_head_weights(self.quant_sim.model.model))
        if len(lm_head_weights) != 1:
            raise RuntimeError("Unable to isolate LM Head weights from ONNX Model.")

        embedding_table = torch.from_numpy(
            onnx.numpy_helper.to_array(lm_head_weights[0]).copy()
        )
        return torch.nn.Embedding(
            self.llm_config.vocab_size,
            self.llm_config.hidden_size,
            self.llm_config.pad_token_id,
            _weight=embedding_table.T,
        )

    def convert_input_ids_to_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._get_embedding_table()(input_ids)


class LLMDynamicBase(LLMBase):
    """LLMBase variant for dynamic-shape models.

    Dynamic models accept any sequence/context length at runtime, so these
    are not part of the construction or from_pretrained API. Shapes flow
    through get_input_spec, which controls what static shapes get compiled.

    Internally, the parent LLMBase.__init__ requires sequence_length and
    context_length (used for EmbeddingClass and stored as attributes).
    This subclass supplies hardcoded defaults so callers never see them.
    """

    # Asset store identification (set by concrete model classes)
    model_id: str
    model_asset_version: int
    default_checkpoint: dict[Precision, str]
    default_precision: Precision

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path,
        is_token_generator: bool = False,
        load_pretrained: bool = True,
        host_device: torch.device | None = None,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
        _skip_optimizations: list[str] | None = None,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            context_length=DEFAULT_CONTEXT_LENGTH,
            is_token_generator=is_token_generator,
            load_pretrained=load_pretrained,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
            _skip_optimizations=_skip_optimizations,
        )


class LLMDynamic_AIMETOnnx(LLM_AIMETOnnx):
    """LLM_AIMETOnnx variant for dynamic-shape models.

    See LLMDynamicBase for rationale.
    """

    def __init__(
        self,
        quant_sim: QuantizationSimModel | None,
        checkpoint: str | os.PathLike | Path | None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        llm_config: PretrainedConfig | None = None,
        host_device: torch.device | None = None,
        attention_mask_min_clip: float | None = None,
        attention_mask_multiplier: float = 1.0,
    ) -> None:
        super().__init__(
            quant_sim=quant_sim,
            checkpoint=checkpoint,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            context_length=DEFAULT_CONTEXT_LENGTH,
            tokenizer=tokenizer,
            llm_config=llm_config,
            host_device=host_device,
            attention_mask_min_clip=attention_mask_min_clip,
            attention_mask_multiplier=attention_mask_multiplier,
        )

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        host_device: torch.device | None = None,
        precision: Precision = Precision.w4,
        fp_model: LLMBase | None = None,
        checkpoint: str | os.PathLike | Path | None = None,
        _skip_quantsim_creation: bool = False,
    ) -> Self:
        if host_device is None:
            host_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return super().from_pretrained(
            host_device=host_device,
            sequence_length=DEFAULT_SEQUENCE_LENGTH,
            context_length=DEFAULT_CONTEXT_LENGTH,
            precision=precision,
            fp_model=fp_model,
            checkpoint=checkpoint,
            _skip_quantsim_creation=_skip_quantsim_creation,
        )

    def _prefill_dataset(
        self,
        generator: Generator,
        dataloader: torch.utils.data.DataLoader,
        num_inputs: int,
        sample_to_kwargs: Callable[[Any, torch.device], dict[str, Any]],
        desc: str = "Pre-filling data",
    ) -> list[list[torch.Tensor | np.ndarray]]:
        """Shared prefill loop for calibration and weight optimization.

        Writes prefilled tensors to memory-mapped files to avoid RAM blow-up
        when accumulating large numbers of KV-cache entries. Uses one file
        per input with uniform entry shapes.
        """
        mmap_dir = tempfile.mkdtemp(prefix="prefill_")
        # Register cleanup on normal process exit
        atexit.register(shutil.rmtree, mmap_dir, ignore_errors=True)

        num_entries = 0
        shapes: list[tuple] = []
        dtypes: list[np.dtype] = []
        files: list[Any] = []
        initialized = False

        device = generator.device
        with self.remove_quantization(), torch.no_grad():
            for sample in tqdm(dataloader, total=len(dataloader), desc=desc):
                # Generic: let caller transform sample to kwargs
                prefill_kwargs = sample_to_kwargs(sample, device)

                for prefilled_inputs in generator.prefill(**prefill_kwargs):
                    arrays = [
                        tensor.cpu().numpy()
                        if isinstance(tensor, torch.Tensor)
                        else np.asarray(tensor)
                        for tensor in prefilled_inputs.values()
                    ]
                    if not initialized:
                        for i, arr in enumerate(arrays):
                            shapes.append(arr.shape)
                            dtypes.append(arr.dtype)
                            fpath = os.path.join(mmap_dir, f"input_{i}.bin")
                            # One long-lived handle per input; closed in bulk
                            # after the loop (not a with-block).
                            files.append(open(fpath, "wb"))  # noqa: SIM115
                            files[i].write(arr.tobytes())
                        initialized = True
                    else:
                        for i, arr in enumerate(arrays):
                            if arr.shape != shapes[i]:
                                raise ValueError(
                                    f"Calibration sample {num_entries} input #{i} has "
                                    f"shape {arr.shape}, expected {shapes[i]} (locked "
                                    "from the first sample)."
                                )
                            files[i].write(arr.tobytes())
                    num_entries += 1
                del prefill_kwargs
                gc.collect()
                torch.cuda.empty_cache()

        for f in files:
            f.close()

        inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in range(num_inputs)]
        for i in range(num_inputs):
            fpath = os.path.join(mmap_dir, f"input_{i}.bin")
            mm = np.memmap(
                fpath, dtype=dtypes[i], mode="r", shape=(num_entries, *shapes[i])
            )
            for j in range(num_entries):
                inputs[i].append(mm[j])

        return inputs

    def _prefill_from_dataset_cls(
        self,
        dataset_cls: type[BaseDataset],
        num_samples: int,
        sequence_length: int,
        context_length: int,
    ) -> DatasetEntries | None:
        """Instantiate a text dataset, prefill it, and return hub dataset entries.

        Shared by get_calibration_data and get_weight_optimization_data so both
        run through the same WikiText-style prefill path. ``dataset_cls`` must
        yield text samples that collate to (input_ids, attention_mask, ...) --
        WikiText, GeneratedDataset, and their interleave all satisfy this.
        """
        from qai_hub_models.datasets import instantiate_dataset
        from qai_hub_models.models.templates.llm.generator_factory import make_generator

        if num_samples == 0:
            num_samples = math.ceil(80000 / context_length)

        dataset = instantiate_dataset(
            dataset_cls,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=num_samples,
        )
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None

        generator = make_generator(
            self,
            sequence_length=sequence_length,
            context_length=context_length,
            model_cls=self.FPModel,
        )

        def text_sample_to_kwargs(
            sample: tuple[torch.Tensor, ...], device: torch.device
        ) -> dict[str, torch.Tensor]:
            input_ids, attention_mask, *_ = sample
            return dict(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )

        inputs = self._prefill_dataset(
            generator,
            dataloader,
            num_inputs=len(input_spec),
            sample_to_kwargs=text_sample_to_kwargs,
            desc=f"Pre-filling data ({dataset_cls.dataset_name()})",
        )
        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))

    # Per-model interleave loaders, keyed by frozenset of source schema names.
    # (Not a default calibration set: this is the name -> loader registry that
    # resolve_dataset_cls uses to map an Interleaved{...} spec a recipe names to
    # the model's local dataset class.)
    extra_interleaved_datasets: dict[frozenset[str], type] = {}

    def prefill_dataset(
        self,
        dataset_spec: DatasetSpec,
        num_samples: int = 0,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] | None = None,
    ) -> DatasetEntries | None:
        """Resolve a dataset spec to a loader and prefill it (text path; VLM
        subclasses override to engage vision machinery for image datasets).
        """
        # Local import avoids the model <-> quantize import cycle.
        from qai_hub_models.models.templates.llm.quantize import resolve_dataset_cls

        dataset_cls = resolve_dataset_cls(dataset_spec, self.extra_interleaved_datasets)
        return self._prefill_from_dataset_cls(
            dataset_cls,
            num_samples=num_samples,
            sequence_length=sequence_length,
            context_length=context_length,
        )

    def _resolve_step_dataset(self, step: TechniqueSpec) -> DatasetSpec:
        """The dataset a data-consuming step runs on. The recipe is the single source
        of truth for data: every SeqMSE / AdaScale / Calibration step must name its
        own ``dataset:`` -- there is no class-level default. (Only data-consuming
        techniques reach here; SpinQuant is pre-sim, Skip/RemoveQuantization carry
        no data.)
        """
        if step.dataset is None:
            raise ValueError(
                f"Recipe step {step.name!r} consumes calibration data but names no "
                f"`dataset:`. Add one to this step in the recipe (manifest "
                f"lm_quantization_details, or the --precision recipe file)."
            )
        return step.dataset
