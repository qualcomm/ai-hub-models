# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3-0.6B - PreSplit-Part architecture for LLM deployment.

The generic PreSplit/Part/Collection machinery lives in
``qai_hub_models.models.templates.llm.model`` (family-agnostic) and
``qai_hub_models.models.templates.qwen3.model`` (Qwen3-coupled: RoPE embedding,
dynamo encoding adaptation, explicit head_dim, attention-mask multiply, and the
tied-embedding encoding fix). This module supplies the 0.6B-specific architecture
constants and the small concrete subclasses (Part classes + the Collection,
whose ``parts`` mapping registers the Part classes).

Quantization uses the SpinQuant (R2+R3) -> AdaScale -> Calibration recipe,
authored in the model's ``manifest.yaml`` under ``lm_quantization_details``: AdaScale
weight-opt runs on Wikitext/train and the final calibration on an interleave of
self-generated text + Wikitext/train. The interleave name in the recipe is mapped
to this model's local loader via ``extra_interleaved_datasets``.
"""

from __future__ import annotations

import logging

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models import Precision
from qai_hub_models.models.qwen3_0_6b.dataset import InterleavedGeneratedWikitext
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    CollectionCompileO0Mixin,
    CompileO0Mixin,
    QuantizeFullTypeW8A16Mixin,
    QuantizeFullTypeW8A16PartMixin,
    SplitForwardMixin,
)
from qai_hub_models.models.templates.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models.templates.qwen3.model import (
    Qwen3PartBase,
    Qwen3PreSplitBase,
    Qwen3PreSplitCollectionBase,
    Qwen3QuantizablePreSplitBase,
)

logger = logging.getLogger(__name__)

# The shared 5-CL family default overflows HTP memory on IQ9 (qcs9075), so pin a
# 3-CL set for every Qwen3-0.6B export.
DEFAULT_EXPORT_CONTEXT_LENGTHS = [512, 1024, 4096]
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
# v2 promotes INT8_PARAM_NAMES to int8 in the published model.encodings.
MODEL_ASSET_VERSION = 2

# Model architecture constants (from Qwen3-0.6B)
NUM_LAYERS = 28
# Part 1 = embedding, Part 2 = 28 layers + LM head. 2 splits fit the HTP at this
# size; bump NUM_SPLITS if on-device load hits err 1002 (see qwen3_1_7b).
NUM_SPLITS = 2
NUM_LAYERS_PER_SPLIT = 28
HIDDEN_SIZE = 1024
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 16
# Qwen3 uses an explicit head_dim that differs from hidden_size // num_attn_heads.
HEAD_DIM = 128

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen3-0.6B"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 8

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "qwen3_0_6b_w4a16",
}

# Name used for split ONNX file basenames (e.g. Qwen3_0_6B_1_of_2.onnx)
SPLIT_MODEL_NAME = "Qwen3_0_6B"

# See ``LLM_AIMETOnnx.int8_param_names``. Leaving this at int4 costs ~13%
# absolute on the 100-prompt grader score.
INT8_PARAM_NAMES = ("model.model.layers.2.mlp.down_proj.weight",)


class Qwen3_0_6B_PreSplit(Qwen3PreSplitBase):
    """FP PreSplit for Qwen3-0.6B."""

    GeneratorClass = HubCompatibleGenerator
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    hf_repo_name = HF_REPO_NAME

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Qwen3_0_6B_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    Qwen3QuantizablePreSplitBase[Qwen3_0_6B_PreSplit],
):
    """Quantizable PreSplit for Qwen3-0.6B."""

    FPModel = Qwen3_0_6B_PreSplit
    GeneratorClass = HubCompatibleGenerator

    num_layers = NUM_LAYERS
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    # AdaScale config (16 attn heads + 8 KV heads + 1).
    ada_scale_num_rmsnorm_per_blk = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1
    supports_thinking = True
    # Base class holds these at int8 (per-channel, symmetric) via _hold_params_at_int8.
    int8_param_names = INT8_PARAM_NAMES

    # The recipe's Calibration step names an Interleaved{GeneratedDataset, Wikitext}.
    # GeneratedDataset is model-specific, so its interleave loader is registered here
    # (rather than in the shared registry) for resolve_dataset_cls to find.
    extra_interleaved_datasets = {
        frozenset({"GeneratedDataset", "Wikitext"}): InterleavedGeneratedWikitext,
    }


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
class Qwen3_0_6B_PartBase(QuantizeFullTypeW8A16PartMixin, Qwen3PartBase):
    """Unified Part base for Qwen3-0.6B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Qwen3_0_6B_PreSplit
    quant_presplit_cls = Qwen3_0_6B_QuantizablePreSplit


class Qwen3_0_6B_Part1_Of_2(Qwen3_0_6B_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Qwen3_0_6B_Part2_Of_2(Qwen3_0_6B_PartBase):
    """Part 2: Transformer layers + LM head."""

    part_id = 2


_SPLIT_PART_CLASSES: list[type] = [
    Qwen3_0_6B_Part1_Of_2,
    Qwen3_0_6B_Part2_Of_2,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Qwen3_0_6B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Qwen3_0_6B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Qwen3_0_6B_Collection(CollectionCompileO0Mixin, Qwen3PreSplitCollectionBase):
    """Unified Collection with 2 Parts for Qwen3-0.6B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Qwen3_0_6B_PreSplit
    part_base_cls = Qwen3_0_6B_PartBase
    supports_thinking = True
    think_start = "<think>\n"
    think_end = "\n</think>"
    default_context_lengths = DEFAULT_EXPORT_CONTEXT_LENGTHS
    parts = {
        "part1_of_2": Qwen3_0_6B_Part1_Of_2,
        "part2_of_2": Qwen3_0_6B_Part2_Of_2,
    }
