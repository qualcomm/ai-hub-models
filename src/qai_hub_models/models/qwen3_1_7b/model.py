# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3-1.7B - PreSplit-Part architecture for LLM deployment.

The generic PreSplit/Part/Collection machinery lives in
``qai_hub_models.models.templates.llm.model`` (family-agnostic) and
``qai_hub_models.models.templates.qwen3.model`` (Qwen3-coupled: RoPE embedding,
dynamo encoding adaptation, explicit head_dim, attention-mask multiply, and the
tied-embedding encoding fix). This module supplies the 1.7B-specific architecture
constants and the small concrete subclasses (Part classes + the Collection,
whose ``parts`` mapping registers the Part classes).

Quantization uses the SpinQuant (R1+R3) -> AdaScale -> Calibration recipe,
authored in the model's ``manifest.yaml`` under ``lm_quantization_details`` (or
supplied as a ``--precision`` recipe YAML on the quantize CLI).
"""

from __future__ import annotations

import logging

from qai_hub_models import Precision

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
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

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
# v2 promotes INT8_PARAM_NAMES to int8 in the published model.encodings.
MODEL_ASSET_VERSION = 2

# Model architecture constants (from Qwen3-1.7B)
NUM_LAYERS = 28
# Part 1 holds the embedding; the 28 transformer blocks are spread across the
# remaining (NUM_SPLITS - 1) parts, with the last also carrying the LM head.
# A single monolithic transformer part (all 28 layers + LM head) produces a
# context binary too large for the HTP to load on-device ("Could not create
# context from binary ... err 1002"), so split into 4 (== 3 transformer parts of
# ~10 layers each, matching the proven qwen3_4b layout).
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 10
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 16
# Qwen3 uses an explicit head_dim that differs from hidden_size // num_attn_heads.
HEAD_DIM = 128

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen3-1.7B"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 16

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "qwen3_1_7b_w4a16",
}

# Name used for split ONNX file basenames (e.g. Qwen3_1_7B_1_of_4.onnx)
SPLIT_MODEL_NAME = "Qwen3_1_7B"

# See ``LLM_AIMETOnnx.int8_param_names``. Found with scripts/llm/sim_vs_device:
# 14.9 dB SQNR at this tap vs 41-52 dB for every other layer's down_proj.
# Leaving it at int4 costs 15.2% absolute on the 100-prompt grader score.
INT8_PARAM_NAMES = ("model.model.layers.2.mlp.down_proj.weight",)


class Qwen3_1_7B_PreSplit(Qwen3PreSplitBase):
    """FP PreSplit for Qwen3-1.7B."""

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
class Qwen3_1_7B_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    Qwen3QuantizablePreSplitBase[Qwen3_1_7B_PreSplit],
):
    """Quantizable PreSplit for Qwen3-1.7B."""

    FPModel = Qwen3_1_7B_PreSplit
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
    # SpinQuant (R1+R3) is a pre-sim step in the quant recipe (lm_quantization_details).
    supports_thinking = True
    int8_param_names = INT8_PARAM_NAMES


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
class Qwen3_1_7B_PartBase(QuantizeFullTypeW8A16PartMixin, Qwen3PartBase):
    """Unified Part base for Qwen3-1.7B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Qwen3_1_7B_PreSplit
    quant_presplit_cls = Qwen3_1_7B_QuantizablePreSplit


class Qwen3_1_7B_Part1_Of_4(Qwen3_1_7B_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Qwen3_1_7B_Part2_Of_4(Qwen3_1_7B_PartBase):
    """Part 2: Transformer layers."""

    part_id = 2


class Qwen3_1_7B_Part3_Of_4(Qwen3_1_7B_PartBase):
    """Part 3: Transformer layers."""

    part_id = 3


class Qwen3_1_7B_Part4_Of_4(Qwen3_1_7B_PartBase):
    """Part 4: Final transformer layers + LM head."""

    part_id = 4


_SPLIT_PART_CLASSES: list[type] = [
    Qwen3_1_7B_Part1_Of_4,
    Qwen3_1_7B_Part2_Of_4,
    Qwen3_1_7B_Part3_Of_4,
    Qwen3_1_7B_Part4_Of_4,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Qwen3_1_7B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Qwen3_1_7B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Qwen3_1_7B_Collection(CollectionCompileO0Mixin, Qwen3PreSplitCollectionBase):
    """Unified Collection with 4 Parts for Qwen3-1.7B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Qwen3_1_7B_PreSplit
    part_base_cls = Qwen3_1_7B_PartBase
    supports_thinking = True
    think_start = "<think>\n"
    think_end = "\n</think>"
    parts = {
        "part1_of_4": Qwen3_1_7B_Part1_Of_4,
        "part2_of_4": Qwen3_1_7B_Part2_Of_4,
        "part3_of_4": Qwen3_1_7B_Part3_Of_4,
        "part4_of_4": Qwen3_1_7B_Part4_Of_4,
    }
