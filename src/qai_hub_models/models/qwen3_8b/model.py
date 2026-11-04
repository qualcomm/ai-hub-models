# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3-8B - PreSplit-Part architecture for LLM deployment.

The generic PreSplit/Part/Collection machinery lives in
``qai_hub_models.models.templates.llm.model`` (family-agnostic) and
``qai_hub_models.models.templates.qwen3.model`` (Qwen3-coupled: RoPE embedding,
dynamo encoding adaptation, explicit head_dim, attention-mask multiply, and the
tied-embedding encoding fix). This module supplies the 8B-specific architecture
constants and the small concrete subclasses (Part classes + the Collection,
whose ``parts`` mapping registers the Part classes).
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
from qai_hub_models.models.templates.llm.model import SplitForwardMixin
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
# v2 was the static (pre-dynamo) qwen3_8b model; v3 was the first dynamic-shape
# (dynamo) version; v4 untied embedding/lm_head (see #3685). v5 re-converts the
# encodings with the updated migration script (#3561) so the FFN-output residual
# adds at the split boundaries keep their activation encodings (the static export
# inserts an identity Cast the dynamo graph elides, which previously dropped them
# and made the HTP linker reject the split with "Non-identical quantization
# parameters").
MODEL_ASSET_VERSION = 5

# Model architecture constants (from Qwen3-8B)
NUM_LAYERS = 36
NUM_SPLITS = 5
# The dynamic split reserves one split for the embedding (split_embedding=True,
# split_lm_head=False), leaving 4 transformer-block splits: ceil(36 / 4) = 9.
# (The old static model used split_lm_head=True -> 3 block splits x 12 layers.)
NUM_LAYERS_PER_SPLIT = 9
HIDDEN_SIZE = 4096
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32
# Qwen3 uses an explicit head_dim. For Qwen3-8B 4096 // 32 == 128 already, but
# we set it explicitly to stay consistent with the family convention.
HEAD_DIM = 128

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen3-8B"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 150

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "qwen3_8b_w4a16",
}

# Name used for split ONNX file basenames (e.g. Qwen3_8B_1_of_5.onnx)
SPLIT_MODEL_NAME = "Qwen3_8B"


class Qwen3_8B_PreSplit(Qwen3PreSplitBase):
    """FP PreSplit for Qwen3-8B."""

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


class Qwen3_8B_QuantizablePreSplit(Qwen3QuantizablePreSplitBase[Qwen3_8B_PreSplit]):
    """Quantizable PreSplit for Qwen3-8B."""

    FPModel = Qwen3_8B_PreSplit
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

    # AdaScale config (32 attn heads + 8 KV heads + 1).
    ada_scale_num_rmsnorm_per_blk = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1
    supports_thinking = True


class Qwen3_8B_PartBase(Qwen3PartBase):
    """Unified Part base for Qwen3-8B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Qwen3_8B_PreSplit
    quant_presplit_cls = Qwen3_8B_QuantizablePreSplit


class Qwen3_8B_Part1_Of_5(Qwen3_8B_PartBase):
    """Part 1: Embedding + first layers."""

    part_id = 1


class Qwen3_8B_Part2_Of_5(Qwen3_8B_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Qwen3_8B_Part3_Of_5(Qwen3_8B_PartBase):
    """Part 3: Middle layers."""

    part_id = 3


class Qwen3_8B_Part4_Of_5(Qwen3_8B_PartBase):
    """Part 4: Middle layers."""

    part_id = 4


class Qwen3_8B_Part5_Of_5(Qwen3_8B_PartBase):
    """Part 5: Final layers + LM head."""

    part_id = 5


_SPLIT_PART_CLASSES: list[type] = [
    Qwen3_8B_Part1_Of_5,
    Qwen3_8B_Part2_Of_5,
    Qwen3_8B_Part3_Of_5,
    Qwen3_8B_Part4_Of_5,
    Qwen3_8B_Part5_Of_5,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Qwen3_8B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Qwen3_8B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Qwen3_8B_Collection(Qwen3PreSplitCollectionBase):
    """Unified Collection with 5 Parts for Qwen3-8B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Qwen3_8B_PreSplit
    part_base_cls = Qwen3_8B_PartBase
    supports_thinking = True
    think_start = "<think>\n"
    think_end = "\n</think>"
    parts = {
        "part1_of_5": Qwen3_8B_Part1_Of_5,
        "part2_of_5": Qwen3_8B_Part2_Of_5,
        "part3_of_5": Qwen3_8B_Part3_Of_5,
        "part4_of_5": Qwen3_8B_Part4_Of_5,
        "part5_of_5": Qwen3_8B_Part5_Of_5,
    }
