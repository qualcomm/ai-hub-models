# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Gemma4 E4B IT - PreSplit-Part architecture for LLM deployment.

Model: google/gemma-4-E4B-it
Architecture: Gemma4 with sliding window + global attention, PLE, GQA
"""

from __future__ import annotations

import logging

from qai_hub_models import Precision
from qai_hub_models.models.templates.gemma4.model import (
    Gemma4PartBase,
    Gemma4PreSplitBase,
    Gemma4PreSplitCollectionBase,
    Gemma4QuantizablePreSplitBase,
)
from qai_hub_models.models.templates.gemma4.vision_encoder import Gemma4VisionEncoder
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import SplitForwardMixin

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4

# Model architecture constants (from Gemma4-E4B config text_config)
NUM_LAYERS = 42
NUM_KV_SHARED_LAYERS = 18  # Layers 24-41 share KV
NUM_KV_LAYERS = NUM_LAYERS - NUM_KV_SHARED_LAYERS  # 24 layers with own KV

# 4-way split (a single context binary is too large to execute on-device):
#   Part 1: layers 0-23    (produce KV)
#   Part 2: layers 24-32   (consume shared KV)
#   Part 3: layers 33-41   (consume shared KV)
#   Part 4: lm_head        (separate)
NUM_SPLITS = 4
SPLIT_LM_HEAD = True
# Explicit transformer-block boundaries (end-layer index, exclusive). [24, 33] ->
# layers[0:24] | layers[24:33] | layers[33:42]; lm_head appended by split_lm_head.
SPLITTING_POINTS = [NUM_KV_LAYERS, 33]  # [24, 33]
NUM_LAYERS_PER_SPLIT = None  # unused when splitting_points is set

HIDDEN_SIZE = 2560
NUM_KEY_VALUE_HEADS = 2  # GQA
NUM_ATTN_HEADS = 8
HEAD_DIM = 256  # SWA layers
GLOBAL_HEAD_DIM = 512  # Full attention layers
SLIDING_WINDOW_PATTERN = 6  # 5 SWA + 1 global (layer_types: every 6th is full)
SLIDING_WINDOW = 512  # SWA KV window size

HF_REPO_NAME = "google/gemma-4-E4B-it"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 120

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
# precision -> published asset name (fetched by from_pretrained("DEFAULT_W4A16")).
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "gemma_4_e4b_it_w4a16",
}

# Name used for split ONNX file basenames
SPLIT_MODEL_NAME = "Gemma4_E4B"

# See ``LLM_AIMETOnnx.int8_param_names``. v4 ships all of these at int8; see
# https://github.com/qcom-ai-hub/tetracode/issues/21286.
INT8_PARAM_NAMES = tuple(
    f"model.model.layers.{i}.mlp.down_proj.weight" for i in range(NUM_LAYERS)
)

# Vision encoder (VEG) default trace resolution. The Gemma4 processor applies
# pan-and-scan tiling, so the actual patch count is derived from this at load
# time (see Gemma4VisionEncoder.from_pretrained).
VISION_DEFAULT_IMAGE_SIZE = 448


class Gemma4_E4B_VisionEncoder(Gemma4VisionEncoder):
    """Gemma4-E4B Visual Embedding Generator (VEG), exported as FP16."""

    _hf_repo_name = HF_REPO_NAME
    default_image_size = VISION_DEFAULT_IMAGE_SIZE


class Gemma4_E4B_PreSplit(Gemma4PreSplitBase):
    """FP PreSplit for Gemma4-E4B."""

    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    global_head_dim = GLOBAL_HEAD_DIM
    num_kv_shared_layers = NUM_KV_SHARED_LAYERS
    sliding_window_pattern = SLIDING_WINDOW_PATTERN
    sliding_window = SLIDING_WINDOW
    hf_repo_name = HF_REPO_NAME

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = SPLIT_LM_HEAD
    splitting_points = SPLITTING_POINTS

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION


class Gemma4_E4B_QuantizablePreSplit(
    Gemma4QuantizablePreSplitBase[Gemma4_E4B_PreSplit]
):
    """Quantizable PreSplit for Gemma4-E4B."""

    FPModel = Gemma4_E4B_PreSplit

    num_layers = NUM_LAYERS
    num_kv_shared_layers = NUM_KV_SHARED_LAYERS
    # Must be set on the Quantizable path too (its _get_output_spec derives the
    # KV tensor name prefixes from it and would otherwise default to pattern-5,
    # mismatching E4B's pattern-6). See templates/gemma4/model.py for details.
    sliding_window_pattern = SLIDING_WINDOW_PATTERN
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = SPLIT_LM_HEAD
    splitting_points = SPLITTING_POINTS
    # Base class holds these at int8 (per-channel, symmetric) via _hold_params_at_int8.
    int8_param_names = INT8_PARAM_NAMES


class Gemma4_E4B_PartBase(Gemma4PartBase):
    """Part base for Gemma4-E4B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    global_head_dim = GLOBAL_HEAD_DIM
    num_layers = NUM_LAYERS
    num_kv_shared_layers = NUM_KV_SHARED_LAYERS
    sliding_window_pattern = SLIDING_WINDOW_PATTERN
    sliding_window = SLIDING_WINDOW
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Gemma4_E4B_PreSplit
    quant_presplit_cls = Gemma4_E4B_QuantizablePreSplit


class Gemma4_E4B_Part1_Of_4(Gemma4_E4B_PartBase):
    """Part 1: layers 0-23 (produce KV)."""

    part_id = 1


class Gemma4_E4B_Part2_Of_4(Gemma4_E4B_PartBase):
    """Part 2: layers 24-32 (consume shared KV)."""

    part_id = 2


class Gemma4_E4B_Part3_Of_4(Gemma4_E4B_PartBase):
    """Part 3: layers 33-41 (consume shared KV)."""

    part_id = 3


class Gemma4_E4B_Part4_Of_4(Gemma4_E4B_PartBase):
    """Part 4: LM head."""

    part_id = 4


_SPLIT_PART_CLASSES: list[type] = [
    Gemma4_E4B_Part1_Of_4,
    Gemma4_E4B_Part2_Of_4,
    Gemma4_E4B_Part3_Of_4,
    Gemma4_E4B_Part4_Of_4,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Gemma4_E4B_QuantizablePreSplit
):
    """Quantized eval via split Parts."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Gemma4_E4B_PreSplit):  # type: ignore[misc]
    """FP eval via split Parts."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Gemma4_E4B_Collection(Gemma4PreSplitCollectionBase):
    """Collection with 4 Parts for Gemma4-E4B (L0-23, L24-32, L33-41, lm_head)."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Gemma4_E4B_PreSplit
    part_base_cls = Gemma4_E4B_PartBase
    vision_encoder_cls = Gemma4_E4B_VisionEncoder
    supports_thinking = True
    think_start = "<|channel>thought\n"
    think_end = "<channel|>"
    think_header = "<|think|>\n"
    parts = {
        "part1_of_4": Gemma4_E4B_Part1_Of_4,
        "part2_of_4": Gemma4_E4B_Part2_Of_4,
        "part3_of_4": Gemma4_E4B_Part3_Of_4,
        "part4_of_4": Gemma4_E4B_Part4_Of_4,
    }
