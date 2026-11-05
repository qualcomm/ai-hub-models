# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging

from qai_hub_models import Precision
from qai_hub_models.models.templates.intern3_5_vl.model import (
    Intern3_5VLCollectionBase,
    Intern3_5VLPartBase,
    Intern3_5VLPreSplitBase,
    Intern3_5VLQuantizablePreSplitBase,
    Intern3_5VLSplitForwardMixin,
    Intern3_5VLVisionEncoderBase,
)

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    CollectionCompileO0Mixin,
    CompileO0Mixin,
    QuantizeFullTypeW8A16Mixin,
    SplitForwardMixin,  # noqa: F401
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)

# Model architecture constants (from OpenGVLab/InternVL3_5-2B llm_config)
NUM_LAYERS = 28
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 9
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 16
HEAD_DIM = 128
NUM_DEEPSTACK_LAYERS = 0

# Vision encoder configuration (from OpenGVLab/InternVL3_5-2B vision_config)
VISION_HIDDEN_SIZE = 1024
VISION_OUT_HIDDEN_SIZE = 2048
VISION_DEPTH = 24
VISION_NUM_HEADS = 16
VISION_PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2

# Hugging Face repo
HF_REPO_NAME = "OpenGVLab/InternVL3_5-2B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 40

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT: dict = {
    Precision.w4a16: "fp16mix",
}

# Default image dimensions (must be divisible by patch_size * spatial_merge_size)
DEFAULT_IMAGE_HEIGHT = 448
DEFAULT_IMAGE_WIDTH = 448


def num_visual_tokens_for_image_size(image_size: tuple[int, int]) -> int:
    """Post-merge visual token count for an image: (W/patch)*(H/patch)/merge^2.

    ``image_size`` is ``(width, height)`` to match the dataset/eval convention
    (PIL ``Image.resize`` takes ``(width, height)``).
    """
    width, height = image_size
    return (
        (height // VISION_PATCH_SIZE)
        * (width // VISION_PATCH_SIZE)
        // (SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE)
    )


DEFAULT_NUM_VISUAL_TOKENS = num_visual_tokens_for_image_size(
    (DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)
)

SPLIT_MODEL_NAME = "Intern3_5_VL_2B"


class Intern3_5_VL_2B_PreSplit(Intern3_5VLPreSplitBase):
    """FP PreSplit for InternVL3.5-2B."""

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    vision_patch_size = VISION_PATCH_SIZE
    spatial_merge_size = SPATIAL_MERGE_SIZE
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS
    _hf_repo_name = HF_REPO_NAME


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Intern3_5_VL_2B_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    Intern3_5VLQuantizablePreSplitBase[Intern3_5_VL_2B_PreSplit],
):
    """Quantizable PreSplit for InternVL3.5-2B."""

    FPModel = Intern3_5_VL_2B_PreSplit
    _hf_repo_name = HF_REPO_NAME

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    num_layers = NUM_LAYERS
    # Exact ONNX/AIMET quantizer names to hold at fp16.
    fp16_param_names = ("model.model.layers.2.mlp.down_proj.weight",)
    ada_scale_num_rmsnorm_per_blk: int | None = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1


class Intern3_5_VL_2B_VisionEncoder(Intern3_5VLVisionEncoderBase):
    """Vision encoder for InternVL3.5-2B (adapted VEG for on-device deployment)."""

    DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    _hf_repo_name = HF_REPO_NAME
    vision_patch_size = VISION_PATCH_SIZE
    vision_hidden_size = VISION_HIDDEN_SIZE
    vision_num_heads = VISION_NUM_HEADS
    default_image_height = DEFAULT_IMAGE_HEIGHT
    default_image_width = DEFAULT_IMAGE_WIDTH
    quant_presplit_cls = Intern3_5_VL_2B_QuantizablePreSplit


Intern3_5_VL_2B_QuantizablePreSplit.vision_encoder_cls = Intern3_5_VL_2B_VisionEncoder


class Intern3_5_VL_2B_PartBase(Intern3_5VLPartBase):
    """Unified Part base for InternVL3.5-2B."""

    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    num_splits = NUM_SPLITS
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    default_precision = DEFAULT_PRECISION
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS
    fp_presplit_cls = Intern3_5_VL_2B_PreSplit
    quant_presplit_cls = Intern3_5_VL_2B_QuantizablePreSplit
    export_sequence_lengths = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    export_context_lengths = DEFAULT_EXPORT_CONTEXT_LENGTHS


class Intern3_5_VL_2B_Part1_Of_4(Intern3_5_VL_2B_PartBase):
    part_id = 1


class Intern3_5_VL_2B_Part2_Of_4(Intern3_5_VL_2B_PartBase):
    part_id = 2


class Intern3_5_VL_2B_Part3_Of_4(Intern3_5_VL_2B_PartBase):
    part_id = 3


class Intern3_5_VL_2B_Part4_Of_4(Intern3_5_VL_2B_PartBase):
    part_id = 4


_SPLIT_PART_CLASSES: list[type] = [
    Intern3_5_VL_2B_Part1_Of_4,
    Intern3_5_VL_2B_Part2_Of_4,
    Intern3_5_VL_2B_Part3_Of_4,
    Intern3_5_VL_2B_Part4_Of_4,
]


class FPSplitModelWrapper(Intern3_5VLSplitForwardMixin, Intern3_5_VL_2B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    split_part_classes = _SPLIT_PART_CLASSES
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    Intern3_5VLSplitForwardMixin, Intern3_5_VL_2B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    split_part_classes = _SPLIT_PART_CLASSES
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Intern3_5_VL_2B_Collection(CollectionCompileO0Mixin, Intern3_5VLCollectionBase):
    """Collection model for InternVL3.5-2B deployment.

    Combines 4 text parts + 1 vision encoder for full VLM deployment.
    """

    _hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Intern3_5_VL_2B_PreSplit
    quant_presplit_cls = Intern3_5_VL_2B_QuantizablePreSplit
    part_base_cls = Intern3_5_VL_2B_PartBase
    vision_encoder_cls = Intern3_5_VL_2B_VisionEncoder
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    vision_patch_size = VISION_PATCH_SIZE
    default_image_height = DEFAULT_IMAGE_HEIGHT
    default_image_width = DEFAULT_IMAGE_WIDTH
    default_precision = DEFAULT_PRECISION
    sample_image = SAMPLE_IMAGE
    parts = {
        "part1_of_4": Intern3_5_VL_2B_Part1_Of_4,
        "part2_of_4": Intern3_5_VL_2B_Part2_Of_4,
        "part3_of_4": Intern3_5_VL_2B_Part3_Of_4,
        "part4_of_4": Intern3_5_VL_2B_Part4_Of_4,
    }
