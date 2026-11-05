# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import cast

from qai_hub_models import Precision

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
from qai_hub_models.models.templates.qwen3_vl.model import (
    Qwen3VLCollectionBase,
    Qwen3VLPartBase,
    Qwen3VLPreSplitBase,
    Qwen3VLQuantizablePreSplitBase,
    Qwen3VLSplitForwardMixin,
    Qwen3VLVisionEncoderBase,
    VisionEncoderCollectionProtocol,
    VisionEncoderExportProtocol,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096]
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
SAMPLE_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)

# Model architecture constants (from Qwen3-VL-8B-Instruct)
NUM_LAYERS = 36
NUM_SPLITS = 6
NUM_LAYERS_PER_SPLIT = 6
HIDDEN_SIZE = 4096
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32
HEAD_DIM = 128
NUM_DEEPSTACK_LAYERS = 3

# Vision encoder configuration
VISION_HIDDEN_SIZE = 1152
VISION_OUT_HIDDEN_SIZE = 4096
VISION_DEPTH = 27
VISION_NUM_HEADS = 16
VISION_PATCH_SIZE = 16
SPATIAL_MERGE_SIZE = 2

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen3-VL-8B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 150

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT: dict = {
    Precision.w4a16: "w4a16",
}

# Default image dimensions (must be divisible by patch_size * spatial_merge_size)
DEFAULT_IMAGE_HEIGHT = 512
DEFAULT_IMAGE_WIDTH = 512


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

SPLIT_MODEL_NAME = "Qwen3_VL_8B"


class Qwen3_VL_8B_PreSplit(Qwen3VLPreSplitBase):
    """FP PreSplit for Qwen3-VL-8B."""

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
class Qwen3_VL_8B_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    Qwen3VLQuantizablePreSplitBase[Qwen3_VL_8B_PreSplit],
):
    """Quantizable PreSplit for Qwen3-VL-8B."""

    FPModel = Qwen3_VL_8B_PreSplit
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

    # SHA produces per-head q_norm/k_norm nodes in the ONNX graph.
    # Between block starts (input_layernorm): NUM_ATTN_HEADS q_norms
    # + NUM_KEY_VALUE_HEADS k_norms + 1 post_attention_layernorm = 41 intermediate ops
    ada_scale_num_rmsnorm_per_blk: int | None = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1


class Qwen3_VL_8B_VisionEncoder(Qwen3VLVisionEncoderBase):
    """Vision encoder for Qwen3-VL-8B (adapted VEG for on-device deployment)."""

    DEFAULT_IMAGE_SIZE = (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    _hf_repo_name = HF_REPO_NAME
    vision_patch_size = VISION_PATCH_SIZE
    vision_hidden_size = VISION_HIDDEN_SIZE
    vision_num_heads = VISION_NUM_HEADS
    default_image_height = DEFAULT_IMAGE_HEIGHT
    default_image_width = DEFAULT_IMAGE_WIDTH
    quant_presplit_cls = Qwen3_VL_8B_QuantizablePreSplit


# Circular dependency: QuantizablePreSplit.vision_encoder_cls -> VisionEncoder and
# VisionEncoder.quant_presplit_cls -> QuantizablePreSplit. One must be post-hoc.
Qwen3_VL_8B_QuantizablePreSplit.vision_encoder_cls = cast(
    type[VisionEncoderExportProtocol], Qwen3_VL_8B_VisionEncoder
)


class Qwen3_VL_8B_PartBase(Qwen3VLPartBase):
    """Unified Part base for Qwen3-VL-8B."""

    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    num_splits = NUM_SPLITS
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    default_precision = DEFAULT_PRECISION
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS
    fp_presplit_cls = Qwen3_VL_8B_PreSplit
    quant_presplit_cls = Qwen3_VL_8B_QuantizablePreSplit
    export_sequence_lengths = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    export_context_lengths = DEFAULT_EXPORT_CONTEXT_LENGTHS


class Qwen3_VL_8B_Part1_Of_6(Qwen3_VL_8B_PartBase):
    part_id = 1


class Qwen3_VL_8B_Part2_Of_6(Qwen3_VL_8B_PartBase):
    part_id = 2


class Qwen3_VL_8B_Part3_Of_6(Qwen3_VL_8B_PartBase):
    part_id = 3


class Qwen3_VL_8B_Part4_Of_6(Qwen3_VL_8B_PartBase):
    part_id = 4


class Qwen3_VL_8B_Part5_Of_6(Qwen3_VL_8B_PartBase):
    part_id = 5


class Qwen3_VL_8B_Part6_Of_6(Qwen3_VL_8B_PartBase):
    part_id = 6


_SPLIT_PART_CLASSES: list[type] = [
    Qwen3_VL_8B_Part1_Of_6,
    Qwen3_VL_8B_Part2_Of_6,
    Qwen3_VL_8B_Part3_Of_6,
    Qwen3_VL_8B_Part4_Of_6,
    Qwen3_VL_8B_Part5_Of_6,
    Qwen3_VL_8B_Part6_Of_6,
]


class FPSplitModelWrapper(Qwen3VLSplitForwardMixin, Qwen3_VL_8B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    split_part_classes = _SPLIT_PART_CLASSES
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    Qwen3VLSplitForwardMixin, Qwen3_VL_8B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    split_part_classes = _SPLIT_PART_CLASSES
    default_num_visual_tokens = DEFAULT_NUM_VISUAL_TOKENS


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Qwen3_VL_8B_Collection(CollectionCompileO0Mixin, Qwen3VLCollectionBase):
    """Collection model for Qwen3-VL-8B deployment.

    Combines 6 text parts + 1 vision encoder for full VLM deployment.
    """

    _hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Qwen3_VL_8B_PreSplit
    quant_presplit_cls = Qwen3_VL_8B_QuantizablePreSplit
    part_base_cls = Qwen3_VL_8B_PartBase
    vision_encoder_cls = cast(
        type[VisionEncoderCollectionProtocol], Qwen3_VL_8B_VisionEncoder
    )
    num_deepstack_layers = NUM_DEEPSTACK_LAYERS
    vision_patch_size = VISION_PATCH_SIZE
    default_image_height = DEFAULT_IMAGE_HEIGHT
    default_image_width = DEFAULT_IMAGE_WIDTH
    default_precision = DEFAULT_PRECISION
    sample_image = SAMPLE_IMAGE
    parts = {
        "part1_of_6": Qwen3_VL_8B_Part1_Of_6,
        "part2_of_6": Qwen3_VL_8B_Part2_Of_6,
        "part3_of_6": Qwen3_VL_8B_Part3_Of_6,
        "part4_of_6": Qwen3_VL_8B_Part4_Of_6,
        "part5_of_6": Qwen3_VL_8B_Part5_Of_6,
        "part6_of_6": Qwen3_VL_8B_Part6_Of_6,
    }
