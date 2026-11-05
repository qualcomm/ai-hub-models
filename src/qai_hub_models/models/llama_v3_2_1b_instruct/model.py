# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Llama 3.2 1B Instruct - PreSplit-Part architecture for LLM deployment.

The generic PreSplit/Part/Collection machinery lives in
``qai_hub_models.models.templates.llama3.model``. This module supplies the
1B-specific architecture constants and the small concrete subclasses
(Part classes + the decorated Collection) that the ``add_component``
decorator requires.
"""

from __future__ import annotations

import logging

from qai_hub_models import Precision

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models.models.templates.llama3.model import (
    LlamaPartBase,
    LlamaPreSplitBase,
    LlamaPreSplitCollectionBase,
    LlamaQuantizablePreSplitBase,
)
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
from qai_hub_models.models.templates.lm_driver.generator import (
    HubCompatibleGenerator,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 6

# Model architecture constants (from Llama 3.2 1B)
NUM_LAYERS = 16
NUM_SPLITS = 3
NUM_LAYERS_PER_SPLIT = 8
HIDDEN_SIZE = 2048
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging Face repo
HF_REPO_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 50

# Precision settings
DEFAULT_PRECISION = Precision.w4
SUPPORTED_PRECISIONS = [Precision.w4, Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4: "w4",
    Precision.w4a16: "w4a16",
}

# Name used for split ONNX file basenames (e.g. Llama3_2_1B_1_of_3.onnx)
SPLIT_MODEL_NAME = "Llama3_2_1B"


class Llama3_2_1B_PreSplit(LlamaPreSplitBase):
    """FP PreSplit for Llama 3.2 1B."""

    GeneratorClass = HubCompatibleGenerator

    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
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
class Llama3_2_1B_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    LlamaQuantizablePreSplitBase[Llama3_2_1B_PreSplit],
):
    """Quantizable PreSplit for Llama 3.2 1B."""

    FPModel = Llama3_2_1B_PreSplit
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


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
class Llama3_2_1B_PartBase(QuantizeFullTypeW8A16PartMixin, LlamaPartBase):
    """Unified Part base for Llama 3.2 1B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Llama3_2_1B_PreSplit
    quant_presplit_cls = Llama3_2_1B_QuantizablePreSplit


class Llama3_2_1B_Part1_Of_3(Llama3_2_1B_PartBase):
    """Part 1: Embedding + first layers."""

    part_id = 1


class Llama3_2_1B_Part2_Of_3(Llama3_2_1B_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Llama3_2_1B_Part3_Of_3(Llama3_2_1B_PartBase):
    """Part 3: Final layers + LM head."""

    part_id = 3


_SPLIT_PART_CLASSES: list[type] = [
    Llama3_2_1B_Part1_Of_3,
    Llama3_2_1B_Part2_Of_3,
    Llama3_2_1B_Part3_Of_3,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Llama3_2_1B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Llama3_2_1B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Llama3_2_1B_Collection(CollectionCompileO0Mixin, LlamaPreSplitCollectionBase):
    """Unified Collection with 3 Parts for Llama 3.2 1B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Llama3_2_1B_PreSplit
    part_base_cls = Llama3_2_1B_PartBase
    parts = {
        "part1_of_3": Llama3_2_1B_Part1_Of_3,
        "part2_of_3": Llama3_2_1B_Part2_Of_3,
        "part3_of_3": Llama3_2_1B_Part3_Of_3,
    }
