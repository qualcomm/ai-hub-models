# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Qwen3.5-4B text decoder."""

from __future__ import annotations

from transformers import PretrainedConfig

from qai_hub_models import Precision

# LLMIOType is re-exported so the CLI input-spec parser can resolve the
# inherited get_input_spec's "llm_io_type" annotation in this module.
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.model import SplitForwardMixin
from qai_hub_models.models.templates.qwen3_5.model import (
    Qwen3_5CollectionBase,
    Qwen3_5DynamicBase,
    Qwen3_5PartBase,
    Qwen3_5QuantizableBase,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

HF_REPO_NAME = "Qwen/Qwen3.5-4B"
DEFAULT_PRECISION = Precision.w4a16
# No published quantized checkpoint yet; quantize.py writes a local one.
SUPPORTED_PRECISIONS = [Precision.w4a16]

# 32 layers: every 4th is full attention, the rest are Gated DeltaNet.
NUM_LAYERS = 32
# Embedding + 4 x 8 layers; 8 keeps two full-attention layers in each part.
NUM_SPLITS = 5
NUM_LAYERS_PER_SPLIT = 8
SPLIT_MODEL_NAME = "Qwen3_5_4B"
HIDDEN_SIZE = 2560
NUM_ATTN_HEADS = 16
NUM_KEY_VALUE_HEADS = 4
HEAD_DIM = 256

MIN_MEMORY_RECOMMENDED = 40


def _verify_4b_config(llm_config: PretrainedConfig) -> None:
    if not (
        llm_config.num_hidden_layers == NUM_LAYERS
        and llm_config.hidden_size == HIDDEN_SIZE
        and llm_config.num_attention_heads == NUM_ATTN_HEADS
        and llm_config.num_key_value_heads == NUM_KEY_VALUE_HEADS
        and llm_config.head_dim == HEAD_DIM
    ):
        raise ValueError("Model config is not compatible with our implementation.")


class Qwen3_5_4B_PreSplit(Qwen3_5DynamicBase):
    """Qwen3.5-4B float model used by quantize.py and evaluate.py."""

    hf_repo_name = HF_REPO_NAME
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    def _verify_ckpt(self) -> None:
        super()._verify_ckpt()
        _verify_4b_config(self.llm_config)


class Qwen3_5_4B_QuantizablePreSplit(Qwen3_5QuantizableBase[Qwen3_5_4B_PreSplit]):
    """Qwen3.5-4B QuantSim model."""

    FPModel = Qwen3_5_4B_PreSplit
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    supported_precisions = SUPPORTED_PRECISIONS

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT


class Qwen3_5_4B_PartBase(Qwen3_5PartBase):
    """Unified Part base for Qwen3.5-4B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    default_precision = Precision.w4a16
    fp_presplit_cls = Qwen3_5_4B_PreSplit
    quant_presplit_cls = Qwen3_5_4B_QuantizablePreSplit


class Qwen3_5_4B_Part1_Of_5(Qwen3_5_4B_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Qwen3_5_4B_Part2_Of_5(Qwen3_5_4B_PartBase):
    """Part 2: Layers 0-7."""

    part_id = 2


class Qwen3_5_4B_Part3_Of_5(Qwen3_5_4B_PartBase):
    """Part 3: Layers 8-15."""

    part_id = 3


class Qwen3_5_4B_Part4_Of_5(Qwen3_5_4B_PartBase):
    """Part 4: Layers 16-23."""

    part_id = 4


class Qwen3_5_4B_Part5_Of_5(Qwen3_5_4B_PartBase):
    """Part 5: Layers 24-31 + LM head."""

    part_id = 5


_SPLIT_PART_CLASSES: list[type] = [
    Qwen3_5_4B_Part1_Of_5,
    Qwen3_5_4B_Part2_Of_5,
    Qwen3_5_4B_Part3_Of_5,
    Qwen3_5_4B_Part4_Of_5,
    Qwen3_5_4B_Part5_Of_5,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Qwen3_5_4B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Qwen3_5_4B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Qwen3_5_4B_Collection(Qwen3_5CollectionBase):
    """Unified Collection with 5 Parts for Qwen3.5-4B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Qwen3_5_4B_PreSplit
    part_base_cls = Qwen3_5_4B_PartBase
    parts = {
        "part1_of_5": Qwen3_5_4B_Part1_Of_5,
        "part2_of_5": Qwen3_5_4B_Part2_Of_5,
        "part3_of_5": Qwen3_5_4B_Part3_Of_5,
        "part4_of_5": Qwen3_5_4B_Part4_Of_5,
        "part5_of_5": Qwen3_5_4B_Part5_Of_5,
    }
