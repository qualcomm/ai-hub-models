# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import torch
from qai_hub.public_rest_api import DatasetEntries
from torch.utils.data import DataLoader
from tqdm import tqdm

from qai_hub_models import Precision

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models.datasets import instantiate_dataset
from qai_hub_models.datasets.wikitext import WikiText, WikiTextJapanese
from qai_hub_models.models.templates.llama3.model import (
    LlamaPartBase,
    LlamaPreSplitBase,
    LlamaPreSplitCollectionBase,
    LlamaQuantizablePreSplitBase,
)
from qai_hub_models.models.templates.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models.templates.llm.generator_factory import make_generator
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    CollectionCompileO0Mixin,
    CompileO0Mixin,
    QuantizeFullTypeW8A16Mixin,
    QuantizeFullTypeW8A16PartMixin,
    SplitForwardMixin,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models.templates.lm_driver.generator import (
    HubCompatibleGenerator,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 5

# Model architecture constants (from Llama-3-ELYZA-JP 8B)
NUM_LAYERS = 32
NUM_SPLITS = 5
NUM_LAYERS_PER_SPLIT = 9
HIDDEN_SIZE = 4096
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 32

# Hugging Face repo
HF_REPO_NAME = "elyza/Llama-3-ELYZA-JP-8B"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 150

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "w4a16",
}

# Name used for split ONNX file basenames (e.g. Llama3_Elyza_JP_8B_1_of_5.onnx)
SPLIT_MODEL_NAME = "Llama3_Elyza_JP_8B"


class Llama3_Elyza_JP_8B_PreSplit(LlamaPreSplitBase):
    """FP PreSplit for Llama-3-ELYZA-JP 8B."""

    # Default prompts for demos (Japanese)
    default_user_prompt = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"
    default_system_prompt = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"

    model_id = MODEL_ID
    GeneratorClass = HubCompatibleGenerator
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    hf_repo_name = HF_REPO_NAME
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Llama3_Elyza_JP_8B_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    LlamaQuantizablePreSplitBase[Llama3_Elyza_JP_8B_PreSplit],
):
    """Quantizable PreSplit for Llama-3-ELYZA-JP 8B."""

    FPModel = Llama3_Elyza_JP_8B_PreSplit
    GeneratorClass = HubCompatibleGenerator

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    supported_precisions = SUPPORTED_PRECISIONS
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    def get_calibration_data(  # type: ignore[override]
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
    ) -> DatasetEntries | None:
        """Calibrate on a 95%/5% mix of English and Japanese WikiText."""
        if num_samples == 0:
            num_samples = math.ceil(84000 / context_length)
        eng_num_samples, ja_num_samples = (
            round(num_samples * 0.95),
            round(num_samples * 0.05),
        )
        dataset_eng = instantiate_dataset(
            WikiText,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=eng_num_samples,
        )
        english_dataset_entries = DataLoader(
            dataset_eng,
            batch_size=1,
            collate_fn=dataset_eng.collate_fn,
        )
        dataset_ja = instantiate_dataset(
            WikiTextJapanese,
            DatasetSplit.TRAIN,
            input_spec=None,
            tokenizer=self.tokenizer,
            block_size=sequence_length,
            context_length=context_length,
            num_samples=ja_num_samples,
        )
        japanese_dataset_entries = DataLoader(
            dataset_ja,
            batch_size=1,
            collate_fn=dataset_ja.collate_fn,
        )

        dataloader = itertools.chain(english_dataset_entries, japanese_dataset_entries)
        num_combined_entries = len(english_dataset_entries) + len(
            japanese_dataset_entries
        )

        input_spec = self.get_input_spec(
            llm_config=self.llm_config.to_dict(),
            sequence_length=sequence_length,
            context_length=context_length,
            llm_io_type=self.llm_io_type,
        )
        assert input_spec is not None
        inputs: list[list[torch.Tensor | np.ndarray]] = [
            [] for _ in range(len(input_spec))
        ]

        generator = make_generator(
            self, sequence_length=sequence_length, context_length=context_length
        )

        with self.remove_quantization():
            for sample in tqdm(
                dataloader,
                total=num_combined_entries,
                desc="Pre-filling calibration data",
            ):
                input_ids, attention_mask, _ = sample
                for prefilled_inputs in generator.prefill(input_ids, attention_mask):
                    for i, tensor in enumerate(prefilled_inputs.values()):
                        inputs[i].append(tensor.cpu())

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
class Llama3_Elyza_JP_8B_PartBase(QuantizeFullTypeW8A16PartMixin, LlamaPartBase):
    """Unified Part base for Llama-3-ELYZA-JP 8B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    fp_presplit_cls = Llama3_Elyza_JP_8B_PreSplit
    quant_presplit_cls = Llama3_Elyza_JP_8B_QuantizablePreSplit
    default_precision = DEFAULT_PRECISION


class Llama3_Elyza_JP_8B_Part1_Of_5(Llama3_Elyza_JP_8B_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Llama3_Elyza_JP_8B_Part2_Of_5(Llama3_Elyza_JP_8B_PartBase):
    """Part 2."""

    part_id = 2


class Llama3_Elyza_JP_8B_Part3_Of_5(Llama3_Elyza_JP_8B_PartBase):
    """Part 3."""

    part_id = 3


class Llama3_Elyza_JP_8B_Part4_Of_5(Llama3_Elyza_JP_8B_PartBase):
    """Part 4."""

    part_id = 4


class Llama3_Elyza_JP_8B_Part5_Of_5(Llama3_Elyza_JP_8B_PartBase):
    """Part 5: Final layers + LM head."""

    part_id = 5


_SPLIT_PART_CLASSES: list[type] = [
    Llama3_Elyza_JP_8B_Part1_Of_5,
    Llama3_Elyza_JP_8B_Part2_Of_5,
    Llama3_Elyza_JP_8B_Part3_Of_5,
    Llama3_Elyza_JP_8B_Part4_Of_5,
    Llama3_Elyza_JP_8B_Part5_Of_5,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Llama3_Elyza_JP_8B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Llama3_Elyza_JP_8B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Llama3_Elyza_JP_8B_Collection(
    CollectionCompileO0Mixin, LlamaPreSplitCollectionBase
):
    """Unified Collection with 5 Parts for Llama-3-ELYZA-JP 8B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Llama3_Elyza_JP_8B_PreSplit
    part_base_cls = Llama3_Elyza_JP_8B_PartBase
    parts = {
        "part1_of_5": Llama3_Elyza_JP_8B_Part1_Of_5,
        "part2_of_5": Llama3_Elyza_JP_8B_Part2_Of_5,
        "part3_of_5": Llama3_Elyza_JP_8B_Part3_Of_5,
        "part4_of_5": Llama3_Elyza_JP_8B_Part4_Of_5,
        "part5_of_5": Llama3_Elyza_JP_8B_Part5_Of_5,
    }
