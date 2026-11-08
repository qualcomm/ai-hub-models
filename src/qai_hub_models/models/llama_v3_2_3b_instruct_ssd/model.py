# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import itertools
import logging
import os
from pathlib import Path
from typing import Any

from qai_hub_models import Precision
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tensor_spec import TensorSpec

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub_models.models.templates.llama3.model import (
    LlamaPartBase,
    LlamaPreSplitBase,
    LlamaPreSplitCollectionBase,
    LlamaQuantizablePreSplitBase,
)
from qai_hub_models.models.templates.llm.common import LLMIOType
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models.templates.llm.model import (
    CollectionCompileO0Mixin,
    CompileO0Mixin,
    QuantizeFullTypeW8A16Mixin,
    QuantizeFullTypeW8A16PartMixin,
    SplitForwardMixin,
)
from qai_hub_models.models.templates.llm.native_kv_generator import (
    HubCompatibleNativeKVGenerator,
)
from qai_hub_models.models.templates.llm_ssd.model import (
    LLMDynamic_SSD_AIMETOnnx,
    append_ssd_forecast_embeddings,
    apply_ssd_genie_assets,
)
from qai_hub_models.models.templates.lm_driver.generator import Generator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import OutputSpec

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
# SSD uses a smaller "token" sequence length (32) than the standard models so
# the speculative-decoding forecast tokens fit in one prompt-processor graph.
# Native KV supports AR32 and AR128 only, so AR1 is not compiled; on device SSD
# runs AR-32 for both prefill and decode and never dispatches an AR-1 graph.
DEFAULT_EXPORT_SEQUENCE_LENGTHS = [32, 128]
# The local demo runs the torch/AIMET model one token at a time, so it still
# needs an AR-1 bucket alongside the two exported ones.
DEFAULT_DEMO_SEQUENCE_LENGTHS = [1, 32, 128]

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4

# Model architecture constants (from Llama 3.2 3B)
NUM_LAYERS = 28
NUM_SPLITS = 4
NUM_LAYERS_PER_SPLIT = 14
HIDDEN_SIZE = 3072
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 24

# Hugging Face repo
HF_REPO_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 80

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "w4a16",
}

# Name used for split ONNX file basenames (e.g. Llama3_2_3B_SSD_1_of_4.onnx)
SPLIT_MODEL_NAME = "Llama3_2_3B_SSD"


def _native_kv_output_spec(num_hidden_layers: int) -> OutputSpec:
    """Output spec with per-layer native KV cache names (ScatterElements)."""
    output_spec: OutputSpec = {"logits": TensorSpec()}
    for layer in range(num_hidden_layers):
        output_spec[f"past_nativekvcache__key_{layer}_out"] = TensorSpec()
        output_spec[f"past_nativekvcache__value_{layer}_out"] = TensorSpec()
    return output_spec


class Llama3_2_3B_SSD_PreSplit(LlamaPreSplitBase):
    """FP PreSplit for Llama 3.2 3B with SSD forecast embeddings."""

    model_id = MODEL_ID
    llm_io_type = LLMIOType.genie_input_ids_native_kv
    GeneratorClass: type[Generator] = HubCompatibleNativeKVGenerator
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    hf_repo_name = HF_REPO_NAME
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = True
    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    @staticmethod
    def _get_output_spec(num_hidden_layers: int) -> OutputSpec:
        return _native_kv_output_spec(num_hidden_layers)

    def get_output_spec(self) -> OutputSpec:
        return _native_kv_output_spec(self.num_layers)

    @classmethod
    def _ssd_forecast_ckpt(cls) -> Path | None:
        """Fetch the SSD self-speculative-decoding forecast module checkpoint.

        Defined as a classmethod (rather than a module-level helper) so the
        shared ``LLMDynamic_SSD_AIMETOnnx.prepare_genie_assets`` can reach it via
        ``cls.FPModel._ssd_forecast_ckpt()``.
        """
        return CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "forecast_module_state_dict.pt"
        ).fetch()

    def __init__(
        self,
        checkpoint: str | os.PathLike | Path | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint, *args, **kwargs)
        # Extend the embedding table with the SSD forecast token embeddings.
        append_ssd_forecast_embeddings(self.model, self._ssd_forecast_ckpt())


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Llama3_2_3B_SSD_QuantizablePreSplit(  # type: ignore[misc]
    QuantizeFullTypeW8A16Mixin,
    CompileO0Mixin,
    LLMDynamic_SSD_AIMETOnnx,
    LlamaQuantizablePreSplitBase[Llama3_2_3B_SSD_PreSplit],
):
    """Quantizable PreSplit for Llama 3.2 3B with SSD genie-asset support."""

    FPModel = Llama3_2_3B_SSD_PreSplit
    GeneratorClass: type[Generator] = HubCompatibleNativeKVGenerator
    llm_io_type = LLMIOType.genie_input_ids_native_kv

    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    num_layers = NUM_LAYERS
    supported_precisions = SUPPORTED_PRECISIONS
    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT
    split_lm_head = True
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    def get_qnn_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """Native KV graphs use no prompt_/token_ prefix."""
        return f"ar{self.sequence_length}_cl{self.context_length}_{split_index + 1}_of_{num_splits}"

    def get_output_spec(self) -> OutputSpec:
        return _native_kv_output_spec(self.num_layers)


# TODO: remove the --quantize_full_type mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22269
class Llama3_2_3B_SSD_PartBase(QuantizeFullTypeW8A16PartMixin, LlamaPartBase):
    """Unified Part base for Llama 3.2 3B SSD."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    fp_presplit_cls = Llama3_2_3B_SSD_PreSplit
    quant_presplit_cls = Llama3_2_3B_SSD_QuantizablePreSplit
    default_precision = DEFAULT_PRECISION

    def _build_graph_names(
        self, sequence_lengths: list[int], context_lengths: list[int]
    ) -> dict[str, tuple[int, int]]:
        """Native KV graphs use no prompt_/token_ prefix."""
        return {
            f"ar{seq_len}_cl{ctx_len}_{self.part_id}_of_{self.num_splits}": (
                seq_len,
                ctx_len,
            )
            for seq_len, ctx_len in itertools.product(sequence_lengths, context_lengths)
        }

    def get_qnn_context_graph_name(self, split_index: int, num_splits: int) -> str:
        """Native KV graphs use no prompt_/token_ prefix."""
        return f"ar{self.sequence_length}_cl{self.context_length}_{split_index + 1}_of_{num_splits}"


class Llama3_2_3B_SSD_Part1_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Llama3_2_3B_SSD_Part2_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 2: Middle layers."""

    part_id = 2


class Llama3_2_3B_SSD_Part3_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 3: Middle layers."""

    part_id = 3


class Llama3_2_3B_SSD_Part4_Of_4(Llama3_2_3B_SSD_PartBase):
    """Part 4: Final layers + LM head."""

    part_id = 4


_SPLIT_PART_CLASSES: list[type] = [
    Llama3_2_3B_SSD_Part1_Of_4,
    Llama3_2_3B_SSD_Part2_Of_4,
    Llama3_2_3B_SSD_Part3_Of_4,
    Llama3_2_3B_SSD_Part4_Of_4,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Llama3_2_3B_SSD_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Llama3_2_3B_SSD_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


# TODO: remove the -O0 mixin; see
# https://github.com/qcom-ai-hub/tetracode/issues/22094
class Llama3_2_3B_SSD_Collection(CollectionCompileO0Mixin, LlamaPreSplitCollectionBase):
    """Unified Collection with 4 Parts for Llama 3.2 3B SSD."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Llama3_2_3B_SSD_PreSplit
    part_base_cls = Llama3_2_3B_SSD_PartBase
    default_sequence_lengths = DEFAULT_EXPORT_SEQUENCE_LENGTHS
    parts = {
        "part1_of_4": Llama3_2_3B_SSD_Part1_Of_4,
        "part2_of_4": Llama3_2_3B_SSD_Part2_Of_4,
        "part3_of_4": Llama3_2_3B_SSD_Part3_Of_4,
        "part4_of_4": Llama3_2_3B_SSD_Part4_Of_4,
    }

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        # Write the standard genie bundle (genie_config.json, tokenizer, etc.),
        # then layer the SSD-specific assets on top: the quantized
        # forecast-prefix KV-cache file and the ssd-q1 dialog config. The
        # legacy export path did this in prepare_genie_assets, which the
        # modernized collection export no longer calls.
        super().write_supplementary_files(output_dir, metadata)

        # The forecast-prefix KV-cache is quantized using the full (unsplit)
        # model's activation encodings, which live alongside the resolved
        # checkpoint as model.encodings.
        first_part = next(iter(self.components.values()))
        assert isinstance(first_part, self.part_base_cls)
        checkpoint = getattr(first_part._presplit, "checkpoint", None)
        if checkpoint is None:
            logger.warning(
                "SSD collection has no resolved checkpoint; skipping SSD genie "
                "assets. The exported bundle will not enable self-speculative "
                "decoding."
            )
            return
        encodings_path = Path(checkpoint) / "model.encodings"
        if not encodings_path.exists():
            logger.warning(
                "Expected model.encodings at %s for SSD genie assets, but it "
                "was not found; skipping SSD genie assets.",
                encodings_path,
            )
            return

        output_path = Path(output_dir)
        apply_ssd_genie_assets(
            output_path=output_path,
            encodings_path=encodings_path,
            ssd_forecast_ckpt=self.fp_presplit_cls._ssd_forecast_ckpt(),
        )
        metadata.supplementary_files["forecast-prefix/kv-cache.primary.qnn-htp"] = (
            "Quantized forecast-prefix KV-cache for SSD self-speculative decoding."
        )
