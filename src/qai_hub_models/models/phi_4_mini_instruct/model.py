# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from typing import Any

from qai_hub.client import Device

from qai_hub_models import Precision, TargetRuntime

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
    SplitForwardMixin,
)
from qai_hub_models.models.templates.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models.templates.phi.dataset import TrickyLLMPromptsPhi35
from qai_hub_models.models.templates.phi.model import (
    PhiPartBase,
    PhiPreSplitBase,
    PhiPreSplitCollectionBase,
    PhiQuantizablePreSplitBase,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 13

NUM_LAYERS = 32
NUM_SPLITS = 4
# LM head 671 MB / 1 layer 54 MB
# Part 2/3 -> ~810
# Part 4   -> ~779
NUM_LAYERS_PER_SPLIT = 15
HIDDEN_SIZE = 3072
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 24
PARTIAL_ROTARY_FACTOR = 0.75

# Hugging Face repo
HF_REPO_NAME = "microsoft/Phi-4-mini-instruct"
HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 120

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "phi_4_mini_instruct_w4a16_spinr1r2r3_seqmse_mp_2",
}

# Name used for split ONNX file basenames (e.g. Phi4Mini_1_of_4.onnx)
SPLIT_MODEL_NAME = "Phi4Mini"

# Enable/disable mixed precision
ENABLE_MP = True

# From AIMET QuantAnalyzer on the SeqMSE + SpinQuant R1R2R3 recipe: MatMuls whose
# weights recover >= 0.01 PPL when left unquantized (lm_head is already int8).
_INT8_V_PROJ_LAYERS = (*range(2, 27), 29)
_INT8_K_PROJ_LAYERS = (3, 4, 5, 17, 26)
_INT8_O_PROJ_LAYERS = (0, 8)
_INT8_GATE_PROJ_LAYERS = (5, 15, 18, 19, 21, 28, 29, 30, 31)
_INT8_UP_PROJ_LAYERS = (4, 5, 7, 11, 13, 15, 17, 19, 21, 22, 23, 25, 27, 28, 30, 31)
_INT8_DOWN_PROJ_LAYERS = (1, 3, 7, 10, 19, 20, 21, 22, 24, 25, 26, 30, 31)
INT8_PARAM_NAMES = (
    *(
        f"model.model.layers.{i}.self_attn.{proj}_proj_sha.{head}.weight"
        for proj, layers in (("v", _INT8_V_PROJ_LAYERS), ("k", _INT8_K_PROJ_LAYERS))
        for i in layers
        for head in range(NUM_KEY_VALUE_HEADS)
    ),
    *(
        f"model.model.layers.{i}.self_attn.o_proj_conv.weight"
        for i in _INT8_O_PROJ_LAYERS
    ),
    *(
        f"model.model.layers.{i}.mlp.{proj}.weight"
        for proj, layers in (
            ("gate_proj", _INT8_GATE_PROJ_LAYERS),
            ("up_proj", _INT8_UP_PROJ_LAYERS),
            ("down_proj", _INT8_DOWN_PROJ_LAYERS),
        )
        for i in layers
    ),
)


# ---------------------------------------------------------------------------
# Phi-4-mini-instruct concrete classes
# ---------------------------------------------------------------------------


class Phi4Mini_PreSplit(PhiPreSplitBase):
    """FP PreSplit for Phi-4-mini-instruct."""

    GeneratorClass = HubCompatibleGenerator
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    hf_repo_name = HF_REPO_NAME
    partial_rotary_factor = PARTIAL_ROTARY_FACTOR

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type]:
        from qai_hub_models.models.templates.llm.model import LLMBase

        return [*LLMBase.get_eval_dataset_classes(), TrickyLLMPromptsPhi4Mini]


class TrickyLLMPromptsPhi4Mini(TrickyLLMPromptsPhi35):
    """Tricky eval prompts for the Phi-4-mini-instruct model.

    Defined after Phi4Mini_PreSplit so LLMBaseClass can bind directly;
    Phi4Mini_PreSplit.get_eval_dataset_classes resolves this name at call time.
    """

    MODEL_NAME = "phi4_mini"
    HF_REPO_NAME = "microsoft/Phi-4-mini-instruct"
    LLMBaseClass = Phi4Mini_PreSplit


class Phi4Mini_QuantizablePreSplit(PhiQuantizablePreSplitBase[Phi4Mini_PreSplit]):
    """Quantizable PreSplit for Phi-4-mini-instruct."""

    FPModel = Phi4Mini_PreSplit
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

    if ENABLE_MP:
        int8_param_names = INT8_PARAM_NAMES

        def _apply_ada_scale(self, *args: Any, **kwargs: Any) -> None:
            # AdaScale writes 4-bit offsets into every quantizer it touches;
            # keep the int8 ones uninitialized so it skips them, then calibrate them.
            assert self.quant_sim is not None
            wide_ops = [
                self.quant_sim.qc_quantize_op_dict[name]
                for name in self.int8_param_names
            ]
            for op in wide_ops:
                op.enabled = False
            super()._apply_ada_scale(*args, **kwargs)
            for op in wide_ops:
                op.enabled = True
            self.quant_sim._compute_param_encodings(overwrite=False)


class Phi4Mini_PartBase(PhiPartBase):
    """Unified Part base for Phi-4-mini-instruct."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    partial_rotary_factor = PARTIAL_ROTARY_FACTOR
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Phi4Mini_PreSplit
    quant_presplit_cls = Phi4Mini_QuantizablePreSplit

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        # Skip DynamicSplitPartBase's override to drop --quantize_full_type.
        return MultiGraphWorkbenchModel.get_graph_hub_compile_options(
            self,
            graph_name,
            target_runtime,
            precision,
            other_compile_options + " --quantize_io",
            device,
        )


def _make_part_cls(part_id: int) -> type[Phi4Mini_PartBase]:
    """Build the Part class for a 1-indexed split: embedding, one layer each, LM head last."""
    if part_id == 1:
        doc = "Embedding."
    elif part_id == NUM_SPLITS:
        doc = f"Decoder layer {part_id - 2} + final norm + LM head."
    else:
        doc = f"Decoder layer {part_id - 2}."
    return type(
        f"Phi4Mini_Part{part_id}_Of_{NUM_SPLITS}",
        (Phi4Mini_PartBase,),
        {"part_id": part_id, "__doc__": doc, "__module__": __name__},
    )


_SPLIT_PART_CLASSES: list[type] = [
    _make_part_cls(part_id) for part_id in range(1, NUM_SPLITS + 1)
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Phi4Mini_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Phi4Mini_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Phi4Mini_Collection(PhiPreSplitCollectionBase):
    """Unified Collection with one Part per decoder layer for Phi-4-mini-instruct."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Phi4Mini_PreSplit
    part_base_cls = Phi4Mini_PartBase
    parts = {
        f"part{i}_of_{NUM_SPLITS}": part_cls
        for i, part_cls in enumerate(_SPLIT_PART_CLASSES, start=1)
    }
