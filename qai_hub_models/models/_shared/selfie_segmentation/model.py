# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.segmentation_evaluator import SegmentationOutputEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec


class SelfieSegmentor(BaseModel):
    MASK_THRESHOLD: float  # Threshold above which a pixel is classified as foreground in the binary mask output by the model.
    DEFAULT_HW: tuple[int, int] = (256, 256)

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = DEFAULT_HW[0],
        width: int = DEFAULT_HW[1],
    ) -> InputSpec:
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["mask"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask"]

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(2, mask_threshold=self.MASK_THRESHOLD)
