# ---------------------------------------------------------------------
# Copyright (c) 2024-2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from torch import nn
from transformers import DeformableDetrForObjectDetection

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.evaluators.detection_evaluator import DetectionEvaluator
from qai_hub_models.models._shared.detr.model import DETR

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1


class DeformableDETR(DETR):
    """Exportable Deformable DETR model, end-to-end."""

    DEFAULT_WEIGHTS = "SenseTime/deformable-detr"
    HF_DETR_CLS = DeformableDetrForObjectDetection

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def get_evaluator(self) -> BaseEvaluator:
        """
        Returns an instance of the DetectionEvaluator class, which is used to evaluate the performance of the DETR model.

        The DetectionEvaluator class is used to compute the mean average precision (mAP) of the model's predictions.

        Returns
        -------
        BaseEvaluator
            An instance of the DetectionEvaluator class.
        """
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, score_threshold=0.4)
