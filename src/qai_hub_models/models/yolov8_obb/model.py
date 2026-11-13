# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from typing import cast

import torch
from typing_extensions import Self
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import OBBModel

from qai_hub_models import Precision
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.templates.ultralytics.obb_patches import (
    patch_ultralytics_obb_head,
)
from qai_hub_models.models.templates.yolo.model import (
    Yolo,
    yolo_obb_postprocess,
)
from qai_hub_models.models.yolov8_obb.dataset import Dota128Dataset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import OutputSpec
from qai_hub_models.utils.labels import write_labels_file

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

SUPPORTED_WEIGHTS = [
    "yolov8n-obb.pt",
    "yolov8s-obb.pt",
    "yolov8m-obb.pt",
    "yolov8l-obb.pt",
    "yolov8x-obb.pt",
]
DEFAULT_WEIGHTS = "yolov8n-obb.pt"


class YoloV8OBB(Yolo):
    """Exportable YoloV8 oriented bounding box detector, end-to-end."""

    def __init__(
        self,
        model: OBBModel,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__(model=model)
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        patch_ultralytics_obb_head(model)

    @classmethod
    def from_pretrained(
        cls,
        ckpt_name: str = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        model = cast(OBBModel, ultralytics_YOLO(ckpt_name).model)
        return cls(
            model,
            include_postprocessing,
            split_output,
        )

    def forward(
        self, image: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """
        Run YoloV8-OBB on `image`, and produce a predicted set of oriented bounding boxes and associated class probabilities.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        result : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor
            If self.include_postprocessing is True, returns:
            boxes
                Bounding box locations. Shape is [batch, num_preds, 4] where 4 == (x_center, y_center, w, h).
            angles
                Orientation (in radians) corresponding to the boxes. Shape is [batch, num_preds, 1].
            scores
                Class scores multiplied by confidence. Shape is [batch, num_preds].
            classes
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.

            If self.include_postprocessing is False and self.split_output is True, returns:
            boxes
                Bounding box predictions in xywh format. Shape [batch, 4, num_preds].
            angles
                Orientation (in radians) corresponding to the boxes. Shape is [batch, 1, num_preds].
            scores
                Full score distribution over all classes for each box. Shape [batch, num_classes, num_preds].

            If self.include_postprocessing is False and self.split_output is False, returns:
            detector_output
                Boxes and scores concatenated into a single tensor. Shape [batch, 5 + num_classes, num_preds].
        """
        boxes, angles, scores = self.model(image)
        if not self.include_postprocessing:
            if self.split_output:
                return boxes, angles, scores
            return torch.cat([boxes, angles, scores], dim=1)

        boxes, angles, scores, classes = yolo_obb_postprocess(boxes, angles, scores)
        return boxes, angles, scores, classes

    def get_output_spec(self) -> OutputSpec:
        if self.include_postprocessing:
            out = ["boxes", "angles", "scores", "class_idx"]
        elif self.split_output:
            out = ["boxes", "angles", "scores"]
        else:
            out = ["detector_output"]
        return {x: TensorSpec() for x in out}

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        if precision in {Precision.w8a8_mixed_int16, Precision.w8a16_mixed_int16}:
            options += f" --range_scheme min_max --lite_mp percentage={self.get_hub_litemp_percentage(precision)};override_qtype=int16"
        elif precision in {Precision.w8a8_mixed_fp16, Precision.w8a16_mixed_fp16}:
            options += f" --range_scheme min_max --lite_mp percentage={self.get_hub_litemp_percentage(precision)};override_qtype=fp16"
        else:
            options += " --range_scheme min_max"
        return options

    @staticmethod
    def get_hub_litemp_percentage(precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type]:
        return [Dota128Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Dota128Dataset

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("dota_v1", output_dir, metadata)

    def get_evaluator(self) -> BaseEvaluator:
        from qai_hub_models.models.yolov8_obb.evaluator import OBBEvaluator

        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return OBBEvaluator(
            num_classes=15,
            image_height=image_height,
            image_width=image_width,
            score_threshold=0.01,
            nms_iou_threshold=0.7,
        )
