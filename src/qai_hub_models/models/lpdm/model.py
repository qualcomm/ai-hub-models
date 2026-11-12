# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import cast

import torch
from qai_hub.client import Device
from typing_extensions import Self
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import DetectionModel

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tensor_spec import (
    BboxFormat,
    BboxMetadata,
    IoType,
    TensorSpec,
)
from qai_hub_models.models.lpdm.dataset import RoboflowCarPlateDataset
from qai_hub_models.models.lpdm.evaluator import (
    LicensePlateEvaluator,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.corevision.labels import write_labels_file
from qai_hub_models.models.templates.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models.templates.yolo.model import (
    Yolo,
    yolo_detect_postprocess,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# NMS thresholds for detection post-processing.
SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45

# Weights trained on the UAE licence-plate dataset (Roboflow car-plate-k6xij v26).
# 51 classes: digits 0-9, letters A-Z, UAE emirate identifiers, and 'plate'.
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "best.pt"
)


class LicensePlateDetector(Yolo):
    """Exportable YOLOv8-based licence-plate detector, end-to-end.

    Detects and classifies UAE licence-plate characters and region identifiers.
    The model was trained on the Roboflow car-plate-k6xij v26 dataset with
    51 classes: digits 0-9, letters A-Z, UAE emirate identifiers, and a
    ``plate`` bounding-box class.

    Input:  RGB image, float32, [0, 1], shape (1, 3, 640, 640).
    Output: boxes (N, 4), scores (N,), class indices (N,).
    """

    def __init__(
        self,
        model: DetectionModel,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> None:
        super().__init__(model=model)
        self.include_postprocessing = include_postprocessing
        self.split_output = split_output
        patch_ultralytics_detection_head(model)

    @classmethod
    def from_pretrained(
        cls,
        weights: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
        include_postprocessing: bool = True,
        split_output: bool = False,
    ) -> Self:
        """Load pretrained weights.

        Parameters
        ----------
        weights
            Path to a ``.pt`` checkpoint or a ``CachedWebModelAsset`` that
            resolves to one.  Defaults to the bundled best.pt weights trained
            on the UAE licence-plate dataset.
        include_postprocessing
            If True, applies NMS and returns (boxes, scores, classes).
        split_output
            If True and include_postprocessing is False, splits raw output
            into separate boxes and scores tensors.

        Returns
        -------
        Self
            Loaded LicensePlateDetector instance.
        """
        if isinstance(weights, CachedWebModelAsset):
            weights = str(weights.fetch())
        model = cast(DetectionModel, ultralytics_YOLO(weights).model)
        return cls(model, include_postprocessing, split_output)

    def forward(
        self, image: torch.Tensor
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
        | torch.Tensor
    ):
        """Run the detector on *image*.

        Parameters
        ----------
        image
            Pre-processed pixel values.  Range ``float[0, 1]``, RGB,
            shape ``(batch, 3, 640, 640)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor
            With ``include_postprocessing=True``: (boxes [B,N,4], scores [B,N], classes [B,N]).
            With ``include_postprocessing=False, split_output=True``: (boxes [B,4,N], scores [B,num_classes,N]).
            With ``include_postprocessing=False, split_output=False``: detector_output [B,4+num_classes,N].
        """
        boxes, scores = self.model(image)
        if not self.include_postprocessing:
            if self.split_output:
                return boxes, scores
            return torch.cat([boxes, scores], dim=1)
        boxes, scores, classes = yolo_detect_postprocess(boxes, scores)
        return boxes, scores, classes

    def get_evaluator(self) -> BaseEvaluator:
        h, w = self.get_input_spec()["image"][0][2:]
        return LicensePlateEvaluator(
            image_height=h,
            image_width=w,
            score_threshold=SCORE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [RoboflowCarPlateDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return RoboflowCarPlateDataset

    def get_output_spec(self) -> OutputSpec:
        if self.include_postprocessing:
            return {
                "boxes": TensorSpec(
                    io_type=IoType.BBOX,
                    bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
                ),
                "scores": TensorSpec(
                    io_type=IoType.TENSOR,
                    softmax_applied=True,
                    labels_file="roboflow_car_plate_labels.txt",
                ),
                "class_idx": TensorSpec(
                    io_type=IoType.TENSOR,
                    labels_file="roboflow_car_plate_labels.txt",
                ),
            }
        if self.split_output:
            return {"boxes": TensorSpec(), "scores": TensorSpec()}
        return {"detector_output": TensorSpec()}

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("roboflow_car_plate", output_dir, metadata)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "car.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
