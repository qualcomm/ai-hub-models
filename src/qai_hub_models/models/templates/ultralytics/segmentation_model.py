# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from ultralytics.nn.modules.head import Segment, Segment26, YOLOESegment
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models import Precision
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.ultralytics.segment_patches import (
    patch_ultralytics_segmentation_head,
    patch_ultralytics_segmentation_head_26,
    patch_ultralytics_yoloe_segmentation_head,
)
from qai_hub_models.models.templates.yolo.utils import (
    get_most_likely_score,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    BboxFormat,
    BboxMetadata,
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW = 640


class UltralyticsSingleClassSegmentor(BaseModel):
    """Ultralytics segmentor that segments 1 class."""

    def __init__(self, model: SegmentationModel) -> None:
        super().__init__(model=model)
        patch_ultralytics_segmentation_head(model)
        self.model = model

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run segmentor on `image` and produce segmentation masks.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]. 3-channel Color Space: RGB.

        Returns
        -------
        boxes : torch.Tensor
            Shape [1, num_anchors, 4] where 4 = [x1, y1, x2, y2] (box coordinates in pixel space).
        scores : torch.Tensor
            Shape [batch_size, num_anchors] per-anchor confidence of whether the anchor box contains an object.
        mask_coeffs : torch.Tensor
            Shape [batch_size, num_anchors, num_prototype_masks] per-anchor mask coefficients.
        mask_protos : torch.Tensor
            Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size] mask protos.
        """
        boxes: torch.Tensor
        scores: torch.Tensor
        mask_coeffs: torch.Tensor
        mask_protos: torch.Tensor
        boxes, scores, mask_coeffs, mask_protos = self.model(image)

        # Convert boxes to (x1, y1, x2, y2)
        boxes = box_xywh_to_xyxy(boxes.permute(0, 2, 1))

        return boxes, scores.squeeze(1), mask_coeffs.permute(0, 2, 1), mask_protos

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
        width: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(
                io_type=IoType.TENSOR,
                softmax_applied=True,
                labels_file="coco_labels.txt",
            ),
            "mask_coeffs": TensorSpec(io_type=IoType.TENSOR),
            "mask_protos": TensorSpec(
                io_type=IoType.TENSOR,
                apply_runtime_channel_reordering=True,
            ),
        }


class UltralyticsMulticlassSegmentor(BaseModel):
    """Ultralytics segmentor that segments multiple classes."""

    def __init__(
        self,
        model: SegmentationModel,
        precision: Precision | None = None,
    ) -> None:
        super().__init__(model=model)
        self.precision = precision
        self.num_classes = model.model[-1].nc
        if isinstance(model.model[-1], Segment26):
            patch_ultralytics_segmentation_head_26(model)
        elif isinstance(model.model[-1], Segment):
            patch_ultralytics_segmentation_head(model)
        elif isinstance(model.model[-1], YOLOESegment):
            patch_ultralytics_yoloe_segmentation_head(model)
        else:
            raise NotImplementedError()

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
        width: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(
                io_type=IoType.TENSOR,
                softmax_applied=True,
                labels_file="coco_labels.txt",
            ),
            "mask_coeffs": TensorSpec(io_type=IoType.TENSOR),
            "class_idx": TensorSpec(
                io_type=IoType.TENSOR,
                labels_file="coco_labels.txt",
            ),
            "mask_protos": TensorSpec(
                io_type=IoType.TENSOR,
                apply_runtime_channel_reordering=True,
            ),
        }

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the segmentor on `image` and produce segmentation masks.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]. 3-channel Color Space: RGB.

        Returns
        -------
        boxes : torch.Tensor
            Shape [1, num_anchors, 4] where 4 = [x1, y1, x2, y2] (box coordinates in pixel space).
        scores : torch.Tensor
            Shape [batch_size, num_anchors, num_classes + 1] per-anchor confidence of whether the anchor box contains an object.
        mask_coeffs : torch.Tensor
            Shape [batch_size, num_anchors, num_prototype_masks] per-anchor mask coefficients.
        class_idx : torch.Tensor
            Shape [batch_size, num_anchors] class index.
        mask_protos : torch.Tensor
            Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size] mask protos.
        """
        boxes: torch.Tensor
        scores: torch.Tensor
        mask_coeffs: torch.Tensor
        mask_protos: torch.Tensor
        boxes, scores, mask_coeffs, mask_protos = self.model(image)

        # Convert boxes to (x1, y1, x2, y2)
        boxes = box_xywh_to_xyxy(boxes.permute(0, 2, 1))

        # Get class ID of most likely score.
        scores = scores.permute(0, 2, 1)
        scores, classes = get_most_likely_score(scores)

        if self.precision == Precision.float:
            classes = classes.to(torch.float32)
        elif self.precision is None:
            classes = classes.to(torch.uint8)
        else:
            classes = classes.to(torch.int32)

        return boxes, scores, mask_coeffs.permute(0, 2, 1), classes, mask_protos
