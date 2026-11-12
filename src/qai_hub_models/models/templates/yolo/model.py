# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.coco import CocoDataset, CocoSegDataset
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.corevision.labels import write_labels_file
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
)
from qai_hub_models.models.templates.yolo.utils import (
    get_most_likely_score,
)
from qai_hub_models.models.templates.yolo.yolo_segmentation_evaluator import (
    YoloSegmentationOutputEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
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

DEFAULT_YOLO_IMAGE_INPUT_HW = 640


def yolo_detect_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post processing to break newer ultralytics yolo models (e.g. Yolov8, Yolo11) detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores and classes.

    Parameters
    ----------
    boxes
        Shape is [batch, 4, num_preds] where 4 == [x_center, y_center, w, h]
    scores
        Shape is [batch, num_classes, num_preds]
        Each element represents the probability that a given box is
            an instance of a given class.

    Returns
    -------
    boxes : torch.Tensor
        Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
    scores : torch.Tensor
        Class scores multiplied by confidence. Shape is [batch, num_preds].
    class_idx : torch.Tensor
        Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    boxes = torch.permute(boxes, [0, 2, 1])
    scores = torch.permute(scores, [0, 2, 1])

    # Convert boxes to (x1, y1, x2, y2)
    boxes = box_xywh_to_xyxy(boxes)

    # TODO(13933) Revert once QNN issues with ReduceMax are fixed
    if scores.shape[-1] == 1:
        scores = F.pad(scores, (0, 1))

    # Get class ID of most likely score.
    scores, class_idx = torch.max(scores, -1, keepdim=False)

    # Cast classes to int8 for imsdk compatibility
    return boxes, scores, class_idx.to(torch.uint8)


def yolo_segment_postprocess(
    detector_output: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post processing to break Yolo Segmentation output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, scores, masks and classes.

    Parameters
    ----------
    detector_output
        The output of Yolo Detection model
        Shape is [batch, k, num_preds]
            where, k = # of classes + 4
            k is structured as follows [boxes (4) : # of classes]
            and boxes are co-ordinates [x_center, y_center, w, h]
    num_classes
        number of classes

    Returns
    -------
    boxes : torch.Tensor
        Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
    scores : torch.Tensor
        Class scores multiplied by confidence. Shape is [batch, num_preds].
    masks : torch.Tensor
        Predicted masks. Shape is [batch, num_preds, 32].
    class_idx : torch.Tensor
        Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    # Break output into parts
    detector_output = torch.permute(detector_output, [0, 2, 1])
    masks_dim = 4 + num_classes
    boxes = detector_output[:, :, :4]
    scores = detector_output[:, :, 4:masks_dim]
    masks = detector_output[:, :, masks_dim:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = box_xywh_to_xyxy(boxes)

    # Get class ID of most likely score.
    scores, class_idx = get_most_likely_score(scores)

    return boxes, scores, masks, class_idx.to(torch.int32)


def yolo_obb_postprocess(
    boxes: torch.Tensor,
    angles: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post processing to break newer ultralytics yolo-obb models detector output
    into multiple, consumable tensors.

    Parameters
    ----------
    boxes
        Shape is [batch, 4, num_preds] where 4 == [x_center, y_center, w, h]
    angles
        Shape is [batch, 1, num_preds] (in radians)
    scores
        Shape is [batch, num_classes, num_preds]

    Returns
    -------
    boxes : torch.Tensor
        Bounding box locations.
        Shape is [batch, num_preds, 4] where 4 == (x_center, y_center, w, h).
    angles : torch.Tensor
        Box rotation angle in radians. Shape is [batch, num_preds, 1].
    scores : torch.Tensor
        Class scores multiplied by confidence. Shape is [batch, num_preds].
    class_idx : torch.Tensor
        Shape is [batch, num_preds] where the last dim is the index of the most probable class.
    """
    # 1. Permute to [Batch, Num_Preds, Channels]
    boxes = torch.permute(boxes, [0, 2, 1])
    angles = torch.permute(angles, [0, 2, 1])
    scores = torch.permute(scores, [0, 2, 1])

    # 2. TODO(13933) Revert once QNN issues with ReduceMax are fixed
    # Workaround: QNN ReduceMax fails on single-element last dim.
    # Safe because all supported OBB models have num_classes > 1.
    if scores.shape[-1] == 1:
        scores = torch.nn.functional.pad(scores, (0, 1))

    # 3. Get class ID of most likely score.
    # scores is [Batch, Num_Preds], class_idx is [Batch, Num_Preds]
    # Get class ID of most likely score.
    scores, class_idx = torch.max(scores, -1, keepdim=False)

    # Cast classes to int8 for imsdk compatibility
    return boxes, angles, scores, class_idx.to(torch.uint8)


class Yolo(BaseModel):
    # All image input spatial dimensions should be a multiple of this stride.
    STRIDE_MULTIPLE = 32

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(
            image_height, image_width, score_threshold=0.25, nms_iou_threshold=0.7
        )

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_YOLO_IMAGE_INPUT_HW,
        width: int = DEFAULT_YOLO_IMAGE_INPUT_HW,
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
            "class_idx": TensorSpec(
                io_type=IoType.TENSOR,
                labels_file="coco_labels.txt",
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "yolov7", 1, "yolov7_demo_640.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDataset

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("coco", output_dir, metadata)


class YoloSegEvalMixin(BaseModel):
    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return YoloSegmentationOutputEvaluator(image_height, image_width, 0.001, 0.7)

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoSegDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoSegDataset
