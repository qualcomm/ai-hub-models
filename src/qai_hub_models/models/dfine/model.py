# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import DFineForObjectDetection
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.coco import CocoDataset
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.detection.detection_evaluator import (
    DetectionEvaluator,
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

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo_image.jpg"
)

# All variants in the D-FINE collection (https://huggingface.co/collections/ustc-community/d-fine).
# Maps a short variant name (exposed on the CLI via --variant) to its HuggingFace repo id.
DFINE_VARIANTS: dict[str, str] = {
    # Trained on COCO.
    "nano-coco": "ustc-community/dfine-nano-coco",
    "small-coco": "ustc-community/dfine-small-coco",
    "medium-coco": "ustc-community/dfine-medium-coco",
    "large-coco": "ustc-community/dfine-large-coco",
    "xlarge-coco": "ustc-community/dfine-xlarge-coco",
    # Trained on Object365.
    "small-obj365": "ustc-community/dfine-small-obj365",
    "medium-obj365": "ustc-community/dfine-medium-obj365",
    "large-obj365": "ustc-community/dfine-large-obj365",
    "xlarge-obj365": "ustc-community/dfine-xlarge-obj365",
    # Pretrained on Object365 -> trained on COCO.
    "small-obj2coco": "ustc-community/dfine-small-obj2coco",
    "medium-obj2coco": "ustc-community/dfine-medium-obj2coco",
    "large-obj2coco": "ustc-community/dfine-large-obj2coco-e25",
    "xlarge-obj2coco": "ustc-community/dfine-xlarge-obj2coco",
}

DEFAULT_VARIANT = "nano-coco"


class DFine(BaseModel):
    """D-FINE real-time object detection model."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    @classmethod
    def from_pretrained(cls, variant: str = DEFAULT_VARIANT) -> Self:
        if variant not in DFINE_VARIANTS:
            raise ValueError(
                f"Unknown D-FINE variant '{variant}'. "
                f"Valid variants: {', '.join(DFINE_VARIANTS)}"
            )
        model = DFineForObjectDetection.from_pretrained(DFINE_VARIANTS[variant])
        return cls(model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run D-FINE on `image` and produce object detection results.

        Parameters
        ----------
        image
            Image tensor to run detection on. Shape (B, 3, H, W). RGB, range [0, 1].

        Returns
        -------
        boxes : torch.Tensor
            Shape (B, num_queries, 4) bounding box coordinates (x1, y1, x2, y2)
            in pixel space.
        scores : torch.Tensor
            Shape (B, num_queries) confidence scores.
        labels : torch.Tensor
            Shape (B, num_queries) class labels.
        """
        _, _, h, w = image.shape

        predictions = self.model(image)
        logits, boxes = predictions[0], predictions[1]

        scores, labels = torch.sigmoid(logits).max(-1)

        # boxes are normalized (center_x, center_y, w, h); convert to pixel xyxy.
        boxes = box_xywh_to_xyxy(boxes)
        boxes = boxes * torch.tensor(
            [w, h, w, h], dtype=boxes.dtype, device=boxes.device
        )

        boxes = boxes.to(torch.float32)
        scores = scores.to(torch.float32)
        labels = labels.to(torch.int32)
        return boxes, scores, labels

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 640,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub Workbench.
        """
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
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "boxes": TensorSpec(
                io_type=IoType.BBOX,
                bbox_metadata=BboxMetadata(bbox_format=BboxFormat.XYXY),
            ),
            "scores": TensorSpec(io_type=IoType.TENSOR),
            "classes": TensorSpec(io_type=IoType.TENSOR),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDataset

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, score_threshold=0.3)
