# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import DetrForObjectDetection, PreTrainedModel
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.coco import Coco91ClassDataset
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
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


class DETR(BaseModel):
    DEFAULT_WEIGHTS: str | None = None
    HF_DETR_CLS: type[PreTrainedModel] = DetrForObjectDetection

    def get_evaluator(self) -> BaseEvaluator:
        """
        Returns an instance of the DetectionEvaluator class, which is used to evaluate the performance of the DETR model.

        The DetectionEvaluator class is used to compute the mean average precision (mAP) of the model's predictions.

        Returns
        -------
        evaluator : BaseEvaluator
            An instance of the DetectionEvaluator class
        """
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return DetectionEvaluator(image_height, image_width, score_threshold=0.5)

    def detr_postprocess(
        self, logits: torch.Tensor, boxes: torch.Tensor, image_shape: tuple
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Postprocess the output of the DETR model.

        Parameters
        ----------
        logits
            Shape (B, 100, # of Classes) classification logits
        boxes
            Shape (B, 100, 4) bounding box coordinates of shape (x, y, w, h)
        image_shape
            The shape of the input image -- (x, y)

        Returns
        -------
        boxes : torch.Tensor
            Shape (B, 100, 4) representing the bounding box coordinates (x1, y1, x2, y2).
        scores : torch.Tensor
            Shape (B, 100) representing the confidence scores.
        labels : torch.Tensor
            Shape (B, 100) representing the class labels.
        """
        _, _, h, w = image_shape

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, -1)

        # Classification logits include no-object for all queries.
        # Remove the "no-object" and get the max of the remaining logits.
        scores, labels = probabilities[..., :-1].max(-1)

        # Convert to [x0, y0, x1, y1] format
        boxes = box_xywh_to_xyxy(boxes)

        # Convert to pixel space
        boxes *= torch.Tensor([w, h, w, h])

        # Cast output tensors to float32 and supported by Qualcomm AI Hub
        boxes = boxes.to(torch.float32)
        scores = scores.to(torch.float32)
        labels = labels.to(torch.int32)

        return boxes, scores, labels

    @classmethod
    def from_pretrained(cls, ckpt_name: str = "") -> Self:
        if not cls.DEFAULT_WEIGHTS:
            raise NotImplementedError(
                f"DEFAULT_WEIGHTS is not set for this DETR subclass ({cls.__name__}). The default from_pretrained implementation cannot be used."
            )
        model = cls.HF_DETR_CLS.from_pretrained(ckpt_name or cls.DEFAULT_WEIGHTS)
        return cls(model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run DETR on `image` and produce high quality detection results.

        Parameters
        ----------
        image
            Image tensor to run detection on. RGB, Range[0-1].

        Returns
        -------
        boxes : torch.Tensor
            Shape (B, 100, 4) representing the bounding box coordinates (x1, y1, x2, y2).
        scores : torch.Tensor
            Shape (B, 100) representing the confidence scores.
        labels : torch.Tensor
            Shape (B, 100) representing the class labels.
        """
        image_array = normalize_image_torchvision(image)
        # boxes: (center_x, center_y, w, h)
        predictions = self.model(image_array)
        logits, boxes = predictions[0], predictions[1]
        boxes, scores, labels = self.detr_postprocess(logits, boxes, image_array.shape)

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
            "logits": TensorSpec(io_type=IoType.TENSOR),
            "classes": TensorSpec(io_type=IoType.TENSOR),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "detr_resnet50", 1, "detr_demo_image.jpg"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [Coco91ClassDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Coco91ClassDataset
