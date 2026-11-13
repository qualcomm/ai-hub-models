# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from qai_hub.client import Device
from transformers import AutoProcessor

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.datasets.coco.coco_owl import CocoOwlDataset
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.owl.owl_evaluator import (
    OwlDetectionEvaluator,
)
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


def owl_postprocess(
    logits: torch.Tensor, boxes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Postprocess the output of the Owl model.

    Parameters
    ----------
    logits
        Shape (B, num_patches, # of Queries) classification logits
    boxes
        Shape (B, num_patches, 4) bounding box coordinates of shape (cx, cy, w, h)

    Returns
    -------
    boxes : torch.Tensor
        Shape (B, num_patches, 4) representing the bounding box coordinates (x1, y1, x2, y2) in [0-1].
    scores : torch.Tensor
        Shape (B, num_patches) representing the confidence scores.
    labels : torch.Tensor
        Shape (B, num_patches) representing the class labels.
    """
    # Classification logits include no-object for all queries.
    # Remove the "no-object" and get the max of the remaining logits.
    batch_class_logits = torch.max(logits, dim=-1)

    scores = torch.sigmoid(batch_class_logits.values)
    labels = batch_class_logits.indices

    # Convert (cx, cy, w, h) → (x1, y1, x2, y2), still in [0, 1] normalized space
    boxes = box_xywh_to_xyxy(boxes)

    return boxes, scores, labels


class Owl(BaseModel):
    """Owl model open-vocabulary object detector."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pixel_values
            Preprocessed image. Shape: [B, 3,   H, W]
            Range: float[0, 1]
            3-channel Color Space: RGB
        input_ids
            Tokenized text queries. Shape: [B, 16]
        attention_mask
            Attention mask for text queries. Shape: [B, 16]

        Returns
        -------
        boxes : torch.Tensor
            Shape (B, num_patches, 4) representing the bounding box coordinates (x1, y1, x2, y2) in [0-1].
        scores : torch.Tensor
            Shape (B, num_patches) representing the confidence scores.
        labels : torch.Tensor
            Shape (B, num_patches) representing the class labels.
        """
        out = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            return_dict=True,
        )

        boxes, scores, labels = owl_postprocess(out.logits, out.pred_boxes)
        return boxes, scores, labels

    def get_output_spec(self) -> OutputSpec:
        return Owl._get_output_spec()

    @staticmethod
    def _get_output_spec() -> OutputSpec:
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
            "labels": TensorSpec(
                io_type=IoType.TENSOR,
                labels_file="coco_labels.txt",
            ),
        }

    @staticmethod
    def _get_input_spec(
        batch_size: int = 1,
        image_height: int = 768,
        image_width: int = 768,
        text_seq_len: int = 16,
    ) -> InputSpec:
        return {
            "pixel_values": TensorSpec(
                shape=(batch_size, 3, image_height, image_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
            "input_ids": TensorSpec(
                shape=(batch_size, text_seq_len),
                dtype="int32",
            ),
            "attention_mask": TensorSpec(
                shape=(batch_size, text_seq_len),
                dtype="int32",
            ),
        }

    @classmethod
    def get_dataset_class(
        cls, hf_model_id: str, input_spec: InputSpec, norm_by_max: bool
    ) -> type[CocoOwlDataset]:
        class BaseCocoOwlDataset(CocoOwlDataset):
            def __init__(self, **kwargs: Any) -> None:
                processor = AutoProcessor.from_pretrained(hf_model_id, use_fast=True)
                super().__init__(
                    processor=processor,
                    input_spec=input_spec,
                    norm_by_max=norm_by_max,
                    **kwargs,
                )

        return BaseCocoOwlDataset

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        return Owl._get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
            self,
        )

    @staticmethod
    def _get_hub_compile_options(
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str,
        device: Device | None,
        context_graph_name: str | None,
        base_instance: BaseModel,
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return BaseModel.get_hub_compile_options(
            base_instance,
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )

    @classmethod
    def detection_evaluator(cls, input_spec: InputSpec) -> BaseEvaluator:
        return OwlDetectionEvaluator(input_spec=input_spec)
