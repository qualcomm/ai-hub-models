# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN as maskrcnn
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import poolers
from typing_extensions import Self

from qai_hub_models.models.common import Precision
from qai_hub_models.models.maskrcnn.model_patches import _onnx_merge_levels_optimized
from qai_hub_models.utils.base_model import BaseModel, CollectionModel, TargetRuntime
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "DEFAULT"
DEFAULT_VERSION = 2
NUM_CLASSES = 91  # COCO has 91 classes (including background)

# optimized the model speed by replacing multiple scatter_elements
# with single gather node.
poolers._onnx_merge_levels = _onnx_merge_levels_optimized


class MaskRCNNProposalGenerator(BaseModel):
    """MaskRCNN Proposal Generator (Backbone + RPN)"""

    def __init__(self, model: maskrcnn) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.rpn = model.rpn

    @classmethod
    def from_pretrained(
        cls, version: int = DEFAULT_VERSION, weights: str = DEFAULT_WEIGHTS
    ) -> Self:
        if version == 1:
            model = maskrcnn_resnet50_fpn(weights=weights)
        elif version == 2:
            model = maskrcnn_resnet50_fpn_v2(weights=weights)
        else:
            raise ValueError(f"Invalid version: {version}. Must be 1 or 2.")
        return cls(model)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        image
            Pixel values pre-processed with shape (B, 3, H, W).
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        features_0: torch.Tensor
            Feature maps from FPN level 0 with shape (B, 256, H//4, W//4)
        features_1: torch.Tensor
            Feature maps from FPN level 1 with shape (B, 256, H//8, W//8)
        features_2: torch.Tensor
            Feature maps from FPN level 2 with shape (B, 256, H//16, W//16)
        features_3: torch.Tensor
            Feature maps from FPN level 3 with shape (B, 256, H//32, W//32)
        proposals: torch.Tensor
            The proposals for image, with shape (B, num_proposals, 4) in xyxy format.
        objectness_logits: torch.Tensor
            The objectness logits for image, with shape (B, num_proposals,)
        """
        # Normalize image (RGB [0, 1] -> normalized)
        image = normalize_image_torchvision(image)

        # Extract features from backbone (FPN)
        # https://github.com/pytorch/vision/blob/0f6d91d9fe514e6de2f5519114cbeb389d498b2d/torchvision/models/detection/generalized_rcnn.py#L114
        features_dict = self.backbone(image)
        features_list = list(features_dict.values())

        # Generate anchors and RPN predictions
        # https://github.com/pytorch/vision/blob/0f6d91d9fe514e6de2f5519114cbeb389d498b2d/torchvision/models/detection/rpn.py#L360
        batch_size = image.shape[0]
        image_size = (image.shape[-2], image.shape[-1])
        image_sizes = [image_size] * batch_size
        image_list = ImageList(image, image_sizes)
        anchors = self.rpn.anchor_generator(image_list, features_list)
        objectness, pred_bbox_deltas = self.rpn.head(features_list)

        # Flatten and concatenate predictions from all feature levels
        objectness_final = torch.cat(
            [o.permute(0, 2, 3, 1).flatten(1) for o in objectness], dim=1
        )
        pred_bbox_deltas_final = torch.cat(
            [
                b.permute(0, 2, 3, 1).reshape(b.shape[0], -1, 4)
                for b in pred_bbox_deltas
            ],
            dim=1,
        )

        proposals = self.rpn.box_coder.decode(
            pred_bbox_deltas_final, anchors
        ).transpose(1, 0)

        return (
            features_dict["0"],
            features_dict["1"],
            features_dict["2"],
            features_dict["3"],
            proposals,
            objectness_final,
        )

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 800,
        width: int = 800,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return [
            "features_0",
            "features_1",
            "features_2",
            "features_3",
            "proposals",
            "objectness_logits",
        ]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]


class MaskRCNNROIHead(BaseModel):
    """MaskRCNN ROI Head (Box Head + Mask Head)"""

    def __init__(self, model: maskrcnn) -> None:
        super().__init__()
        self.roi_heads = model.roi_heads

    @classmethod
    def from_pretrained(
        cls, version: int = DEFAULT_VERSION, weights: str = DEFAULT_WEIGHTS
    ) -> Self:
        if version == 1:
            model = maskrcnn_resnet50_fpn(weights=weights)
        elif version == 2:
            model = maskrcnn_resnet50_fpn_v2(weights=weights)
        else:
            raise ValueError(f"Invalid version: {version}. Must be 1 or 2.")
        return cls(model)

    def forward(
        self,
        features_0: torch.Tensor,
        features_1: torch.Tensor,
        features_2: torch.Tensor,
        features_3: torch.Tensor,
        proposals_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features_0
            Feature maps from FPN level 0 with shape (B, 256, H//4, W//4).
        features_1
            Feature maps from FPN level 1 with shape (B, 256, H//8, W//8).
        features_2
            Feature maps from FPN level 2 with shape (B, 256, H//16, W//16).
        features_3
            Feature maps from FPN level 3 with shape (B, 256, H//32, W//32).
        proposals_boxes
            The proposals for image, with shape (B, num_proposals, 4)
            in xyxy format in range[0,1].

        Returns
        -------
        boxes: torch.Tensor
            A tensor of shape (B, max_detections, 4) in xyxy format containing the predicted boxes.
        scores: torch.Tensor
            A tensor of shape (B, max_detections) containing the scores for each box.
        classes: torch.Tensor
            A tensor of shape (B, max_detections) containing the labels for each box.
        masks: torch.Tensor
            A tensor of shape (B, max_detections, mask_size, mask_size) containing the predicted masks.
            Typically mask_size is 28 for standard MaskRCNN.
        """
        batch_size = proposals_boxes.shape[0]

        # Reconstruct features_dict from FPN level inputs
        features_dict = {
            "0": features_0,
            "1": features_1,
            "2": features_2,
            "3": features_3,
        }

        # Infer image shape from features_0 (features are at 1/4 resolution of input)
        # Calculate the stride from feature map to input image
        feature_stride = 4  # FPN level 0 has stride of 4
        image_h, image_w = (
            features_0.shape[2] * feature_stride,
            features_0.shape[3] * feature_stride,
        )

        # Prepare proposals (scale to image coordinates)
        # proposals_boxes are normalized [0, 1], scale by actual image dimensions
        scale_factor = torch.tensor(
            [image_w, image_h, image_w, image_h],
            dtype=proposals_boxes.dtype,
            device=proposals_boxes.device,
        )
        proposals_boxes = proposals_boxes * scale_factor
        proposals_list = list(proposals_boxes)
        image_shapes = [(image_h, image_w)] * batch_size

        # Box ROI pooling and prediction
        # https://github.com/pytorch/vision/blob/0f6d91d9fe514e6de2f5519114cbeb389d498b2d/torchvision/models/detection/roi_heads.py#L772
        box_features = self.roi_heads.box_roi_pool(
            features_dict, proposals_list, image_shapes
        )
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        # Decode boxes and get scores
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals_list)
        pred_scores = F.softmax(class_logits, -1)

        # Process predictions per image
        num_proposals_per_img = [len(p) for p in proposals_list]
        pred_boxes_list = pred_boxes.split(num_proposals_per_img, 0)
        pred_scores_list = pred_scores.split(num_proposals_per_img, 0)

        batch_boxes = []
        batch_scores = []
        batch_labels = []

        for boxes_per_img, scores_per_img in zip(
            pred_boxes_list, pred_scores_list, strict=True
        ):
            # Get best class (excluding background)
            scores_no_bg = scores_per_img[:, 1:]
            scores_max, labels = torch.max(scores_no_bg, dim=1)
            labels = labels + 1

            # Select boxes for predicted class
            num_proposals = boxes_per_img.shape[0]
            boxes_per_img = boxes_per_img.reshape(num_proposals, -1, 4)
            boxes_per_img = boxes_per_img[
                torch.arange(num_proposals, device=boxes_per_img.device), labels
            ]

            # Clip to image boundaries
            boxes_per_img[:, 0::2].clamp_(min=0, max=image_w)
            boxes_per_img[:, 1::2].clamp_(min=0, max=image_h)

            batch_boxes.append(boxes_per_img)
            batch_scores.append(scores_max)
            batch_labels.append(labels)

        # Mask prediction
        batch_masks = []
        mask_features = self.roi_heads.mask_roi_pool(
            features_dict, batch_boxes, image_shapes
        )
        mask_logits = self.roi_heads.mask_head(mask_features)
        mask_logits = self.roi_heads.mask_predictor(mask_logits)

        for i in range(batch_size):
            image_masks = mask_logits[
                torch.arange(batch_labels[i].shape[0]), batch_labels[i]
            ]
            batch_masks.append(image_masks)

        # Stack results (proposals are already padded to fixed size)
        boxes_tensor = torch.stack(batch_boxes, dim=0)
        scores_tensor = torch.stack(batch_scores, dim=0)
        labels_tensor = torch.stack(batch_labels, dim=0)
        masks_tensor = torch.stack(batch_masks, dim=0)

        return boxes_tensor, scores_tensor, labels_tensor, masks_tensor

    @staticmethod
    def get_input_spec(
        height: int = 200, width: int = 200, num_boxes: int = 200
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {
            "features_0": ((1, 256, height, width), "float32"),
            "features_1": ((1, 256, height // 2, width // 2), "float32"),
            "features_2": ((1, 256, height // 4, width // 4), "float32"),
            "features_3": ((1, 256, height // 8, width // 8), "float32"),
            "proposals_boxes": ((1, num_boxes, 4), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "classes", "masks"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["features_0", "features_1", "features_2", "features_3"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Any = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors True --truncate_64bit_io True"

        return compile_options


@CollectionModel.add_component(MaskRCNNProposalGenerator)
@CollectionModel.add_component(MaskRCNNROIHead)
class MaskRCNN(CollectionModel):
    """MaskRCNN Instance Segmentation Model"""

    def __init__(
        self,
        proposal_generator: MaskRCNNProposalGenerator,
        roi_head: MaskRCNNROIHead,
    ) -> None:
        super().__init__(*[proposal_generator, roi_head])
        self.proposal_generator = proposal_generator
        self.roi_head = roi_head

    @classmethod
    def from_pretrained(
        cls, version: int = DEFAULT_VERSION, weights: str = DEFAULT_WEIGHTS
    ) -> Self:
        return cls(
            MaskRCNNProposalGenerator.from_pretrained(version, weights),
            MaskRCNNROIHead.from_pretrained(version, weights),
        )
