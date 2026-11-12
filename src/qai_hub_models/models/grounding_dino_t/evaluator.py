# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection
from typing import cast

import torch

from qai_hub_models.models.templates.detection.coco_detection_evaluator import (
    COCODetectionEvaluator,
)
from qai_hub_models.utils.image_processing import letterbox_geometry

# Matches GroundingDINO's own COCO eval pipeline (demo/test_ap_on_coco.py).
# Caps predicted boxes post-inference -- distinct from dataset.py's max_boxes,
# which caps ground-truth boxes per sample.
TOP_K = 300


class GroundingDINOEvaluator(COCODetectionEvaluator):
    def __init__(
        self,
        positive_map: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> None:
        """
        Parameters
        ----------
        positive_map
            (num_classes, MAX_TEXT_LEN) float32 tensor.
            Maps text logits to per-class scores.
            Build via build_positive_map(tokenizer) in dataset.py.
        image_height
            Model input height.
        image_width
            Model input width.
        """
        # Upstream's PostProcessCocoGrounding (demo/test_ap_on_coco.py) applies no
        # score threshold and no NMS before handing predictions to COCOeval, so
        # neither is applied here either.
        super().__init__(
            image_height=image_height,
            image_width=image_width,
            nms_iou_threshold=None,
            score_threshold=None,
        )
        self.positive_map = positive_map  # (num_classes, MAX_TEXT_LEN)
        self._scale = torch.tensor(
            [self.image_width, self.image_height, self.image_width, self.image_height],
            dtype=torch.float32,
        )

    def _letterbox_geometry(
        self, orig_h: float, orig_w: float
    ) -> tuple[float, float, float]:
        """Match GroundingDINO's own rounded-resize + floor-divided-padding convention."""
        scale, _new_h, _new_w, pad_top, pad_left = letterbox_geometry(
            orig_h, orig_w, self.image_height, self.image_width
        )
        return scale, float(pad_left), float(pad_top)

    def add_batch(
        self,
        output: torch.Tensor
        | torch.NumberType
        | Collection[torch.Tensor | torch.NumberType],
        gt: torch.Tensor
        | torch.NumberType
        | Collection[torch.Tensor | torch.NumberType],
    ) -> None:
        """
        Parameters
        ----------
        output
            pred_logits
                Shape (batch_size, num_queries, MAX_TEXT_LEN). Raw text token logits.
            pred_boxes
                Shape (batch_size, num_queries, 4). Normalized cxcywh in [0, 1].
        gt
            image_ids
                Shape (batch_size,).
            heights
                Shape (batch_size,).
            widths
                Shape (batch_size,).
            boxes
                Shape (batch_size, max_boxes, 4). Normalized xyxy in [0, 1].
            labels
                Shape (batch_size, max_boxes). 0-indexed num_classes-class labels.
            num_boxes
                Shape (batch_size, 1).
        """
        output = cast(Collection[torch.Tensor], output)
        pred_logits, pred_boxes = output

        prob = pred_logits.float().sigmoid()
        pm = self.positive_map.to(prob.device)
        scores_per_class = prob @ pm.T

        # Upstream top-k flatten: expand all (query, class) pairs and pick
        # top-TOP_K by score, matching GroundingDINO's own COCO eval pipeline.
        # This allows multiple class hypotheses per query box, unlike .max().
        batch_size = pred_logits.shape[0]

        cx, cy, w, h = (
            pred_boxes[..., 0],
            pred_boxes[..., 1],
            pred_boxes[..., 2],
            pred_boxes[..., 3],
        )
        boxes_xyxy = torch.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1
        )

        scale = self._scale.to(boxes_xyxy.device)
        pixel_boxes = boxes_xyxy * scale  # → pixel space, (B, num_queries, 4)

        # Flatten scores across all classes: (B, num_queries*num_classes) → top-TOP_K
        scores_flat = scores_per_class.flatten(1)  # (B, num_queries*num_classes)
        topk_scores, topk_idx = scores_flat.topk(
            min(TOP_K, scores_flat.shape[1]), dim=1
        )  # (B, 300)
        topk_query_idx = topk_idx // scores_per_class.shape[2]  # which query
        topk_class_idx = topk_idx % scores_per_class.shape[2]  # which class

        topk_boxes = torch.stack(
            [pixel_boxes[b][topk_query_idx[b]] for b in range(batch_size)]
        )  # (B, TOP_K, 4) pixel space (model input), xyxy

        # Unpad/unscale/clip to original image coordinates, remap class index to
        # COCO category_id, and accumulate -- all handled by the base class, using
        # our _letterbox_geometry override for GroundingDINO's rounded-padding convention.
        super().add_batch((topk_boxes, topk_scores, topk_class_idx), gt)
