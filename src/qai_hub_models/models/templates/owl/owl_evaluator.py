# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Collection

import torch
from podm.metrics import BoundingBox

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.detection.detection_evaluator import mAPEvaluator
from qai_hub_models.utils.input_spec import InputSpec


class OwlDetectionEvaluator(mAPEvaluator):
    """mAP evaluator for Owl models."""

    def __init__(
        self,
        input_spec: InputSpec,
        score_threshold: float = 0.001,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        shape = input_spec["pixel_values"][0]
        self.image_height = int(shape[2])
        self.image_width = int(shape[3])
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # Scale factors to normalise predicted pixel-space boxes to [0, 1]
        self.scale_x = 1.0 / self.image_width
        self.scale_y = 1.0 / self.image_height

    def add_batch(
        self,
        output: Collection[torch.Tensor],
        gt: Collection[torch.Tensor],
    ) -> None:
        """
        Process one batch of Owl predictions and ground-truth annotations.

        Parameters
        ----------
        output
            boxes   float32  [B, P, 4]  - (x1, y1, x2, y2) in model pixel space
            scores  float32  [B, P]     - sigmoid confidence scores
            labels  int32    [B, P]     - query indices (0 for single-query samples)

        gt
            image_ids  [B]
            heights    [B]
            widths     [B]
            boxes      [B, max_boxes, 4]  - (x1, y1, x2, y2) normalised [0, 1]
            labels     [B, max_boxes]     - 0-indexed 80-class labels
            num_boxes  [B, 1]
        """
        # Unpack postprocessed model output
        pred_boxes_xyxy, pred_scores, _pred_query_labels = output
        image_ids, _heights, _widths, all_gt_boxes, all_gt_labels, all_num_boxes = gt

        batch_size = pred_boxes_xyxy.shape[0]

        for b in range(batch_size):
            image_id = int(image_ids[b].item())
            num_valid = int(all_num_boxes[b].item())
            gt_boxes = all_gt_boxes[b][:num_valid]  # [num_valid, 4] normalised [0,1]
            gt_labels = all_gt_labels[b][:num_valid]  # [num_valid]

            # Predictions for this sample
            boxes = pred_boxes_xyxy[b]  # [P, 4] in model pixel space
            scores = pred_scores[b]  # [P]

            actual_class_idx = int(gt_labels[0].item()) if num_valid > 0 else 0
            labels = torch.full((boxes.shape[0],), actual_class_idx, dtype=torch.long)

            # Score threshold  →  remove low-confidence patch detections
            mask = scores > self.score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Per-class NMS  →  remove duplicate detections
            if len(boxes) > 0:
                nms_boxes_list, nms_scores_list, nms_classes_list = batched_nms(
                    self.nms_iou_threshold,
                    self.score_threshold,
                    boxes.unsqueeze(0),  # [1, N, 4]
                    scores.unsqueeze(0),  # [1, N]
                    labels.unsqueeze(0),  # [1, N]
                )
                nms_boxes = nms_boxes_list[0]  # [M, 4]
                nms_scores = nms_scores_list[0]  # [M]
                nms_classes = nms_classes_list[0]  # [M]
            else:
                nms_boxes = torch.zeros((0, 4))
                nms_scores = torch.zeros(0)
                nms_classes = torch.zeros(0, dtype=torch.long)

            gt_bb = [
                BoundingBox.of_bbox(
                    image_id,
                    int(cls),
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    1.0,
                )
                for cls, box in zip(gt_labels.tolist(), gt_boxes.tolist(), strict=False)
            ]

            pd_bb = [
                BoundingBox.of_bbox(
                    image_id,
                    int(cls),
                    float(box[0]) * self.scale_x,
                    float(box[1]) * self.scale_y,
                    float(box[2]) * self.scale_x,
                    float(box[3]) * self.scale_y,
                    float(score),
                )
                for cls, score, box in zip(
                    nms_classes.tolist(),
                    nms_scores.tolist(),
                    nms_boxes.tolist(),
                    strict=False,
                )
                if not any(math.isnan(v) for v in box)
            ]

            self.store_bboxes_for_eval(gt_bb, pd_bb)
