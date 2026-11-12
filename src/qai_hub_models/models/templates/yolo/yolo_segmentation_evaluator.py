# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torchmetrics.detection import MeanAveragePrecision
from ultralytics.utils.ops import process_mask

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    MEAN_AVERAGE_PRECISION_IOU_5_95,
    MetricMetadata,
)


class YoloSegmentationOutputEvaluator(BaseEvaluator):
    """Evaluator for comparing Instance Segmentation output against ground truth."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_score_threshold: float = 0.001,
        nms_iou_threshold: float = 0.7,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.reset()

    def reset(self) -> None:
        self.metric = MeanAveragePrecision(iou_type="segm")

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor) -> None:
        """
        Add a batch of predictions and ground truth for evaluation.

        Parameters
        ----------
        output
            pred_boxes
                Shape (batch_size, num_preds, 4) in normalized (x, y, w, h) format.
            pred_scores
                Shape (batch_size, num_preds).
            pred_masks
                Shape [batch_size, num_preds, 32].
            pred_class_idx
                Shape (batch_size, num_preds).
            protos
                Shape [batch_size, 32, mask_h, mask_w]. Multiply masks and protos to generate output masks.
        gt
            gt_mask
                Shape (batch_size, max_boxes, height, width).
            gt_classes
                Shape (batch_size, max_boxes).
            num_boxes
                Number of nonzero boxes for each sample, shape (batch_size,).
        """
        pred_boxes, pred_scores, pred_masks, pred_class_idx, proto = output
        gt_mask, gt_label, num_box = gt

        # Non Maximum Suppression on each batch
        pred_boxes, pred_scores, pred_class_idx, pred_masks = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes,
            pred_scores,
            pred_class_idx,
            pred_masks,
        )

        for batch_idx in range(len(pred_masks)):
            # Process mask and upsample to input shape
            pred_mask_single = process_mask(
                proto[batch_idx],
                pred_masks[batch_idx],
                pred_boxes[batch_idx],
                (self.image_height, self.image_width),
                upsample=True,
            ).to(torch.uint8)

            self.metric.update(
                [
                    {
                        "masks": pred_mask_single,
                        "scores": pred_scores[batch_idx],
                        "labels": pred_class_idx[batch_idx].to(torch.uint8),
                    }
                ],
                [
                    {
                        "masks": gt_mask[batch_idx][: num_box[batch_idx]],
                        "labels": gt_label[batch_idx][: num_box[batch_idx]],
                    }
                ],
            )

    def get_accuracy_score(self) -> float:
        # Calculate mAP_mask[0.5-0.95]
        results = self.metric.compute()
        return results["map"] * 100

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} mAP"

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_AVERAGE_PRECISION_IOU_5_95
