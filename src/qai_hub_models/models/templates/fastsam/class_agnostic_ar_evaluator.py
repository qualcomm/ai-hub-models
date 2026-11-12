# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torchmetrics.detection import MeanAveragePrecision

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    AVERAGE_RECALL_1000,
    MetricMetadata,
)


class ClassAgnosticARkEvaluator(BaseEvaluator):
    """Class-agnostic box Average Recall @ k evaluator."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_score_threshold: float = 0.001,
        nms_iou_threshold: float = 0.7,
        max_detections_per_image: int = 1000,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_image = max_detections_per_image
        self.reset()

    def reset(self) -> None:
        self.metric = MeanAveragePrecision(
            iou_type="bbox",
            max_detection_thresholds=[1, 10, self.max_detections_per_image],
            class_metrics=False,
        )

    def add_batch(
        self,
        output: tuple[torch.Tensor, ...],
        gt: tuple[torch.Tensor, ...],
    ) -> None:
        """
        Parameters
        ----------
        output
            First two elements must be:
                pred_boxes  (batch, num_preds, 4) — pixel coords (x1,y1,x2,y2)
                            of the resized/padded model-input frame.
                pred_scores (batch, num_preds).
            Any further tensors (e.g. mask coefficients, prototypes) are
            ignored.

        gt
            Tuple matching ``CocoDataset.__getitem__`` output:
                image_id, target_h, target_w,
                gt_boxes  (batch, max_boxes, 4) — normalized [0, 1] in the
                          resized/padded model-input frame, format (x1,y1,x2,y2),
                gt_labels (batch, max_boxes),
                num_boxes (batch,).
        """
        pred_boxes, pred_scores = output[0], output[1]
        _, _, _, gt_boxes, _, num_boxes = gt

        # Class-agnostic NMS — every proposal is treated as the same class.
        nms_boxes, nms_scores = batched_nms(
            self.nms_iou_threshold,
            self.nms_score_threshold,
            pred_boxes.float(),
            pred_scores.float(),
        )

        for i in range(len(nms_boxes)):
            order = torch.argsort(nms_scores[i], descending=True)
            top = order[: self.max_detections_per_image]
            preds_boxes_i = nms_boxes[i][top]
            preds_scores_i = nms_scores[i][top]

            n = int(num_boxes[i].item())
            gt_boxes_i = gt_boxes[i][:n].float()
            # Convert GT from normalized [0, 1] to pixel coordinates of the
            # resized/padded model-input frame so prediction and GT live in
            # the same coordinate space.
            gt_boxes_pixels = gt_boxes_i.clone()
            gt_boxes_pixels[:, 0::2] *= self.image_width
            gt_boxes_pixels[:, 1::2] *= self.image_height

            self.metric.update(
                [
                    {
                        "boxes": preds_boxes_i,
                        "scores": preds_scores_i,
                        "labels": torch.zeros(
                            preds_boxes_i.shape[0],
                            dtype=torch.long,
                        ),
                    }
                ],
                [
                    {
                        "boxes": gt_boxes_pixels,
                        "labels": torch.zeros(n, dtype=torch.long),
                    }
                ],
            )

    def get_accuracy_score(self) -> float:
        results = self.metric.compute()
        return float(results[f"mar_{self.max_detections_per_image}"]) * 100

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.3f} AR@{self.max_detections_per_image}"

    def get_metric_metadata(self) -> MetricMetadata:
        return AVERAGE_RECALL_1000
