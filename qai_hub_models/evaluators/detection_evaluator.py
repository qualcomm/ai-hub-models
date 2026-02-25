# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Collection

import torch

# podm comes from the object-detection-metrics pip package
from podm.metrics import BoundingBox, MetricPerClass, get_pascal_voc_metrics

from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata
from qai_hub_models.utils.bounding_box_processing import batched_nms


class mAPEvaluator(BaseEvaluator):
    """Evaluator that calculates mAP given stored bounding boxes."""

    DEFAULT_LOW_IOU = 0.5
    DEFAULT_HIGH_IOU = 0.95
    DEFAULT_INCREMENT_IOU = 0.05

    def __init__(
        self,
        mAP_default_low_iOU: float | None = None,
        mAP_default_high_iOU: float | None = None,
        mAP_default_increment_iOU: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        mAP_default_low_iOU
            The default low iOU (inclusive) of the range for which average mAP should be calculated.
        mAP_default_high_iOU
            The default high iOU (inclusive) of the range for which average mAP should be calculated.
        mAP_default_increment_iOU
            The default iOU increment for which average mAP should be calculated.
        """
        self.mAP_default_low_iOU = (
            mAP_default_low_iOU
            if mAP_default_low_iOU is not None
            else mAPEvaluator.DEFAULT_LOW_IOU
        )
        self.mAP_default_high_iOU = (
            mAP_default_high_iOU
            if mAP_default_high_iOU is not None
            else mAPEvaluator.DEFAULT_HIGH_IOU
        )
        self.mAP_default_increment_iOU = (
            mAP_default_increment_iOU
            if mAP_default_increment_iOU is not None
            else mAPEvaluator.DEFAULT_INCREMENT_IOU
        )
        self.reset()

    def reset(self) -> None:
        self.gt_bbox: list[BoundingBox] = []
        self.pred_bbox: list[BoundingBox] = []

    def store_bboxes_for_eval(
        self, gt_bbox: list[BoundingBox], pred_bbox: list[BoundingBox]
    ) -> None:
        """
        Save the given bounding boxes for evaluation (mAP calculation) to be completed later.
        The boxes will be used when a user calls get_mAP().

        Parameters
        ----------
        gt_bbox
            Ground truth bounding boxes. Expected box format is
                x1, y1, x2, y2, in pixel space.
        pred_bbox
            Predicted bounding boxes. Expected box format is
                x1, y1, x2, y2, in pixel space.
        """
        self.gt_bbox += gt_bbox
        self.pred_bbox += pred_bbox

    def get_mAP_for_iOU(self, iOU: float) -> float:
        return MetricPerClass.mAP(
            get_pascal_voc_metrics(self.gt_bbox, self.pred_bbox, iOU)
        )

    def get_mAP(
        self,
        low_iOU: float | None = None,
        high_iOU: float | None = None,
        increment_iOU: float | None = None,
    ) -> tuple[float, list[tuple[float, float]], float, float, float]:
        """
        Get mAP averaged over the given iOU range.

        Parameters
        ----------
        low_iOU
            Bottom of iOU range (inclusive). Default is self.mAP_default_low_iOU
        high_iOU
            Top of iOU range (inclusive). Default is self.mAP_default_high_iOU
        increment_iOU
            iOU increments at which to calculate mAP. Default is self.mAP_default_increment_iOU

        Returns
        -------
        mAP : float
            mAP averaged over the given iOU range with the given increment.
        mAP_by_iOU : list[tuple[float, float]]
            mAP calculated for each increment iOU in the range.
        low_iOU : float
            Bottom of iOU range (inclusive).
        high_iOU : float
            Top of iOU range (inclusive).
        increment_iOU : float
            iOU increments at which to calculate mAP.
        """
        low_iOU = low_iOU if low_iOU is not None else self.mAP_default_low_iOU
        high_iOU = high_iOU if high_iOU is not None else self.mAP_default_high_iOU
        increment_iOU = (
            increment_iOU
            if increment_iOU is not None
            else self.mAP_default_increment_iOU
        )

        mAP_by_iOU: list[tuple[float, float]] = []
        iOU = low_iOU
        while iOU <= high_iOU:
            iOU_mAP = self.get_mAP_for_iOU(iOU)
            mAP_by_iOU.append((iOU, iOU_mAP))
            iOU += increment_iOU

        return (
            (sum(x[1] for x in mAP_by_iOU) / len(mAP_by_iOU)) * 100,
            mAP_by_iOU,
            low_iOU,
            high_iOU,
            increment_iOU,
        )

    def get_accuracy_score(
        self,
        low_iOU: float | None = None,
        high_iOU: float | None = None,
        increment_iOU: float | None = None,
    ) -> float:
        """
        Get mAP averaged over the given iOU range.

        Parameters
        ----------
        low_iOU
            Bottom of iOU range (inclusive). Default is self.mAP_default_low_iOU
        high_iOU
            Top of iOU range (inclusive). Default is self.mAP_default_high_iOU
        increment_iOU
            iOU increments at which to calculate mAP. Default is self.mAP_default_increment_iOU

        Returns
        -------
        mAP : float
            mAP averaged over the given iOU range.
        """
        return self.get_mAP(low_iOU, high_iOU, increment_iOU)[0]

    def formatted_accuracy(
        self,
        low_iOU: float | None = None,
        high_iOU: float | None = None,
        increment_iOU: float | None = None,
    ) -> str:
        """
        Get mAP averaged over the given iOU range.

        Parameters
        ----------
        low_iOU
            Bottom of iOU range (inclusive). Default is self.mAP_default_low_iOU
        high_iOU
            Top of iOU range (inclusive). Default is self.mAP_default_high_iOU
        increment_iOU
            iOU increments at which to calculate mAP. Default is self.mAP_default_increment_iOU

        Returns
        -------
        formatted_mAP : str
            Formatted mAP string.
        """
        mAP, _, low_iOU, high_iOU, increment_iOU = self.get_mAP(
            low_iOU, high_iOU, increment_iOU
        )
        return f"{mAP:.3f} mAP@{low_iOU:.2f}" + (
            f":{high_iOU:.2f}" if high_iOU != low_iOU else ""
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Mean Average Precision",
            unit="mAP@0.5:0.95",
            description="Mean Average Precision averaged over IOU thresholds 0.5 to 0.95 in 0.05 increments.",
            range=(0.0, 100.0),
            float_vs_device_threshold=10.0,
        )


class DetectionEvaluator(mAPEvaluator):
    """Generic evaluator for detection tasks."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        nms_iou_threshold: float | None = None,
        score_threshold: float | None = None,
        mAP_default_low_iOU: float | None = None,
        mAP_default_high_iOU: float | None = None,
        mAP_default_increment_iOU: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        image_height
            Model input image height.
        image_width
            Model input image width.
        nms_iou_threshold
            If set, class-dependent non-maximum-suppression is applied
            to the model's output boxes with this iou threshold.
        score_threshold
            If set, all detections below the given score are discarded.
        mAP_default_low_iOU
            The default low iOU (inclusive) of the range for which average mAP should be calculated.
        mAP_default_high_iOU
            The default high iOU (inclusive) of the range for which average mAP should be calculated.
        mAP_default_increment_iOU
            The default iOU increment for which average mAP should be calculated.
        """
        super().__init__(
            mAP_default_low_iOU, mAP_default_high_iOU, mAP_default_increment_iOU
        )
        self.reset()
        self.scale_x = 1 / image_width
        self.scale_y = 1 / image_height
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold

    def add_batch(
        self, output: Collection[torch.Tensor], gt: Collection[torch.Tensor]
    ) -> None:
        """
        Parameters
        ----------
        output
            pred_boxes
                Shape (batch_size, num_candidate_boxes, 4) where 4 order is (x1, y1, x2, y2).
            pred_scores
                Shape (batch_size, num_candidate_boxes).
            pred_class_idx
                Shape (batch_size, num_candidate_boxes).
        gt
            image_ids
                Shape (batch_size,).
            image_heights
                Shape (batch_size,).
            image_widths
                Shape (batch_size,).
            bounding_boxes
                Shape (batch_size, max_boxes, 4) where 4 order is (x1, y1, x2, y2).
            classes
                Shape (batch_size, max_boxes).
            num_boxes
                Number of nonzero boxes for each sample, shape (batch_size,).
        """
        image_ids, _, _, all_bboxes, all_classes, all_num_boxes = gt
        pred_boxes, pred_scores, pred_class_idx = output

        if self.nms_iou_threshold is not None:
            (
                pred_nms_boxes,
                pred_nms_scores,
                pred_nms_class_idx,
            ) = batched_nms(
                self.nms_iou_threshold,
                self.score_threshold,
                pred_boxes.float(),
                pred_scores.float(),
                pred_class_idx,
            )
        else:
            pred_nms_boxes = list(pred_boxes)
            pred_nms_scores = list(pred_scores)
            pred_nms_class_idx = list(pred_class_idx)

        for i in range(len(image_ids)):
            image_id = image_ids[i]
            bboxes = all_bboxes[i][: int(all_num_boxes[i].item())]
            classes = all_classes[i][: int(all_num_boxes[i].item())]
            curr_pred_box = pred_nms_boxes[i : i + 1]
            curr_pred_score = pred_nms_scores[i : i + 1]
            curr_pred_class = pred_nms_class_idx[i : i + 1]

            # Collect GT and prediction boxes
            gt_bb_entry = [
                BoundingBox.of_bbox(
                    image_id, cat, bbox[0], bbox[1], bbox[2], bbox[3], 1.0
                )
                for cat, bbox in zip(classes.tolist(), bboxes.tolist(), strict=False)
            ]

            pd_bb_entry = [
                BoundingBox.of_bbox(
                    image_id,
                    pred_cat,
                    float(pred_bbox[0]) * self.scale_x,
                    float(pred_bbox[1]) * self.scale_y,
                    float(pred_bbox[2]) * self.scale_x,
                    float(pred_bbox[3]) * self.scale_y,
                    float(pred_score),
                )
                for pred_cat, pred_score, pred_bbox in zip(
                    curr_pred_class[0].tolist(),
                    curr_pred_score[0].tolist(),
                    curr_pred_box[0].tolist(),
                    strict=False,
                )
                if not any(math.isnan(x) for x in pred_bbox)
            ]

            if self.nms_iou_threshold is None and self.score_threshold is not None:
                # Create a new list that includes only the predictions with scores above the threshold
                pd_bb_entry = [b for b in pd_bb_entry if b.score > self.score_threshold]

            self.store_bboxes_for_eval(gt_bb_entry, pd_bb_entry)
