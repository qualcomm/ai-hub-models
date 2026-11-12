# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import torch

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
    box_xywh_to_xyxy,
    get_bbox_iou_matrix,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MEAN_AVERAGE_PRECISION_IOU_50, MetricMetadata

# Number of confidence thresholds used to sweep the precision-recall curve.
# Matches upstream widerface_evaluate/evaluation.py — more steps = finer AP estimate.
_THRESH_NUM = 1000


def _image_eval(
    pred: np.ndarray,
    gt_xywh: np.ndarray,
    ignores: tuple[np.ndarray, np.ndarray, np.ndarray],
    iou_threshold: float,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Per-image TP/FP accumulation for Easy, Medium, and Hard subsets at once."""
    # Convert GT xywh → xyxy once, shared across all three subsets.
    gt_xyxy = gt_xywh.copy()
    gt_xyxy[:, 2] += gt_xyxy[:, 0]
    gt_xyxy[:, 3] += gt_xyxy[:, 1]

    # Compute IoU once; reuse for all three subsets.
    overlaps = get_bbox_iou_matrix(pred[:, :4], gt_xyxy)

    results = []
    for ignore in ignores:
        pred_recall = np.zeros(pred.shape[0])
        recall_list = np.zeros(gt_xywh.shape[0])
        proposal_list = np.ones(pred.shape[0])

        for h in range(pred.shape[0]):
            max_overlap = overlaps[h].max()
            max_idx = overlaps[h].argmax()
            if max_overlap >= iou_threshold:
                if ignore[max_idx] == 0:
                    recall_list[max_idx] = -1
                    proposal_list[h] = -1
                elif recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1
            pred_recall[h] = (recall_list == 1).sum()

        results.append((pred_recall, proposal_list))

    return results[0], results[1], results[2]


def _img_pr_info(
    pred: np.ndarray,
    proposal_list: np.ndarray,
    pred_recall: np.ndarray,
) -> np.ndarray:
    """Accumulate per-image precision/recall counts across 1000 score thresholds."""
    pr_info = np.zeros((_THRESH_NUM, 2))
    for t in range(_THRESH_NUM):
        thresh = 1 - (t + 1) / _THRESH_NUM
        r_index = np.where(pred[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            continue
        r_index = r_index[-1]
        pr_info[t, 0] = (proposal_list[: r_index + 1] == 1).sum()
        pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Compute VOC all-points-interpolated AP from a recall/precision curve."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    pts = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[pts + 1] - mrec[pts]) * mpre[pts + 1]))


def _pr_curve_to_ap(pr_curve: np.ndarray, count_face: int) -> float:
    """Convert a raw cumulative PR curve to AP using VOC interpolation."""
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(pr_curve[:, 0] > 0, pr_curve[:, 1] / pr_curve[:, 0], 0.0)
    rec = pr_curve[:, 1] / count_face
    return _voc_ap(rec, prec)


class YoloV5FaceEvaluator(BaseEvaluator):
    """Evaluator for YoloV5Face using the official WIDER FACE protocol."""

    def __init__(
        self,
        conf_threshold: float = 0.02,
        nms_iou_threshold: float = 0.5,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # Each entry: (pred [N,5], gt_xywh [M,4], ign_easy [M,], ign_med [M,], ign_hard [M,])
        self._samples: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self._cached_aps: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self._samples = []
        self._cached_aps = None

    def add_batch(
        self,
        output: Collection[torch.Tensor],
        gt: Collection[torch.Tensor],
    ) -> None:
        """Accumulate one batch of model predictions and ground-truth."""
        boxes_raw, scores_raw, _ = output
        scales, paddings, gt_boxes_xywh, ign_easy, ign_med, ign_hard, num_faces_t = gt

        batch_size = boxes_raw.shape[0]
        for i in range(batch_size):
            xyxy_lbox = box_xywh_to_xyxy(boxes_raw[i])
            (xyxy_nms,), (scores_nms,) = batched_nms(
                self.nms_iou_threshold,
                self.conf_threshold,
                xyxy_lbox.unsqueeze(0),
                scores_raw[i, :, 0].unsqueeze(0),
            )

            scale = float(scales[i].item())
            pad_left = float(paddings[i, 0].item())
            pad_top = float(paddings[i, 1].item())

            if len(xyxy_nms) > 0:
                pad = xyxy_nms.new_tensor([pad_left, pad_top, pad_left, pad_top])
                xyxy_orig = ((xyxy_nms - pad) / scale).cpu().numpy()
                pred_arr = np.concatenate(
                    [xyxy_orig, scores_nms.cpu().numpy()[:, None]], axis=1
                ).astype(np.float32)
            else:
                pred_arr = np.zeros((0, 5), dtype=np.float32)

            n = int(num_faces_t[i].item())
            self._samples.append(
                (
                    pred_arr,
                    gt_boxes_xywh[i, :n].cpu().numpy().astype(np.float32),
                    ign_easy[i, :n].cpu().numpy().astype(np.float32),
                    ign_med[i, :n].cpu().numpy().astype(np.float32),
                    ign_hard[i, :n].cpu().numpy().astype(np.float32),
                )
            )

    def get_accuracy_score(self) -> float:
        """Return Easy AP x 100 (headline WIDER FACE metric)."""
        easy_ap, _, _ = self._get_aps()
        return easy_ap * 100.0

    def formatted_accuracy(self) -> str:
        easy_ap, med_ap, hard_ap = self._get_aps()
        return (
            f"Easy {easy_ap * 100:.2f}%  "
            f"Medium {med_ap * 100:.2f}%  "
            f"Hard {hard_ap * 100:.2f}%  "
            "AP@IoU=0.5 (WIDER FACE)"
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_AVERAGE_PRECISION_IOU_50.with_description(
            "Easy-subset AP@IoU=0.5 on WIDER FACE validation split."
        )

    def _get_aps(self) -> tuple[float, float, float]:
        """Return cached Easy/Medium/Hard AP, computing on first call after reset."""
        if self._cached_aps is None:
            self._cached_aps = self._compute_aps()
        return self._cached_aps

    def _compute_aps(self) -> tuple[float, float, float]:
        if not self._samples:
            return 0.0, 0.0, 0.0

        # Score normalisation across the full dataset.
        # Computed once and reused across all three subset loops.
        all_scores = (
            np.concatenate([s[0][:, 4] for s in self._samples if len(s[0]) > 0], axis=0)
            if any(len(s[0]) > 0 for s in self._samples)
            else np.array([0.0, 1.0])
        )
        lo, hi = all_scores.min(), all_scores.max()
        diff = hi - lo

        pr_curves = [np.zeros((_THRESH_NUM, 2)) for _ in range(3)]
        count_faces = [0, 0, 0]

        for pred_arr, gt_xywh, ign_easy, ign_med, ign_hard in self._samples:
            count_faces[0] += int(ign_easy.sum())
            count_faces[1] += int(ign_med.sum())
            count_faces[2] += int(ign_hard.sum())

            if len(gt_xywh) == 0 or len(pred_arr) == 0:
                continue

            p = pred_arr.copy()
            if diff > 0:
                p[:, 4] = (p[:, 4] - lo) / diff

            # IoU computed once; reused for all three subsets inside _image_eval.
            (recall_e, proposal_e), (recall_m, proposal_m), (recall_h, proposal_h) = (
                _image_eval(
                    p, gt_xywh, (ign_easy, ign_med, ign_hard), self.nms_iou_threshold
                )
            )
            pr_curves[0] += _img_pr_info(p, proposal_e, recall_e)
            pr_curves[1] += _img_pr_info(p, proposal_m, recall_m)
            pr_curves[2] += _img_pr_info(p, proposal_h, recall_h)

        aps = [
            _pr_curve_to_ap(pr_curves[k], count_faces[k]) if count_faces[k] > 0 else 0.0
            for k in range(3)
        ]
        return aps[0], aps[1], aps[2]
