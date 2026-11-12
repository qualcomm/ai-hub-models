# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, batch_probiou

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    rotated_batched_nms,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    MEAN_AVERAGE_PRECISION_IOU_5_95,
    MetricMetadata,
)


@dataclass
class UltralyticsOBBConfig:
    num_classes: int
    iouv: torch.Tensor = field(default_factory=lambda: torch.linspace(0.5, 0.95, 10))
    angles_in_radians: bool = True
    enforce_0_90_deg: bool = True


class OBBEvaluator(BaseEvaluator):
    """Evaluator for oriented bounding box detection outputs."""

    mAP_default_low_iOU: float = 0.50
    mAP_default_high_iOU: float = 0.95
    mAP_default_increment_iOU: float = 0.05
    stats: dict[str, list[torch.Tensor]]

    def __init__(
        self,
        num_classes: int,
        image_height: int,
        image_width: int,
        nms_iou_threshold: float = 0.45,
        score_threshold: float = 0.25,
        names: dict[int, str] | None = None,
        cfg: UltralyticsOBBConfig | None = None,
    ) -> None:
        self.cfg = cfg or UltralyticsOBBConfig(num_classes=num_classes)
        self.names = names or {i: str(i) for i in range(num_classes)}
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold
        self.class_aware_nms = True
        self.canonicalize_rboxes = False
        self.scale_x = 1 / image_width
        self.scale_y = 1 / image_height
        self.scale_wh = 1 / max(image_width, image_height)
        self.reset()

    def reset(self) -> None:
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        self._cached_results: dict[str, float] | None = None

    def get_accuracy_score(self) -> float:
        return float(self.compute().get("map", 0.0))

    def formatted_accuracy(
        self,
        low_iOU: float | None = None,
        high_iOU: float | None = None,
        increment_iOU: float | None = None,
    ) -> str:
        low = self.mAP_default_low_iOU if low_iOU is None else float(low_iOU)
        high = self.mAP_default_high_iOU if high_iOU is None else float(high_iOU)
        increment = (
            self.mAP_default_increment_iOU
            if increment_iOU is None
            else float(increment_iOU)
        )
        m_ap = float(self.compute().get("map", 0.0))
        return f"{m_ap:.3f} mAP@{low:.2f}:{high:.2f}:{increment:.2f}"

    @torch.no_grad()
    def add_batch(
        self,
        output: Collection[torch.Tensor],
        gt: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> None:
        self._cached_results = None
        pred_boxes, pred_angles, pred_scores, pred_class_idx = output
        gt_boxes = gt[3]
        gt_labels = gt[4]
        gt_counts = gt[5].squeeze(-1)
        device = pred_boxes.device

        boxes_list, scores_list, angles_list, classes_list = rotated_batched_nms(
            pred_boxes,
            pred_angles,
            pred_scores,
            pred_class_idx,
            score_thr=self.score_threshold or 0.0,
            iou_thr=self.nms_iou_threshold or 0.5,
            class_aware=self.class_aware_nms,
            canonicalize=self.canonicalize_rboxes,
        )

        for batch_idx in range(pred_boxes.shape[0]):
            n_gt = int(gt_counts[batch_idx].item())
            gtb = gt_boxes[batch_idx, :n_gt].to(device)
            gtc = gt_labels[batch_idx, :n_gt].to(device)
            pb4 = boxes_list[batch_idx].to(device)
            ps = scores_list[batch_idx].to(device)
            pa = angles_list[batch_idx].to(device)
            pc = classes_list[batch_idx].to(device)

            if pa.ndim == 2 and pa.shape[-1] == 1:
                pa = pa.squeeze(-1)

            if pb4.numel() > 0:
                pb4 = pb4.clone()
                pb4[:, 0] *= self.scale_x
                pb4[:, 1] *= self.scale_y
                pb4[:, 2] *= self.scale_wh
                pb4[:, 3] *= self.scale_wh

            if pb4.numel() == 0:
                self.stats["tp"].append(
                    torch.zeros((0, self.cfg.iouv.numel()), dtype=torch.bool)
                )
                self.stats["conf"].append(torch.zeros((0,), dtype=torch.float32))
                self.stats["pred_cls"].append(torch.zeros((0,), dtype=torch.int64))
                self.stats["target_cls"].append(gtc.detach().cpu())
                continue

            pb = torch.cat([pb4, pa.unsqueeze(1)], dim=1)

            if n_gt == 0:
                tp = torch.zeros(
                    (pb.shape[0], self.cfg.iouv.numel()),
                    device=device,
                    dtype=torch.bool,
                )
                self.stats["tp"].append(tp.cpu())
                self.stats["conf"].append(ps.cpu())
                self.stats["pred_cls"].append(pc.cpu())
                self.stats["target_cls"].append(torch.zeros((0,), dtype=torch.int64))
                continue

            if ps.numel() > 0:
                order = torch.argsort(ps, descending=True)
                pb = pb[order]
                ps = ps[order]
                pc = pc[order]

            iou = batch_probiou(pb, gtb)
            cls_match = pc[:, None] == gtc[None, :]
            iou = iou * cls_match.float()

            tp = torch.zeros(
                (pb.shape[0], self.cfg.iouv.numel()), device=device, dtype=torch.bool
            )
            matched_gt = torch.zeros((n_gt,), device=device, dtype=torch.bool)

            best_iou, best_gt = iou.max(dim=1)
            candidate_pred_idx = torch.nonzero(best_iou > 0, as_tuple=False).flatten()
            min_iou = self.cfg.iouv[0]

            # Predictions are already sorted by descending confidence above, so
            # this preserves the existing greedy matching behavior while avoiding
            # a per-prediction argmax in Python.
            for pred_idx in candidate_pred_idx.tolist():
                gt_idx = int(best_gt[pred_idx].item())
                if matched_gt[gt_idx]:
                    continue
                best = best_iou[pred_idx]
                tp[pred_idx] = best >= self.cfg.iouv
                if best >= min_iou:
                    matched_gt[gt_idx] = True

            self.stats["tp"].append(tp.cpu())
            self.stats["conf"].append(ps.cpu())
            self.stats["pred_cls"].append(pc.cpu())
            self.stats["target_cls"].append(gtc.cpu())

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_AVERAGE_PRECISION_IOU_5_95

    def compute(self) -> dict[str, float]:
        if self._cached_results is not None:
            return self._cached_results

        zero_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "map": 0.0,
        }
        if len(self.stats["target_cls"]) == 0:
            self._cached_results = zero_metrics
            return self._cached_results

        tp = torch.cat(self.stats["tp"], dim=0).cpu().numpy()
        conf = torch.cat(self.stats["conf"], dim=0).cpu().numpy()
        pred_cls = torch.cat(self.stats["pred_cls"], dim=0).cpu().numpy()
        target_cls = torch.cat(self.stats["target_cls"], dim=0).cpu().numpy()

        if conf.size == 0 or target_cls.size == 0:
            self._cached_results = zero_metrics
            return self._cached_results

        _, _, p, r, f1, ap, *_ = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=False,
            names=self.names,
        )

        map50 = float(ap[:, 0].mean()) * 100.0 if ap.size else 0.0
        map75 = float(ap[:, 5].mean()) * 100.0 if ap.shape[1] > 5 else 0.0
        map_all = float(ap.mean()) * 100.0 if ap.size else 0.0
        self._cached_results = {
            "precision": float(np.nanmean(p)) * 100.0 if len(p) else 0.0,
            "recall": float(np.nanmean(r)) * 100.0 if len(r) else 0.0,
            "f1": float(np.nanmean(f1)) * 100.0 if len(f1) else 0.0,
            "map50": map50,
            "map75": map75,
            "map": map_all,
        }
        return self._cached_results
