# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from qai_hub_models.extern.xtcocotools.cocoeval import COCOeval
from qai_hub_models.models.templates.pose.pose_evaluator import (
    CocoKeypointsPoseEvaluator,
)
from qai_hub_models.models.templates.pose.pose_utils import get_final_preds, oks_nms
from qai_hub_models.utils.printing import suppress_stdout


class HRNetPosePrecomputedBoxEvaluator(CocoKeypointsPoseEvaluator):
    """Evaluator for HRNet using precomputed COCO person boxes."""

    def add_batch(
        self,
        output: tuple[torch.Tensor, ...] | torch.Tensor,
        gt: tuple,
    ) -> None:
        heatmaps = output[0] if isinstance(output, tuple) else output
        image_ids, category_ids, centers, scales, box_scores, areas = gt
        preds, maxvals = get_final_preds(
            heatmaps.detach().cpu().numpy(),
            centers.numpy(),
            scales.numpy(),
        )
        self._store_predictions(
            preds, maxvals, image_ids, category_ids, box_scores, areas
        )


class HRNetPoseEvaluator(CocoKeypointsPoseEvaluator):
    """Evaluator for the end-to-end RTMDet and HRNet pipeline."""

    def reset(self) -> None:
        super().reset()
        self.evaluated_image_ids: set[int] = set()

    def add_batch(
        self,
        output: tuple[torch.Tensor, ...],
        gt: tuple,
    ) -> None:
        """Process a batch of HRNet model outputs and ground truth data.

        Parameters
        ----------
        output
            Heatmaps plus detector crop metadata.
        gt
            Tuple of (image_ids, category_ids, image_scales, pads).

            image_ids : torch.Tensor
                COCO image IDs, shape (B,).
            category_ids : torch.Tensor
                COCO category IDs, shape (B,).
            image_scales : torch.Tensor
                Letterbox resize factors, shape (B,).
            pads : torch.Tensor
                Left and top padding in detector pixels, shape (B, 2).
        """
        heatmaps, centers, scales, box_scores, areas, batch_indices = output
        image_ids, category_ids, image_scales, pads = gt
        self.evaluated_image_ids.update(int(image_id) for image_id in image_ids)
        if heatmaps.shape[0] == 0:
            return
        preds, maxvals = get_final_preds(
            heatmaps.detach().cpu().numpy(),
            centers.numpy(),
            scales.numpy(),
        )
        batch_indices = batch_indices.long()
        pred_image_ids = image_ids[batch_indices]
        pred_category_ids = category_ids[batch_indices]
        pred_scales = image_scales[batch_indices].numpy()[:, None, None]
        pred_pads = pads[batch_indices].numpy()[:, None, :]
        preds = (preds - pred_pads) / pred_scales
        areas = areas / image_scales[batch_indices].square()
        self._store_predictions(
            preds,
            maxvals,
            pred_image_ids,
            pred_category_ids,
            box_scores,
            areas,
        )

    def get_coco_mAP(self) -> dict[str, Any]:
        """Score every evaluated image, including images with no detections."""
        if not self.predictions:
            return {"AP": 0.0, "AP@.5": 0.0}
        per_image: dict[int, list[dict]] = defaultdict(list)
        for prediction in self.predictions:
            per_image[prediction["image_id"]].append(prediction)
        results: list[dict] = []
        for predictions in per_image.values():
            results.extend(
                {
                    key: value
                    for key, value in predictions[index].items()
                    if key != "area"
                }
                for index in oks_nms(predictions)
            )
        with suppress_stdout():
            coco_dt = self.coco_gt.loadRes(results)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "keypoints")
            coco_eval.params.useSegm = None
            coco_eval.params.imgIds = sorted(self.evaluated_image_ids)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        return {"AP": coco_eval.stats[0] * 100, "AP@.5": coco_eval.stats[1] * 100}
