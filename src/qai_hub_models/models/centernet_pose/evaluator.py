# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
)
from qai_hub_models.models.templates.pose.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.utils.printing import suppress_stdout

# Visibility threshold for detection score filtering
DET_SCORE_THRE = 0.1


class CenternetPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for CenterNet multi-pose estimation."""

    def __init__(
        self,
        decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
        max_dets: int = 100,
    ) -> None:
        """
        Parameters
        ----------
        decode
            Function to decode the raw model outputs
            into detected objects/detections and keypoints.
        max_dets
            Maximum number of detections per image.
        """
        self.decode = decode
        self.max_dets = max_dets
        with suppress_stdout():
            super().__init__()

    def add_batch(
        self,
        output: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        gt: list[Any],
    ) -> None:
        """Process a batch of Centernet model outputs and ground truth data.

        Parameters
        ----------
        output
            Model predictions which can be:

            hm
                Heatmap with the shape of [B, num_classes, H//4, W//4].
            wh
                Width/Height value with the shape of [B, 2, H//4, W//4].
            hps
                Keypoint offsets relative to the object center
                with the shape of [B, 2* num_joints, H//4, W//4].
            reg
                2D regression value with the shape of [B, 2, H//4, W//4].
            hm_hp
                Keypoint heatmap with the shape of [B, num_joints, H//4, W//4].
            hm_offset
                Heatmap offset with the shape of [B, 2, H//4, W//4].

            where num_joints = 17, num_classes = 1.
        gt
            Ground truth data containing:

            image_ids
                Tensor (int) of COCO image IDs [batch].
            category_ids
                Tensor (int) of COCO category IDs [batch]. Unused — predictions
                hard-code category_id=1 (person).
            centers
                Tensor (float) of bounding box centers [batch, 2].
            scales
                Tensor (float) of scale factors [batch, 2].
        """
        hm, wh, hps, reg, hm_hp, hm_offset = output
        image_ids, _category_ids, centers, scales = gt

        dets_pt = self.decode(hm, wh, hps, reg, hm_hp, hm_offset, self.max_dets)
        dets = dets_pt.detach().cpu().numpy()  # [B, max_dets, 40]

        hm_h, hm_w = hm.shape[2], hm.shape[3]

        for b in range(dets.shape[0]):
            img_id = int(image_ids[b])
            center = centers[b].numpy() if hasattr(centers[b], "numpy") else centers[b]
            scale_arr = (
                scales[b].numpy()
                if hasattr(scales[b], "numpy")
                else np.array(scales[b])
            )

            for k in range(dets.shape[1]):
                det_score = float(dets[b, k, 4])
                if det_score < DET_SCORE_THRE:
                    continue

                # Keypoint coords: dets[b, k, 5:39] = 17*(x, y)
                kpts_raw = dets[b, k, 5:39].reshape(17, 2)
                kpts = denormalize_coordinates_affine(
                    kpts_raw, center, scale_arr, 0, (hm_w, hm_h)
                )

                # Source sets visibility=1 for all joints (not predicted score)
                keypoints: list[float] = []
                for j in range(17):
                    keypoints += [float(kpts[j, 0]), float(kpts[j, 1]), 1.0]

                self.predictions.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": keypoints,
                        "score": det_score,
                    }
                )
