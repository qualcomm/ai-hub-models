# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
import torch

from qai_hub_models.models.lightglue.postprocess import DEFAULT_HEIGHT, DEFAULT_WIDTH
from qai_hub_models.models.superpoint.postprocess import warp_points_xy
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import HOMOGRAPHY_ACCURACY, MetricMetadata


class LightGlueEvaluator(BaseEvaluator):
    """Evaluates LightGlue on HPatches using homography estimation @3px.

    Unlike a from-scratch matcher evaluation, this trusts LightGlue's own
    mutual-nearest-neighbour matches directly (no brute-force re-matching):
    it builds point correspondences from `matches0`/`mscores0` as LightGlue
    reports them, then estimates a homography via RANSAC and checks the mean
    corner reprojection error.

    This reports correctness@3px (pass/fail rate at a single, fixed 3px
    reprojection threshold -- matching HOMOGRAPHY_ACCURACY's registered
    name/unit), not the paper's AUC@{1,3,5}px metric, and necessarily pairs
    LightGlue's official checkpoint (trained against Magic-Leap's SuperPoint)
    with this repo's MIT-licensed SuperPoint port -- the resulting numbers
    are directionally meaningful but not a direct reproduction of the
    paper's reported figure.
    """

    CORRECTNESS_THRESH_PX = 3.0

    def __init__(
        self,
        image_height: int = DEFAULT_HEIGHT,
        image_width: int = DEFAULT_WIDTH,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        self.image_height = image_height
        self.image_width = image_width
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.reset()

    def reset(self) -> None:
        self._num_total = 0
        self._homo_correct = 0

    def get_accuracy_score(self) -> float:
        if self._num_total == 0:
            return 0.0
        return 100.0 * self._homo_correct / self._num_total

    def formatted_accuracy(self) -> str:
        return f"Homography@3 {self.get_accuracy_score():.1f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return HOMOGRAPHY_ACCURACY

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Accumulate LightGlue matches for one pair at a time.

        Parameters
        ----------
        output
            LightGlue outputs: (matches0, matches1, mscores0, mscores1).
            matches0/1 shape (B, MAX_K) int, mscores0/1 shape (B, MAX_K) float.
        gt
            (H_gt, keypoints0, keypoints1, num_keypoints0, num_keypoints1).
            H_gt shape (B, 3, 3); keypoints{0,1} shape (B, MAX_K, 2);
            num_keypoints{0,1} shape (B,).
        """
        m0_batch, _m1_batch, _mscores0_batch, _mscores1_batch = output
        h_gt_batch, kp0_batch, kp1_batch, n0_batch, n1_batch = gt

        B = m0_batch.shape[0]
        for i in range(B):
            n0 = int(n0_batch[i].item())
            n1 = int(n1_batch[i].item())
            m0 = m0_batch[i, :n0].cpu().numpy()
            kp0 = kp0_batch[i, :n0].cpu().numpy()
            kp1 = kp1_batch[i, :n1].cpu().numpy()
            h_gt = h_gt_batch[i].cpu().numpy().astype(np.float64)

            # Padded targets are already excluded from the raw matches
            # themselves (see LightGlue.forward()), so filtering on the
            # -1 sentinel is sufficient here.
            valid = m0 > -1
            idx0 = np.where(valid)[0]
            idx1 = m0[valid].astype(np.int64)

            correct = self._evaluate_pair(kp0[idx0], kp1[idx1], h_gt)
            self._homo_correct += int(correct)
            self._num_total += 1

    def _evaluate_pair(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        h_gt: np.ndarray,
    ) -> bool:
        """RANSAC-fit a homography from matched points and check corner error.

        Parameters
        ----------
        pts0
            Matched reference keypoints (x, y), shape (N, 2).
        pts1
            Corresponding target keypoints (x, y), shape (N, 2).
        h_gt
            Ground-truth homography (3, 3) mapping ref -> tgt.

        Returns
        -------
        bool
            True if the estimated homography is correct within threshold.
        """
        if len(pts0) < 4:
            return False
        h_est, inliers = cv2.findHomography(
            pts0.astype(np.float32),
            pts1.astype(np.float32),
            cv2.RANSAC,
            self.ransac_reproj_threshold,
        )
        if h_est is None or inliers is None or inliers.sum() < 4:
            return False
        h, w = self.image_height, self.image_width
        corners = torch.tensor(
            [[0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]]
        )
        true_corners = warp_points_xy(corners, torch.from_numpy(h_gt).float())
        estimated_corners = warp_points_xy(corners, torch.from_numpy(h_est).float())
        mean_dist = (true_corners - estimated_corners).norm(dim=1).mean().item()
        return mean_dist < self.CORRECTNESS_THRESH_PX
