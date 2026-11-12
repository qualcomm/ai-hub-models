# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.posenet_mobilenet.app import decode_multiple_poses
from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
)
from qai_hub_models.models.templates.pose.pose_evaluator import CocoBodyPoseEvaluator


class PosenetMobilenetEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for Posenet Mobilenet estimation models"""

    def __init__(
        self, image_height: int, image_width: int, in_vis_thre: float = 0.2
    ) -> None:
        super().__init__(in_vis_thre=in_vis_thre)

        self.input_width = image_width
        self.input_height = image_height

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        gt: list[torch.Tensor],
    ) -> None:
        """Process a batch of Posenet Mobilenet model outputs and ground truth data.

        Parameters
        ----------
        output
            Model predictions containing:

            heatmaps_result
                Tensor (float), [batch, 17, 33, 17].
            offsets_result
                Tensor (float), [batch, 34, 33, 17].
            displacement_fwd_result
                Tensor (float), [batch, 32, 33, 17].
            displacement_bwd_result
                Tensor (float), [batch, 32, 33, 17].
        gt
            Ground truth data containing:

            image_ids
                Tensor (int), [batch].
            category_ids
                Tensor (int), [batch].
            centers
                Tensor (float), [batch, 2].
            scale
                Tensor (float), [batch, 2].
        """
        heatmaps, offsets, disp_fwd, disp_bwd = output
        img_ids, cat_ids, centers, scales = gt

        pose_coords_list: list[np.ndarray] = []
        pose_scores_list: list[np.ndarray] = []

        for idx in range(heatmaps.shape[0]):
            _, k_scores, k_coords = decode_multiple_poses(
                heatmaps[idx : idx + 1].squeeze(0),
                offsets[idx : idx + 1].squeeze(0),
                disp_fwd[idx : idx + 1].squeeze(0),
                disp_bwd[idx : idx + 1].squeeze(0),
                max_pose_detections=10,
                min_pose_score=0.25,
            )

            # Take first pose with high score
            k_coords = k_coords[0:1]
            k_scores = np.expand_dims(k_scores[0:1], -1).astype(np.float32)

            # Transforms predicted keypoints from model's output space to original image coordinates:
            # 1. Computes an inverse affine transformation to undo the preprocessing (cropping, scaling)
            #    using the original image center and scale, with no rotation (rot=0).
            # 2. Converts from (y,x) heatmap grid format to standard (x,y) pixel coordinates
            t_coords = np.zeros_like(k_coords)
            for pose_idx in range(k_coords.shape[0]):
                t_coords[pose_idx] = denormalize_coordinates_affine(
                    np.flip(k_coords[pose_idx], axis=1),
                    center=centers[idx].numpy(),
                    scale=scales[idx].numpy(),
                    rot=0,
                    output_size=(self.input_width, self.input_height),
                )
            pose_coords_list.append(t_coords)
            pose_scores_list.append(k_scores)

        pose_coords = np.concatenate(pose_coords_list, axis=0)
        pose_scores = np.concatenate(pose_scores_list, axis=0)

        self._store_predictions(pose_coords, pose_scores, img_ids, cat_ids)
