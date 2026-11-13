# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
)
from qai_hub_models.models.templates.pose.pose_evaluator import CocoBodyPoseEvaluator


class MovenetPoseEvaluator(CocoBodyPoseEvaluator):
    """Evaluator for MoveNet pose estimation models"""

    def __init__(self, height: int, width: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.input_height = height
        self.input_width = width

    def add_batch(
        self, output: torch.Tensor | tuple[torch.Tensor], gt: list[torch.Tensor]
    ) -> None:
        """
        Processes MoveNet outputs and converts them to COCO-format keypoint predictions.

        Parameters
        ----------
        output
            Model predictions with shape [batch, N_people, 17, 3] (y, x, confidence).
        gt
            List with the following tensors:

            image_ids
                Tensor (int) of image IDs [batch].
            category_ids
                Tensor (int) of category IDs [batch].
            centers
                Tensor (float) of bounding box centers [batch, 2].
            scale
                Tensor (float) of scale factors [batch, 2].
        """
        if isinstance(output, tuple):
            output = output[0]

        image_ids, category_ids, centers, scales = gt
        batch_size = output.shape[0]

        for idx in range(batch_size):
            img_id = int(image_ids[idx])
            cat_id = int(category_ids[idx])
            center = centers[idx].cpu().numpy()
            scale = scales[idx].cpu().numpy()
            input_size = (self.input_width, self.input_height)

            # Loop over all detected people
            n_people = output.shape[1]
            for person_idx in range(n_people):
                kp_with_scores = output[idx, person_idx].cpu().numpy()
                keypoints = kp_with_scores[:, :2]
                scores = kp_with_scores[:, 2]

                # Scale normalized (0-1) keypoints to input size
                keypoints *= input_size

                # Flip (y, x) -> (x, y)
                keypoints = np.flip(keypoints, axis=1).copy()

                # Apply inverse affine transform
                coords_tf = denormalize_coordinates_affine(
                    keypoints, center, scale, 0, input_size
                )

                # Store predictions
                self._store_predictions(
                    np.expand_dims(coords_tf, 0),  # (1, 17, 2)
                    np.expand_dims(scores, 0),  # (1, 17)
                    torch.tensor([img_id]),
                    torch.tensor([cat_id]),
                )
