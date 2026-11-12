# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
)
from qai_hub_models.models.templates.corevision.image_processing_3d import ddd2locrot


def ddd_post_process(
    dets: np.ndarray,
    c: list[np.ndarray],
    s: list[np.ndarray],
    out_shape: tuple[int, int],
    calibs: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Post-processes results from the CenterNet model for 3D object detection.

    Parameters
    ----------
    dets
        Raw detection output from the model.
        Shape: (batch_size, max_dets, 18), where 18 includes:
        (center_x, center_y, score, alpha_components[8],
        depth, dimensions[3], wh[2], class_id).
    c
        List of center coordinates for each image in the batch,
        as returned by pre_process. Shape: (batch_size, 2).
    s
        List of scale factors for each image in the batch,
        as returned by pre_process. Shape: (batch_size, 2).
    out_shape
        The output feature map shape (height, width) from the model.
    calibs
        List of camera calibration matrices (P) for each image
        in the batch. Each P is a 3x4 projection matrix.

    Returns
    -------
    detections : list[np.ndarray]
        A list (batch_size) of NumPy array. Each NumPy has the shape (max_dets, 14)
        in the format: (alpha, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        dim_x, dim_y, dim_z, loc_x, loc_y, loc_z, rotation_y, score, label).
    """
    output_h, output_w = out_shape
    ret = []

    for i in range(dets.shape[0]):
        # Transform center coordinates back to original image space
        center = denormalize_coordinates_affine(
            dets[i, :, :2], c[i], s[i], 0, (output_w, output_h)
        )
        score = dets[i, :, 2:3]

        # Decode alpha from the 8-dim rotation output
        alpha = get_alpha(dets[i, :, 3:11])
        depth = dets[i, :, 11:12]
        dimensions = dets[i, :, 12:15]  # h, w, l
        label = dets[i, :, 17:18]

        # Transform center coordinates back to original image space
        wh = denormalize_coordinates_affine(
            dets[i, :, 15:17], c[i], s[i], 0, (output_w, output_h)
        )
        # Convert 2D detection and depth to 3D location and rotation_y
        locations, rotation_y = ddd2locrot(
            center, alpha, dimensions[:, 0], depth, calibs[i]
        )

        bbox = np.array(
            [
                center[:, 0] - wh[:, 0] / 2,
                center[:, 1] - wh[:, 1] / 2,
                center[:, 0] + wh[:, 0] / 2,
                center[:, 1] + wh[:, 1] / 2,
            ]
        ).transpose(1, 0)
        pred = np.concatenate(
            [
                alpha,
                bbox,
                dimensions,
                locations,
                rotation_y,
                score,
                label,
            ],
            1,
        )
        ret.append(pred)
    return ret


def get_alpha(rot: np.ndarray) -> np.ndarray:
    """
    Decodes the observation angle (alpha).

    Parameters
    ----------
    rot
        The 8-dimensional rotation output for a batch of detections.
        Shape: (N, 8), where N is the number of detections.
        Format: [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
                bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos].

    Returns
    -------
    alpha : np.ndarray
        The decoded observation angle `alpha` for each detection. Shape: (N, 1).
    """
    # Calculate alpha for bin 1 and bin 2 separately
    alpha1 = np.arctan2(rot[:, 2:3], rot[:, 3:4]) + (-0.5 * np.pi)
    alpha2 = np.arctan2(rot[:, 6:7], rot[:, 7:8]) + (0.5 * np.pi)

    # Select alpha based on the dominant bin
    idx = rot[:, 1:2] > rot[:, 5:6]
    return alpha1 * idx + alpha2 * (1 - idx)
