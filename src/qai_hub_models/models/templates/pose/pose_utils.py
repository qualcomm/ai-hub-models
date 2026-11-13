# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import math

import numpy as np

from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
)

BODY_SIGMAS = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]
FOOT_SIGMAS = [0.068, 0.066, 0.066, 0.092, 0.094, 0.094]
FACE_SIGMAS = [
    0.042,
    0.043,
    0.044,
    0.043,
    0.040,
    0.035,
    0.031,
    0.025,
    0.020,
    0.023,
    0.029,
    0.032,
    0.037,
    0.038,
    0.043,
    0.041,
    0.045,
    0.013,
    0.012,
    0.011,
    0.011,
    0.012,
    0.012,
    0.011,
    0.011,
    0.013,
    0.015,
    0.009,
    0.007,
    0.007,
    0.007,
    0.012,
    0.009,
    0.008,
    0.016,
    0.010,
    0.017,
    0.011,
    0.009,
    0.011,
    0.009,
    0.007,
    0.013,
    0.008,
    0.011,
    0.012,
    0.010,
    0.034,
    0.008,
    0.008,
    0.009,
    0.008,
    0.008,
    0.007,
    0.010,
    0.008,
    0.009,
    0.009,
    0.009,
    0.007,
    0.007,
    0.008,
    0.011,
    0.008,
    0.008,
    0.008,
    0.01,
    0.008,
]
LEFTHAND_SIGMAS = [
    0.029,
    0.022,
    0.035,
    0.037,
    0.047,
    0.026,
    0.025,
    0.024,
    0.035,
    0.018,
    0.024,
    0.022,
    0.026,
    0.017,
    0.021,
    0.021,
    0.032,
    0.02,
    0.019,
    0.022,
    0.031,
]
RIGHTHAND_SIGMAS = [
    0.029,
    0.022,
    0.035,
    0.037,
    0.047,
    0.026,
    0.025,
    0.024,
    0.035,
    0.018,
    0.024,
    0.022,
    0.026,
    0.017,
    0.021,
    0.021,
    0.032,
    0.02,
    0.019,
    0.022,
    0.031,
]


# OKS-NMS constants — derived from BODY_SIGMAS above.
# https://cocodataset.org/#keypoints-eval
_OKS_VARS = (np.array(BODY_SIGMAS) * 2) ** 2
OKS_THRE = 0.9  # standard OKS-NMS suppression threshold
IN_VIS_THRE = 0.2  # minimum keypoint visibility score


def get_final_preds(
    batch_heatmaps: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get final prediction after post process and transform over several batch.

    Parameters
    ----------
    batch_heatmaps
        Heatmap from the prediction,
        the shape is [batch_size, num_keypoints, height, width].
    center
        Center for each image, the shape is [batch_size, 2].
    scale
        Scale for each image, the shape is [batch_size, 2].

    Returns
    -------
    preds : np.ndarray
        Final prediction based on keypoints after post process and transform,
        the shape is [batch_size, num_keypoints, 2].
    maxval : np.ndarray
        Score for each prediction, the shape is [batch_size, num_keypoints, 1].
    """
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-process
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = math.floor(coords[n][p][0] + 0.5)
            py = math.floor(coords[n][p][1] + 0.5)
            if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                diff = np.array(
                    [
                        hm[py][px + 1] - hm[py][px - 1],
                        hm[py + 1][px] - hm[py - 1][px],
                    ]
                )
                coords[n][p] += np.sign(diff) * 0.25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = denormalize_coordinates_affine(
            coords[i],
            center[i],
            scale[i],
            0,
            (heatmap_width, heatmap_height),
        )
    return preds, maxvals


def get_max_preds(batch_heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from score maps over several batch.

    Parameters
    ----------
    batch_heatmaps
        Heatmap array with shape [batch_size, num_keypoints, height, width].

    Returns
    -------
    preds : np.ndarray
        Final prediction based on keypoints,
        the shape is [batch_size, num_keypoints, 2].
    maxval : np.ndarray
        Score for each prediction,
        the shape is [batch_size, num_keypoints, 1].
    """
    assert isinstance(batch_heatmaps, np.ndarray), (
        "batch_heatmaps should be numpy.ndarray"
    )
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # shape [batch, num joints, image h * w]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # index of max pixel per joint
    idx = np.argmax(heatmaps_reshaped, 2)

    # value of max pixel per joint
    maxvals = np.amax(heatmaps_reshaped, 2)

    # Reshape to prep for tiling
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # Tile indices to make room for (x, y)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # Convert index [..., 0] to x
    preds[:, :, 0] = (preds[:, :, 0]) % width
    # Convert [..., 1] to y
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # Tile mask back to (x, y) plane to ignore negatives
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # apply mask
    preds *= pred_mask
    return preds, maxvals


def oks_iou(
    g: np.ndarray,
    d: np.ndarray,
    a_g: float,
    a_d: np.ndarray,
) -> np.ndarray:
    """Compute OKS between one GT keypoint set and N detections.

    Parameters
    ----------
    g
        GT keypoints flattened: [x0, y0, v0, ...], shape (51,).
    d
        Detected keypoints, shape (N, 51).
    a_g
        GT bbox area (pixels^2).
    a_d
        Detected bbox areas, shape (N,).

    Returns
    -------
    np.ndarray
        OKS scores, shape (N,).
    """
    xg, yg = g[0::3], g[1::3]
    ious = np.zeros(d.shape[0])
    for n in range(d.shape[0]):
        xd, yd = d[n, 0::3], d[n, 1::3]
        dx, dy = xd - xg, yd - yg
        e = (dx**2 + dy**2) / _OKS_VARS / ((a_g + a_d[n]) / 2 + np.spacing(1)) / 2
        ious[n] = np.sum(np.exp(-e)) / e.shape[0]
    return ious


def oks_nms(kpts_db: list[dict]) -> list[int]:
    """Greedy OKS-NMS — returns indices of kept predictions."""
    if not kpts_db:
        return []
    scores = np.array([k["score"] for k in kpts_db])
    kpts = np.array([np.array(k["keypoints"]).flatten() for k in kpts_db])
    areas = np.array([k["area"] for k in kpts_db])
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        ious = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]])
        order = order[1:][ious <= OKS_THRE]
    return keep
