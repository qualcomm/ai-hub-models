# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np

_FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
_FULL_KERNEL_9 = np.ones((9, 9), np.uint8)

_CROSS_KERNEL_3 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

_CROSS_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ],
    dtype=np.uint8,
)

_CROSS_KERNEL_7 = np.array(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


def fill_in_multiscale(
    depth_map: np.ndarray,
    max_depth: float = 8.0,
    dilation_kernel_far: np.ndarray = _CROSS_KERNEL_3,
    dilation_kernel_med: np.ndarray = _CROSS_KERNEL_5,
    dilation_kernel_near: np.ndarray = _CROSS_KERNEL_7,
    extrapolate: bool = False,
    blur_type: str = "bilateral",
) -> np.ndarray:
    """
    Multi-scale depth hole-filling using distance-aware dilation.

    Fills holes in a depth map by applying morphological dilation at three
    distance ranges (near/med/far) with separate kernels, followed by
    morphological closing, median blur, top-row extrapolation, and a final
    bilateral or Gaussian smoothing pass.

    Copied verbatim from AG-Pose data_utils.py::fill_in_multiscale().

    Parameters
    ----------
    depth_map
        Input depth map in metres as float32 array of shape (H, W).
        Zero or negative values are treated as holes.
    max_depth
        Maximum depth value used for inversion-based dilation. Default 8.0 m.
    dilation_kernel_far
        Morphological kernel applied to far-range pixels (depth > 2 m).
    dilation_kernel_med
        Morphological kernel applied to mid-range pixels (1 m < depth <= 2 m).
    dilation_kernel_near
        Morphological kernel applied to near-range pixels (depth <= 1 m).
    extrapolate
        If True, extrapolate depth values above the topmost valid pixel
        in each column. Default False.
    blur_type
        Final smoothing pass — 'bilateral' (default) or 'gaussian'.

    Returns
    -------
    np.ndarray
        Hole-filled depth map in metres, same shape as input.
    """
    depths_in = np.float32(depth_map)

    valid_pixels_near = np.asarray((depths_in > 0.01) & (depths_in <= 1.0))
    valid_pixels_med = np.asarray((depths_in > 1.0) & (depths_in <= 2.0))
    valid_pixels_far = np.asarray(depths_in > 2.0)

    s1 = np.copy(depths_in)
    valid_pixels = s1 > 0.01
    s1[valid_pixels] = max_depth - s1[valid_pixels]

    dilated_far = cv2.dilate(np.multiply(s1, valid_pixels_far), dilation_kernel_far)
    dilated_med = cv2.dilate(np.multiply(s1, valid_pixels_med), dilation_kernel_med)
    dilated_near = cv2.dilate(np.multiply(s1, valid_pixels_near), dilation_kernel_near)

    valid_pixels_near = np.asarray(dilated_near > 0.01)
    valid_pixels_med = np.asarray(dilated_med > 0.01)
    valid_pixels_far = np.asarray(dilated_far > 0.01)

    s2 = np.copy(s1)
    s2[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2[valid_pixels_near] = dilated_near[valid_pixels_near]

    s3 = cv2.morphologyEx(s2, cv2.MORPH_CLOSE, _FULL_KERNEL_5)

    s4 = np.copy(s3)
    blurred = cv2.medianBlur(s3, 5)
    s4[s3 > 0.01] = blurred[s3 > 0.01]

    top_mask = np.ones(depths_in.shape, dtype=np.bool_)
    for col in range(s4.shape[1]):
        top_row = np.argmax(s4[:, col] > 0.01)
        top_mask[0:top_row, col] = False

    empty_pixels = ~(s4 > 0.01) & top_mask
    dilated = cv2.dilate(s4, _FULL_KERNEL_9)
    s5 = np.copy(s4)
    s5[empty_pixels] = dilated[empty_pixels]

    s6 = np.copy(s5)
    top_mask2 = np.ones(s5.shape, dtype=np.bool_)
    top_rows = np.argmax(s5 > 0.01, axis=0)
    top_vals = s5[top_rows, range(s5.shape[1])]
    for col in range(s5.shape[1]):
        if extrapolate:
            s6[0 : top_rows[col], col] = top_vals[col]
        else:
            top_mask2[0 : top_rows[col], col] = False

    s7 = np.copy(s6)
    for _ in range(6):
        empty = (s7 < 0.01) & top_mask2
        dilated = cv2.dilate(s7, _FULL_KERNEL_5)
        s7[empty] = dilated[empty]

    blurred = cv2.medianBlur(s7, 5)
    valid = (s7 > 0.01) & top_mask2
    s7[valid] = blurred[valid]

    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(s7, (5, 5), 0)
        valid = (s7 > 0.01) & top_mask2
        s7[valid] = blurred[valid]
    elif blur_type == "bilateral":
        blurred = cv2.bilateralFilter(s7, 5, 0.5, 2.0)
        s7[valid] = blurred[valid]

    s8 = np.copy(s7)
    valid = np.where(s8 > 0.01)  # type: ignore[assignment]
    s8[valid] = max_depth - s8[valid]

    return s8


def fill_missing(
    depth: np.ndarray,
    cam_scale: float,
    scale_2_80m: float,
    fill_type: str = "multiscale",
    blur_type: str = "bilateral",
) -> np.ndarray:
    """
    Depth hole-filling entry point for AG-Pose preprocessing.

    Normalises the raw uint16 depth map to metres, delegates to
    ``fill_in_multiscale`` for hole filling, then converts back to
    the original scale. Typically called as::

        fill_missing(depth, DEPTH_NORM_SCALE, 1)

    Copied verbatim from AG-Pose data_utils.py::fill_missing().

    Parameters
    ----------
    depth
        Raw depth array of shape (H, W) in sensor units (e.g. uint16 mm).
    cam_scale
        Scale factor to convert sensor units to metres
        (e.g. 1000.0 for mm → m).
    scale_2_80m
        Additional scale applied before hole-filling to normalise depth
        to the 0-8 m range expected by ``fill_in_multiscale``. Typically 1.
    fill_type
        Hole-filling strategy. Only 'multiscale' is supported.
    blur_type
        Smoothing pass passed through to ``fill_in_multiscale``.
        'bilateral' (default) or 'gaussian'.

    Returns
    -------
    np.ndarray
        Hole-filled depth array in the same sensor units as input.
    """
    dpt = depth / cam_scale * scale_2_80m
    if fill_type == "multiscale":
        dpt = fill_in_multiscale(
            dpt, extrapolate=False, blur_type=blur_type, max_depth=3.0
        )
    else:
        raise ValueError(f"Invalid fill_type: {fill_type}")
    return dpt / scale_2_80m * cam_scale
