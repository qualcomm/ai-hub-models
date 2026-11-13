# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import torch

from qai_hub_models.models.superpoint.postprocess import extract_keypoints

MAX_KEYPOINTS = 500
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640

DEFAULT_DETECTION_THRESHOLD = 0.001
DEFAULT_STRIDE = 8
DEFAULT_NMS_RADIUS = 4
DEFAULT_REMOVE_BORDERS = 4


def extract_padded_features(
    extractor: Callable[
        [torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    max_keypoints: int = MAX_KEYPOINTS,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    stride: int = DEFAULT_STRIDE,
    nms_radius: int = DEFAULT_NMS_RADIUS,
    remove_borders: int = DEFAULT_REMOVE_BORDERS,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Run SuperPoint + host-side keypoint extraction on an image pair, zero-
    padded to a fixed length so the result can feed LightGlue's fixed-shape
    forward pass directly.

    SuperPoint's current (dense-output) recipe only returns per-pixel score/
    descriptor maps -- keypoint extraction (NMS, top-k, border removal) runs
    host-side. LightGlue's own forward pass needs a fixed keypoint count, so
    `max_keypoints` is both the `top_k` passed to extraction and the length
    both images' keypoints/descriptors are zero-padded to.

    Parameters
    ----------
    extractor
        A SuperPoint model (already ``.eval()``) or on-device stand-in with
        the same call signature.
    image_a
        RGB tensor, shape ``(1, 3, H, W)``, values in ``[0, 1]``.
    image_b
        RGB tensor, shape ``(1, 3, H, W)``, values in ``[0, 1]``.
    max_keypoints
        Keypoints are extracted with ``top_k=max_keypoints`` and the result
        is zero-padded up to exactly this length.
    detection_threshold
        Minimum score for a pixel to be considered a keypoint.
    stride
        Downsampling factor between the score map and the descriptor map.
    nms_radius
        Radius (pixels) for non-maximum suppression.
    remove_borders
        Width (pixels) of the border to exclude from detection.

    Returns
    -------
    keypoints0 : torch.Tensor
        Shape ``(1, max_keypoints, 2)`` float32, ``(x, y)`` pixel
        coordinates for image_a; padded rows are zero.
    descriptors0 : torch.Tensor
        Shape ``(1, max_keypoints, 256)`` float32 for image_a; padded rows
        are zero.
    num_keypoints0 : torch.Tensor
        Shape ``(1,)`` int32, actual (unpadded) keypoint count for image_a.
    keypoints1 : torch.Tensor
        Same as ``keypoints0``, for image_b.
    descriptors1 : torch.Tensor
        Same as ``descriptors0``, for image_b.
    num_keypoints1 : torch.Tensor
        Same as ``num_keypoints0``, for image_b.
    """
    with torch.no_grad():
        scores_a, desc_a, scores_b, desc_b = extractor(image_a, image_b)

    def _extract_and_pad(
        scores: torch.Tensor, descriptors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp, _scores, desc = extract_keypoints(
            scores[0],
            descriptors[0],
            detection_threshold,
            max_keypoints,
            stride,
            nms_radius,
            remove_borders,
        )
        n = kp.shape[0]
        kp_padded = kp.new_zeros(max_keypoints, 2)
        desc_padded = desc.new_zeros(max_keypoints, desc.shape[-1])
        kp_padded[:n] = kp
        desc_padded[:n] = desc
        num_kp = torch.tensor([n], dtype=torch.int32)
        return kp_padded[None], desc_padded[None], num_kp

    kp0, desc0, n0 = _extract_and_pad(scores_a, desc_a)
    kp1, desc1, n1 = _extract_and_pad(scores_b, desc_b)
    return kp0, desc0, n0, kp1, desc1, n1
