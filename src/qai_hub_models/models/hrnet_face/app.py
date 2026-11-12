# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL.Image import Image, fromarray

from qai_hub_models.models.templates.corevision.draw import draw_points
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


class HRNetFaceApp:
    """End-to-end app for HRNet Face landmark inference."""

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image]:
        return self.predict_face_keypoints(*args, **kwargs)

    def predict_face_keypoints(
        self,
        input_image: Image,
        raw_output: bool = False,
    ) -> np.ndarray | list[Image]:
        """
        Predict facial keypoints from an input image.

        Parameters
        ----------
        input_image
            Input image as a PIL Image.
        raw_output
            If True, returns raw keypoint coordinates.

        Returns
        -------
        keypoints_or_annotated_images : np.ndarray | list[Image]
            If raw_output is True, returns np.ndarray of shape [B, K, 2], where B is batch size,
            K is number of keypoints, and 2 is (x, y) coordinates scaled to the input image dimensions.
            If raw_output is False, returns list of PIL Images with keypoints drawn as red dots.
        """
        # Convert inputs to list of RGB uint8 frames and a torch tensor in [0,1], NCHW.
        rgb_frames_uint8, input_tensor = app_to_net_image_inputs(input_image)
        _, _, input_height, input_width = input_tensor.shape

        heatmaps = self.model(input_tensor)
        heatmap_array = heatmaps.numpy()

        # Heatmaps to keypoints
        _, _, heatmap_height, heatmap_width = heatmap_array.shape
        keypoints = refine_keypoints_from_heatmaps(heatmap_array)

        keypoints[..., 0] *= input_width / float(heatmap_width)  # Scale X coordinates
        keypoints[..., 1] *= input_height / float(heatmap_height)  # Scale Y coordinates

        if raw_output:
            return keypoints

        # Draw landmarks over the frames
        output_images = []
        for i, frame in enumerate(rgb_frames_uint8):
            draw_points(frame, keypoints[i], color=(255, 0, 0), size=2)
            output_images.append(fromarray(frame))
        return output_images


def refine_keypoints_from_heatmaps(
    heatmaps: np.ndarray,
) -> np.ndarray:
    """
    Extracts precise keypoint coordinates from heatmaps using argmax with sub-pixel refinement.

    Adapted from HRNet-Facial-Landmark-Detection:
    https://github.com/HRNet/HRNet-Facial-Landmark-Detection/blob/master/lib/core/evaluation.py#L67C5-L78C27

    Parameters
    ----------
    heatmaps
        [B, K, H, W] numpy float heatmaps.

    Returns
    -------
    keypoints : np.ndarray
        [B, K, 2] keypoints in heatmap coordinates (x, y).
    """
    B, K, H, W = heatmaps.shape
    reshaped = heatmaps.reshape(B, K, -1)
    idx = np.argmax(reshaped, axis=2)

    coords = np.tile(idx[..., None], (1, 1, 2)).astype(np.float32)
    coords[..., 0] = coords[..., 0] % W
    coords[..., 1] = np.floor(coords[..., 1] / W)

    for b in range(B):
        for k in range(K):
            x, y = int(coords[b, k, 0]), int(coords[b, k, 1])
            if 1 < x < W - 1 and 1 < y < H - 1:
                hm = heatmaps[b, k]
                dx = hm[y, x + 1] - hm[y, x - 1]
                dy = hm[y + 1, x] - hm[y - 1, x]
                coords[b, k, 0] += np.sign(dx) * 0.25
                coords[b, k, 1] += np.sign(dy) * 0.25
    coords += 0.5
    return coords
