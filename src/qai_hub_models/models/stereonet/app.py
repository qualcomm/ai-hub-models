# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
    undo_resize_pad,
)


class StereoNetApp:
    """End-to-End Pipeline for the StereoNet disparity model."""

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        height: int = 786,
        width: int = 490,
    ) -> None:
        self.model = model
        self.height = height
        self.width = width

    def predict(self, *args: Any, **kwargs: Any) -> Image.Image | np.ndarray:
        return self.predict_disparity(*args, **kwargs)

    @staticmethod
    def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
        """Min-max normalize a float array and cast to uint8.

        Parameters
        ----------
        img:
            Input float array of arbitrary shape.

        Returns
        -------
        np.ndarray
            Array of the same shape with dtype ``uint8`` and values in
            ``[0, 255]``.  Returns an all-zero array when the input is
            constant.
        """
        img_min, img_max = img.min(), img.max()
        if img_max <= img_min:
            return np.zeros_like(img, dtype=np.uint8)
        out = (img - img_min) / (img_max - img_min)
        return (out * 255.0).clip(0, 255).astype(np.uint8)

    def predict_disparity(
        self,
        left: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        right: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        crop: float = 1.0,
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """
        Run the stereo disparity pipeline on a left/right image pair.

        Parameters
        ----------
        left
            Left-view PIL image.
        right
            Right-view PIL image.
        crop
            Fraction of width to keep via center-crop before visualization.
            Must be in ``(0, 1]``.
        raw_output
            If ``True``, return the raw float32 disparity array ``(H, W)``.
            If ``False``, return a uint8 ``PIL.Image`` visualization.

        Returns
        -------
        np.ndarray | Image.Image
            Raw float32 disparity array of shape ``(H, W)`` when
            ``raw_output=True``, otherwise a uint8 ``PIL.Image``
            visualization.
        """
        orig_img, left_t = app_to_net_image_inputs(left, image_layout="L")
        _, right_t = app_to_net_image_inputs(right, image_layout="L")

        left_t, scale, padding = resize_pad(left_t, (self.height, self.width))
        right_t, _, _ = resize_pad(right_t, (self.height, self.width))

        stack = torch.cat([left_t, right_t], dim=1)
        with torch.no_grad():
            disp_pred = self.model(stack)

        h, w = orig_img[0].shape
        disp_pred = undo_resize_pad(disp_pred, (w, h), scale, padding)
        disp = disp_pred.squeeze().numpy().astype(np.float32)

        if raw_output:
            return disp

        if crop != 1.0:
            crop_w = round(w * crop)
            w_start = (w - crop_w) // 2
            disp = disp[:, w_start : w_start + crop_w]

        return Image.fromarray(self._normalize_to_uint8(disp))
