# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    numpy_image_to_torch,
)
from qai_hub_models.utils.input_spec import InputSpec


class CREStereoApp:
    """End-to-end pipeline for the CREStereo disparity model."""

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        input_spec: InputSpec,
    ) -> None:
        _, _, self.height, self.width = input_spec["left_image"][0]
        self.model = model
        self.check_image_size()

    def predict(
        self,
        left: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        right: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        return self.predict_disparity(left, right, raw_output)

    @staticmethod
    def _disp_to_pil(
        disp_np: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
    ) -> Image.Image:
        """Min-max normalize a float32 disparity array and apply a colormap.

        Parameters
        ----------
        disp_np
            Float32 disparity array of shape ``(H, W)``.
        colormap
            OpenCV colormap constant (default: ``COLORMAP_INFERNO``).

        Returns
        -------
        Image.Image
            RGB PIL image with the colormap applied.
        """
        d_min, d_max = disp_np.min(), disp_np.max()
        if d_max <= d_min:
            norm = np.zeros_like(disp_np, dtype=np.uint8)
        else:
            norm = (
                ((disp_np - d_min) / (d_max - d_min) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
        colored_bgr = cv2.applyColorMap(norm, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(colored_rgb)

    def check_image_size(self) -> None:
        """Verify that the input spatial dimensions are both divisible by 8.

        CREStereo downsamples features to 1/8 resolution internally, so
        inputs whose height or width are not a multiple of 8 will produce
        misaligned feature maps.

        Raises
        ------
        AssertionError
            If ``H % 8 != 0`` or ``W % 8 != 0``.
        """
        assert self.height % 8 == 0, (
            f"Input height ({self.height}) must be divisible by 8."
        )
        assert self.width % 8 == 0, (
            f"Input width ({self.width}) must be divisible by 8."
        )

    def predict_disparity(
        self,
        left: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        right: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """
        Run the stereo disparity pipeline.

        Parameters
        ----------
        left
            Left-view image (any resolution). PIL Image, numpy HWC uint8,
            or torch NCHW float32 [0, 1].
        right
            Right-view image (same formats as left).
        raw_output
            If ``True``, return the raw float32 disparity array ``(H, W)``
            at original resolution. If ``False``, return a colorized
            ``PIL.Image``.

        Returns
        -------
        np.ndarray | Image.Image
            Raw float32 disparity ``(H, W)`` when ``raw_output=True``,
            otherwise a colorized ``PIL.Image`` at original resolution.
        """
        # Decode inputs to BGR uint8 HWC.
        left_bgr = app_to_net_image_inputs(left, image_layout="BGR", to_float=False)[0][
            0
        ]
        right_bgr = app_to_net_image_inputs(right, image_layout="BGR", to_float=False)[
            0
        ][0]

        assert left_bgr.shape == right_bgr.shape, (
            f"Left and right images must be the same size, got "
            f"{left_bgr.shape[:2]} and {right_bgr.shape[:2]}."
        )

        in_h, in_w = left_bgr.shape[:2]
        t = float(in_w) / float(self.width)

        imgL = cv2.resize(
            left_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        imgR = cv2.resize(
            right_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

        left_t = numpy_image_to_torch(imgL)
        right_t = numpy_image_to_torch(imgR)
        left_dw2_t = F.interpolate(
            left_t, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        right_dw2_t = F.interpolate(
            right_t, scale_factor=0.5, mode="bilinear", align_corners=True
        )

        with torch.no_grad():
            disp_tensor = self.model(
                left_t, right_t, left_dw2_t, right_dw2_t
            )  # (1, 1, H, W)

        disp_np = disp_tensor.squeeze().cpu().numpy().astype(np.float32)  # (H, W)
        disp_np = cv2.resize(disp_np, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

        if raw_output:
            return disp_np

        return self._disp_to_pil(disp_np)
