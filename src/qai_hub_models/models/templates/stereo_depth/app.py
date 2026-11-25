# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    colorize_disparity,
    numpy_image_to_torch,
)
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.input_spec import InputSpec


class StereoDisparityApp(ABC):
    """Base class for stereo disparity estimation models.

    Subclasses must implement ``_prepare_model_inputs`` to handle
    model-specific input preparation (e.g. RGB vs BGR, extra
    downsampled inputs).
    """

    def __init__(
        self,
        model: ExecutableModelProtocol,
        input_spec: InputSpec,
        image_layout: str = "RGB",
        align_divisor: int = 32,
    ) -> None:
        """
        Parameters
        ----------
        model
            Callable stereo model (torch module or on-device wrapper).
        input_spec
            Model input spec; used to read ``(H, W)`` from ``left_image``.
        image_layout
            Colour channel order passed to ``app_to_net_image_inputs``.
        align_divisor
            Input height and width must be multiples of this value.
        """
        _, _, self.height, self.width = input_spec["left_image"][0]
        self.model = model
        self.image_layout = image_layout
        self.align_divisor = align_divisor
        self._check_image_size()

    def _check_image_size(self) -> None:
        """Verify input spatial dimensions are divisible by ``align_divisor``."""
        assert self.height % self.align_divisor == 0, (
            f"Input height ({self.height}) must be divisible by {self.align_divisor}."
        )
        assert self.width % self.align_divisor == 0, (
            f"Input width ({self.width}) must be divisible by {self.align_divisor}."
        )

    @abstractmethod
    def _prepare_model_inputs(
        self,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Prepare the tuple of tensors to pass to the model.

        Parameters
        ----------
        left_t
            Left image NCHW float32 tensor, normalised to ``[0, 1]``.
        right_t
            Right image NCHW float32 tensor, normalised to ``[0, 1]``.

        Returns
        -------
        tuple[torch.Tensor, ...]
            All positional arguments forwarded to ``self.model``.
        """

    def predict_disparity(
        self,
        left: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        right: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """Run the stereo disparity pipeline.

        Parameters
        ----------
        left
            Left-view image (any resolution). PIL Image, numpy HWC uint8,
            or torch NCHW float32 ``[0, 1]``.
        right
            Right-view image (same formats as ``left``).
        raw_output
            If ``True``, return the raw float32 disparity array ``(H, W)``
            rescaled to the original input resolution.
            If ``False``, return a colorized ``PIL.Image``.

        Returns
        -------
        np.ndarray | Image.Image
            Raw float32 disparity ``(H, W)`` when ``raw_output=True``,
            otherwise a colorized ``PIL.Image`` at original resolution.
        """
        img = app_to_net_image_inputs(
            left, image_layout=self.image_layout, to_float=False
        )[0][0]
        img_r = app_to_net_image_inputs(
            right, image_layout=self.image_layout, to_float=False
        )[0][0]

        assert img.shape == img_r.shape, (
            f"Left and right images must be the same size, got "
            f"{img.shape[:2]} and {img_r.shape[:2]}."
        )

        in_h, in_w = img.shape[:2]
        t = float(in_w) / float(self.width)

        imgL = cv2.resize(
            img, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        imgR = cv2.resize(
            img_r, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )

        left_t = numpy_image_to_torch(imgL)
        right_t = numpy_image_to_torch(imgR)

        with torch.no_grad():
            model_inputs = self._prepare_model_inputs(left_t, right_t)
            disp_tensor = self.model(*model_inputs)

        disp_np = disp_tensor.squeeze().cpu().numpy().astype(np.float32)
        disp_np = cv2.resize(disp_np, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

        if raw_output:
            return disp_np

        return colorize_disparity(disp_np)
