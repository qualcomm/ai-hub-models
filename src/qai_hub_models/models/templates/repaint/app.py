# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    torch_tensor_to_PIL_image,
)
from qai_hub_models.models.templates.repaint.utils import preprocess_inputs


class RepaintMaskApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with AOTGAN.

    The app uses 1 model:
        * AOTGAN

    For a given image input, the app will:
        * pre-process the image
        * Run AOTGAN inference
        * Convert the output tensor into a PIL Image
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> list[Image]:
        # See paint_mask_on_image.
        return self.paint_mask_on_image(*args, **kwargs)

    @staticmethod
    def preprocess_inputs(
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        mask_pixel_values_or_image: torch.Tensor | np.ndarray | Image,
    ) -> dict[str, torch.Tensor]:
        return preprocess_inputs(pixel_values_or_image, mask_pixel_values_or_image)

    def paint_mask_on_image(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        mask_pixel_values_or_image: torch.Tensor | np.ndarray | Image,
    ) -> list[Image]:
        """
        Erases and repaints the source image[s] in the pixel values given by the mask.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout.
        mask_pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout.
            If one mask is provided, it will be used for every input image.

        Returns
        -------
        repainted_images : list[Image]
            A list of predicted images (one list element per batch).
        """
        inputs = self.preprocess_inputs(
            pixel_values_or_image, mask_pixel_values_or_image
        )
        out = self.model(inputs["image"], inputs["mask"])
        return [torch_tensor_to_PIL_image(img) for img in out]
