# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from PIL.Image import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    torch_tensor_to_PIL_image,
)

SCALE = 4


class SuperResolutionApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with Super Resolution models.

    The app uses 1 model:
        * SuperResolution models

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run inference
        * post-process the image
        * display the input and output side-by-side
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> list[Image]:
        # See upscale_image.
        return self.upscale_image(*args, **kwargs)

    def upscale_image(
        self,
        pixel_values_or_image: torch.Tensor | Image | list[Image],
    ) -> list[Image]:
        """
        Upscale provided images.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        Returns
        -------
        upscaled_images : list[Image]
            A list of upscaled images (one for each input image).
        """
        _, NCHW_fp32_torch_frames = app_to_net_image_inputs(pixel_values_or_image)

        # Run prediction
        upscaled_images = self.model(NCHW_fp32_torch_frames)

        return [torch_tensor_to_PIL_image(img) for img in upscaled_images]
