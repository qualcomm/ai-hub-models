# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


class DnCNNApp:
    """
    End-to-end application for image denoising using DnCNN.

    Handles preprocessing (convert to grayscale, normalize to [0,1])
    and postprocessing (convert back to uint8 image).
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model = model

    def denoise_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
    ) -> list[Image.Image]:
        """
        Denoise one or more images.

        Parameters
        ----------
        pixel_values_or_image
            PIL image, or list of PIL images, or
            numpy array (H W C x uint8) or (N H W C x uint8), or
            pyTorch tensor (N C H W x fp32, value range [0, 1]).
            RGB images are converted to grayscale for processing.

        Returns
        -------
        list[Image.Image]
            Denoised grayscale images.
        """
        if isinstance(pixel_values_or_image, Image.Image):
            pixel_values_or_image = [pixel_values_or_image]

        if isinstance(pixel_values_or_image, list):
            pixel_values_or_image = [img.convert("L") for img in pixel_values_or_image]

        _, input_tensor = app_to_net_image_inputs(pixel_values_or_image)

        output = self.model(input_tensor)

        results = []
        for i in range(output.shape[0]):
            out_arr = output[i].squeeze().detach().numpy()
            out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
            results.append(Image.fromarray(out_arr, mode="L"))
        return results
