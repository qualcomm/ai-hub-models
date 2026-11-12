# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torchvision import transforms

from qai_hub_models.models.templates.corevision.image_processing import (
    pil_resize_pad,
    undo_resize_pad,
)


class DepthEstimationApp:
    """
    This class is required to perform end to end inference for Depth Estimation

    The app uses 2 models:
        * Midas
        * DepthAnything

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run DepthAnything inference
        * Convert the depth into visual representation(heatmap) and return as image
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        input_height: int,
        input_width: int,
    ) -> None:
        self.model = model
        self.input_height = input_height
        self.input_width = input_width

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Image.Image | npt.NDArray[np.float32]:
        return self.estimate_depth(*args, **kwargs)

    def estimate_depth(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> Image.Image | npt.NDArray[np.float32]:
        """
        Estimates the depth at each point in an image and produces a heatmap.

        Parameters
        ----------
        image
            PIL Image to estimate depth.
        raw_output
            If set, returns the raw depth estimates instead of a heatmap.

        Returns
        -------
        depth_output : Image.Image | npt.NDArray[np.float32]
            A heatmap PIL Image or an np array of depth estimates.
            np array will be shape (h, w) where h, w are the dimensions of the input.
            np array will contain raw depth estimates, while PIL image will normalize
            the values and display them as an RGB image.
        """
        resized_image, scale, padding = pil_resize_pad(
            image, (self.input_height, self.input_width)
        )
        image_tensor = transforms.ToTensor()(resized_image).unsqueeze(0)
        prediction = self.model(image_tensor)
        prediction = undo_resize_pad(prediction, image.size, scale, padding)
        numpy_output = cast(npt.NDArray[np.float32], prediction.squeeze().cpu().numpy())
        if raw_output:
            return numpy_output
        heatmap = plt.cm.get_cmap("plasma")(numpy_output / numpy_output.max())[..., :3]
        return Image.fromarray((heatmap * 255).astype(np.uint8))
