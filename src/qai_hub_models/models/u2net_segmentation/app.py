# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.duts.duts import preprocess_image_for_u2net
from qai_hub_models.models.templates.corevision.image_processing import (
    torch_tensor_to_PIL_image,
)
from qai_hub_models.utils.input_spec import InputSpec


class SegmentationU2NetApp:
    """
    End-to-end application for U²-Net salient object segmentation.

    Handles:
        - Preprocessing  : skimage resize → normalize by max → ImageNet mean/std
        - Model call     : returns d0 saliency map [0, 1]
        - Postprocessing : grayscale mask → resize to original size
    """

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        input_spec: InputSpec,
    ) -> None:
        self.model = model
        _, _, self.model_height, self.model_width = input_spec["image"][0]

    def postprocess(
        self,
        pred: torch.Tensor,
        original_size: tuple[int, int],
    ) -> Image.Image:
        """
        Convert model output saliency map to a PIL image resized to original dimensions.

        Parameters
        ----------
        pred
            Model output [1, 1, H, W] saliency map in range [0, 1].
        original_size
            (W, H) of the original input image to resize mask back to.

        Returns
        -------
        Image.Image
            Grayscale saliency mask resized to original image dimensions.
        """
        mask = pred.squeeze(0)[0:1]
        mask_pil = torch_tensor_to_PIL_image(mask)
        w_orig, h_orig = original_size
        return mask_pil.resize((w_orig, h_orig), Image.BILINEAR)

    def predict(
        self,
        image: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
    ) -> Image.Image:
        """
        Run end-to-end salient object segmentation.

        Parameters
        ----------
        image
            Input image — PIL Image, numpy array, torch tensor, or list of PIL Images.

        Returns
        -------
        Image.Image
            Grayscale saliency mask resized to input image dimensions.
            White = foreground, Black = background.
        """
        if isinstance(image, list):
            image = image[0]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                assert image.shape[0] == 1, (
                    f"Batch size must be 1, got {image.shape[0]}"
                )
                image = image.squeeze(0)
            assert image.dim() == 3, f"Expected [C,H,W] tensor, got shape {image.shape}"
            image = Image.fromarray(
                (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

        original_size = image.size  # (W, H)
        input_tensor = preprocess_image_for_u2net(
            image, self.model_height, self.model_width
        )
        output = self.model(input_tensor)
        return self.postprocess(output, original_size)
