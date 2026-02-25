# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import numpy as np
import torch
from PIL import Image

from qai_hub_models.utils.draw import create_color_map
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
    undo_resize_pad,
)


class SelfieSegmentationApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with SINet.

    For a given image input, the app will:
        * Pre-process the image (normalize)
        * Run image segmentation
        * Blend the segmentation mask with the original image for visualization
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        img_shape: tuple[int, int],
        mask_threshold: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        model
            A callable that takes in a image and outputs a segmentation mask.
        img_shape
            The expected input image shape for the model as (height, width).
        mask_threshold
            The threshold to use when generating the binary mask from the model's output.
        """
        self.model = model
        self.img_shape = img_shape
        self.mask_threshold = mask_threshold

    @overload
    def predict(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
    ) -> list[Image.Image]: ...

    @overload
    def predict(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: Literal[False],
    ) -> list[Image.Image]: ...

    @overload
    def predict(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: Literal[True],
    ) -> list[np.ndarray]: ...

    def predict(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: bool = False,
    ) -> list[Image.Image] | list[np.ndarray]:
        """
        From the provided image or tensor, segment the image

        Parameters
        ----------
        pixel_values_or_image
            PIL image
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout.
        raw_output
            See "returns" doc section for details.

        Returns
        -------
        list[Image.Image] | list[np.ndarray]
            If raw_output is true, return:

            face_map
                Array of face mask predictions per pixel as 0 (background) or 1 ( face).
                Shape: (H, W)

            Otherwise, returns:
            segmented_image
                Input image with segmentation results blended on top.
        """
        # Load & Resize image to fix the network input size
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        source_image_size = (
            NCHW_fp32_torch_frames.shape[2],
            NCHW_fp32_torch_frames.shape[3],
        )
        resized_images, scales, paddings = resize_pad(
            NCHW_fp32_torch_frames, self.img_shape
        )

        # Run the model; resize the predicted mask to match the original image size
        mask = self.model(resized_images)
        resized_mask = undo_resize_pad(
            mask, (source_image_size[1], source_image_size[0]), scales, paddings
        )

        # Convert the mask into a PIL image and blend it with the original image for visualization.
        masks: list[np.ndarray] = []
        images: list[Image.Image] = []
        for frame, mask in zip(NHWC_int_numpy_frames, resized_mask, strict=False):
            face_map = (mask > self.mask_threshold).int().numpy()

            if raw_output:
                masks.append(face_map)
            else:
                color_map = create_color_map(num_classes=2)
                images.append(
                    Image.blend(
                        Image.fromarray(frame),
                        Image.fromarray(color_map[face_map[0]]),
                        alpha=0.5,
                    )
                )

        return masks if raw_output else images
