# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    pil_resize_pad,
    torch_tensor_to_PIL_image,
    undo_resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec


class NAFSSRApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with NAFNet Stereo model.

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run inference
    """

    def __init__(
        self,
        model: Callable[..., torch.Tensor],
        input_spec: InputSpec,
        scale_factor: int,
    ) -> None:
        self.model = model
        (_, _, self.model_height, self.model_width) = input_spec["l_image"][0]
        self.scale_factor = scale_factor

    @overload
    def restore_image(
        self,
        l_image: Image,
        r_image: Image,
    ) -> tuple[Image, Image]: ...
    @overload
    def restore_image(
        self, l_image: Image, r_image: Image, raw_output: Literal[False]
    ) -> tuple[Image, Image]: ...
    @overload
    def restore_image(
        self, l_image: Image, r_image: Image, raw_output: Literal[True]
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def restore_image(
        self, l_image: Image, r_image: Image, raw_output: bool = False
    ) -> tuple[Image, Image] | tuple[np.ndarray, np.ndarray]:
        """
        Denoise/Deblur provided images.

        Parameters
        ----------
        l_image
            Left view PIL Image in RGB format.
        r_image
            Right view PIL Image in RGB format.
        raw_output
            Whether to return raw output or annotated image.

        Returns
        -------
        output: tuple[Image, Image] | tuple[np.ndarray, np.ndarray]
            If raw_output is False, returns a tuple of PIL Image object containing the
            left and right upscaled images with the same dimensions as the input.
            If raw_output is True, returns a tuple of numpy array of shape (1, C, H, W)
            containing the left and right raw output pixel values in RGB format.
        """
        # Run prediction

        resized_image_l, l_scale, l_padding = pil_resize_pad(
            l_image, (self.model_height, self.model_width)
        )
        resized_image_r, r_scale, r_padding = pil_resize_pad(
            r_image, (self.model_height, self.model_width)
        )

        _, left_NCHW_fp32_torch_frames = app_to_net_image_inputs(resized_image_l)
        _, right_NCHW_fp32_torch_frames = app_to_net_image_inputs(resized_image_r)
        pred_images = self.model(
            left_NCHW_fp32_torch_frames, right_NCHW_fp32_torch_frames
        )
        target_size = (
            l_image.size[0] * self.scale_factor,
            l_image.size[1] * self.scale_factor,
        )

        restored_image_l = undo_resize_pad(
            pred_images[0],
            target_size,
            l_scale,
            (l_padding[0] * self.scale_factor, l_padding[1] * self.scale_factor),
        )
        restored_image_r = undo_resize_pad(
            pred_images[1],
            target_size,
            r_scale,
            (r_padding[0] * self.scale_factor, r_padding[1] * self.scale_factor),
        )

        if raw_output:
            return (
                restored_image_l.detach().numpy(),
                restored_image_r.detach().numpy(),
            )

        return (
            torch_tensor_to_PIL_image(restored_image_l.squeeze(0)),
            torch_tensor_to_PIL_image(restored_image_r.squeeze(0)),
        )
