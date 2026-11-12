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


class NAFNetApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with NAFNet models.

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run inference
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        if input_spec is None and hasattr(model, "get_input_spec"):
            input_spec = model.get_input_spec()
        if input_spec is None:
            raise ValueError(
                "input_spec must be provided when model has no get_input_spec()"
            )
        (_, _, self.model_height, self.model_width) = input_spec["image"][0]

    @overload
    def restore_image(
        self,
        image: Image,
    ) -> Image: ...

    @overload
    def restore_image(
        self,
        image: Image,
        raw_output: Literal[False],
    ) -> Image: ...

    @overload
    def restore_image(
        self,
        image: Image,
        raw_output: Literal[True],
    ) -> np.ndarray: ...

    def restore_image(
        self, image: Image, raw_output: bool = False
    ) -> Image | np.ndarray:
        """
        Denoise/Deblur provided images.

        Parameters
        ----------
        image
            A PIL Image in RGB format.
        raw_output
            Whether to return raw output or annotated image.

        Returns
        -------
        output: Image | np.ndarray
            If raw_output is False, returns a PIL Image object containing the
            restored/denoised image with the same dimensions as the input.
            If raw_output is True, returns a numpy array of shape (1, C, H, W)
            containing the raw output pixel values in RGB format.
        """
        # Run prediction

        resized_image, scale, padding = pil_resize_pad(
            image, (self.model_height, self.model_width)
        )

        _, NCHW_fp32_torch_frames = app_to_net_image_inputs(resized_image)
        restored_image_tensor = self.model(NCHW_fp32_torch_frames)
        restored_image_tensor = undo_resize_pad(
            restored_image_tensor, image.size, scale, padding
        )
        if raw_output:
            return restored_image_tensor.detach().numpy()

        return torch_tensor_to_PIL_image(restored_image_tensor.squeeze(0))
