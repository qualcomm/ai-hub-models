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

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.input_spec import InputSpec


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
        mask_threshold: float = 0.5,
        input_spec: InputSpec | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            A callable that takes in a image and outputs a segmentation mask.
        mask_threshold
            The threshold to use when generating the binary mask from the model's output.
        input_spec
            Model input spec. If None and model exposes ``get_input_spec()``,
            shape is auto-detected from the model.
        """
        self.model = model
        if input_spec is None and hasattr(model, "get_input_spec"):
            input_spec = model.get_input_spec()
        if input_spec is None:
            raise ValueError(
                "input_spec must be provided when model has no get_input_spec()"
            )
        _, _, h, w = input_spec["image"][0]
        self.img_shape: tuple[int, int] = (h, w)
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
        # Load & stretch image to the network input size (no letterboxing,
        # matching the official MediaPipe preprocessing).
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )
        orig_h, orig_w = (
            NCHW_fp32_torch_frames.shape[2],
            NCHW_fp32_torch_frames.shape[3],
        )
        resized_images = torch.nn.functional.interpolate(
            NCHW_fp32_torch_frames,
            size=self.img_shape,
            mode="bilinear",
            align_corners=False,
        )

        # Run the model, then resize the predicted mask to original image size.
        mask = self.model(resized_images)
        if not raw_output:
            # Smooth the contour at model resolution. Threshold first so
            # the blur operates on a symmetric 0/1 step — this keeps the
            # boundary in the correct position (no net expansion) while
            # reducing the waviness inherent in the 256x256 grid.
            binary = (mask > self.mask_threshold).float()
            mask = torch.nn.functional.avg_pool2d(
                binary, kernel_size=7, stride=1, padding=3
            )
        resized_mask = torch.nn.functional.interpolate(
            mask, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

        if raw_output:
            masks: list[np.ndarray] = []
            for frame_mask in resized_mask:
                face_map = (frame_mask > self.mask_threshold).int().numpy()
                masks.append(face_map)
            return masks

        # Supersampled anti-aliased mask (batched): upsample 4x,
        # threshold at high res, then downsample back. The downscale
        # creates natural 1px AA.
        h, w = resized_mask.shape[2], resized_mask.shape[3]
        hi_res = torch.nn.functional.interpolate(
            resized_mask,
            size=(h * 4, w * 4),
            mode="bilinear",
            align_corners=False,
        )
        binary_hi = (hi_res > self.mask_threshold).float()
        aa_masks = torch.nn.functional.interpolate(
            binary_hi,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        images: list[Image.Image] = []
        h, w = resized_mask.shape[2], resized_mask.shape[3]
        overlay = Image.new("RGB", (w, h), (68, 132, 255))
        for i, frame in enumerate(NHWC_int_numpy_frames):
            alpha = (aa_masks[i, 0].numpy() * 0.7 * 255).astype(np.uint8)
            images.append(
                Image.composite(overlay, Image.fromarray(frame), Image.fromarray(alpha))
            )

        return images
