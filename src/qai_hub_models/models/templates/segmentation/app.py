# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.templates.corevision.draw import create_color_map
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    pil_resize_pad,
    pil_undo_resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec


class SegmentationApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference for Segmentation.

    For a given image input, the app will:
        * pre-process the image (resize + pad to model input shape)
        * Run inference
        * Convert the output segmentation mask into a visual representation
        * Undo the resize/pad and overlay the segmentation mask onto the original image
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        if input_spec is not None:
            _, _, h, w = input_spec["image"][0]
            self.model_image_input_shape: tuple[int, int] | None = (h, w)
        else:
            self.model_image_input_shape = None

    def predict(self, *args: Any, **kwargs: Any) -> list[Image.Image] | np.ndarray:
        # See segment_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
        pad_mode: str = "constant",
    ) -> list[Image.Image] | np.ndarray:
        """
        Return the input image with the segmentation mask overlayed on it.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout.
        raw_output
            See "returns" doc section for details.
        pad_mode
            Padding mode for resize_pad when resizing is needed.

        Returns
        -------
        output : list[Image.Image] | np.ndarray
            If raw_output is False, returns:
                segmented_images : list[Image.Image]
                    Images with segmentation map overlaid with an alpha of 0.5.

            If raw_output is True, returns:
                masks : np.ndarray
                    A list of predicted masks.
        """
        orig_size = None
        if self.model_image_input_shape is not None and isinstance(
            pixel_values_or_image, Image.Image
        ):
            orig_size = pixel_values_or_image.size
            pixel_values_or_image, scale, padding = pil_resize_pad(
                pixel_values_or_image,
                self.model_image_input_shape,
                pad_mode=pad_mode,
            )

        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # pred_mask is downsampled
        pred_masks = self.model(NCHW_fp32_torch_frames)

        if isinstance(pred_masks, tuple):
            pred_masks = pred_masks[0]

        # Upsample pred mask to original image size
        # Need to upsample in the probability space, not in class labels
        pred_masks = F.interpolate(
            input=pred_masks,
            size=NCHW_fp32_torch_frames.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        if raw_output:
            return pred_masks.detach().numpy()

        # Create color map and convert segmentation mask to RGB image
        pred_mask_img = torch.argmax(pred_masks, 1)

        # Overlay the segmentation mask on the image. alpha=1 is mask only,
        # alpha=0 is image only.
        color_map = create_color_map(int(pred_mask_img.max().item()) + 1)
        out = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            blended = Image.blend(
                Image.fromarray(img_tensor),
                Image.fromarray(color_map[pred_mask_img[i]]),
                alpha=0.5,
            )
            if orig_size is not None:
                blended = pil_undo_resize_pad(blended, orig_size, scale, padding)
            out.append(blended)
        return out
