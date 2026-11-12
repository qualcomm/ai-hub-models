# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import PIL.Image
import torch
from PIL.Image import Image
from torchvision import transforms

from qai_hub_models.models.fcn_resnet50.model import NUM_CLASSES
from qai_hub_models.models.templates.corevision.draw import create_color_map
from qai_hub_models.models.templates.corevision.image_processing import (
    pil_resize_pad,
    pil_undo_resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec


def preprocess_image(image: Image) -> torch.Tensor:
    """
    Preprocesses images to be run through torch FCN segmenter
    as prescribed here:
    https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

    Parameters
    ----------
    image
        Input image to be run through the classifier model.

    Returns
    -------
    input_tensor : torch.Tensor
        Torch tensor to be directly passed to the model.
    """
    out_tensor: torch.Tensor = transforms.ToTensor()(image)
    return out_tensor.unsqueeze(0)


class FCN_ResNet50App:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with FCN_ResNet50.

    For a given image input, the app will:
        * Pre-process the image (resize + pad to model input shape)
        * Run image segmentation
        * Convert the raw output into probabilities using softmax
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

    def predict(self, image: Image, raw_output: bool = False) -> Image | np.ndarray:
        """
        From the provided image or tensor, segment the image

        Parameters
        ----------
        image
            A PIL Image in RGB format.
        raw_output
            If true, returns raw prediction masks. Otherwise returns segmented image.

        Returns
        -------
        output : Image | np.ndarray
            If raw_output is true:
                A list of predicted masks.
            Otherwise:
                Images with segmentation map overlaid with an alpha of 0.5.
        """
        orig_size = None
        if self.model_image_input_shape is not None:
            orig_size = image.size
            image, scale, padding = pil_resize_pad(image, self.model_image_input_shape)

        input_tensor = preprocess_image(image)
        output = self.model(input_tensor)
        output = output[0]
        predictions = output.cpu().numpy()

        if raw_output:
            return predictions

        color_map = create_color_map(NUM_CLASSES)
        blended = PIL.Image.blend(
            image, PIL.Image.fromarray(color_map[predictions]), 0.5
        )
        if orig_size is not None:
            blended = pil_undo_resize_pad(blended, orig_size, scale, padding)
        return blended
