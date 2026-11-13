# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from PIL.Image import Image
from torchvision import transforms

from qai_hub_models.models.templates.corevision.image_processing import (
    IMAGENET_TRANSFORM,
    normalize_image_transform,
)
from qai_hub_models.protocols import ExecutableModelProtocol


def preprocess_image(
    image: Image,
    normalize: bool = False,
    transform: transforms.Compose = IMAGENET_TRANSFORM,
) -> torch.Tensor:
    """
    Preprocesses images to be run through torch imagenet classifiers
    as prescribed here:
    https://pytorch.org/hub/pytorch_vision_resnet/

    Parameters
    ----------
    image
        Input image to be run through the classifier model.
    normalize
        Perform normalization to the standard imagenet mean and standard deviation.
    transform
        Torchvision transform to apply to the image before inference.
        Defaults to the standard 224x224 ImageNet transform.

    Returns
    -------
    preprocessed_tensor : torch.Tensor
        Torch tensor to be directly passed to the model.
    """
    out_tensor = transform(image)
    assert isinstance(out_tensor, torch.Tensor)
    if normalize:
        out_tensor = normalize_image_transform()(out_tensor)

    return out_tensor.unsqueeze(0)


class ImagenetClassifierApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an ImagenetClassifier.

    For a given image input, the app will:
        * Pre-process the image (resize and normalize)
        * Run Imagnet Classification
        * Convert the raw output into probabilities using softmax
    """

    def __init__(
        self,
        model: ExecutableModelProtocol,
        normalization_in_network: bool = True,
        transform: transforms.Compose = IMAGENET_TRANSFORM,
    ) -> None:
        """
        Parameters
        ----------
        model
            The imagenet classifier.
        normalization_in_network
            Whether the classifier normalizes the input using the standard imagenet mean and standard deviation.
            If false, the app will preform the normalization in a preprocessing step.
        transform
            Torchvision transform to apply to the image before inference.
            Defaults to the standard 224x224 ImageNet transform.
        """
        self.model = model
        self.normalization_in_network = normalization_in_network
        self.transform = transform

    def predict(self, image: Image) -> torch.Tensor:
        """
        From the provided image or tensor, predict probability distribution
        over the 1k Imagenet classes.

        Parameters
        ----------
        image
            A PIL Image in RGB format.

        Returns
        -------
        class_probabilities : torch.Tensor
            A (1000,) size torch tensor of probabilities, each one corresponding
            to a different Imagenet1K class.
        """
        input_tensor = preprocess_image(
            image, not self.normalization_in_network, self.transform
        )
        with torch.no_grad():
            output = self.model(input_tensor)
        return torch.softmax(output[0], dim=0)
