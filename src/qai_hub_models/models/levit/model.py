# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import LevitForImageClassification
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "facebook/levit-128S"


class LeViT(ImagenetClassifier):
    """Exportable LeViT model, end-to-end."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__(net=net)

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        model = LevitForImageClassification.from_pretrained(ckpt_name)
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for an input `image`.

        Parameters
        ----------
        image
            A [1, 3, 224, 224] image.
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        class_logits : torch.Tensor
            A [1, 1000] where each value is the log-likelihood of
            the image belonging to the corresponding Imagenet class.
        """
        predictions = self.net(normalize_image_torchvision(image), return_dict=False)
        return predictions[0]

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
