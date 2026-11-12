# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models import Precision
from qai_hub_models.models.midas.external_repos.midas.hubconf import MiDaS_small
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.depth_estimation.model import DepthEstimationModel
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_torch,
)
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3

DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "midas_v21_small_256.pt",
)
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256


class Midas(DepthEstimationModel):
    """Exportable Midas depth estimation model."""

    def __init__(
        self,
        model: torch.nn.Module,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        normalize_input: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.normalize_input = normalize_input
        self.height = height
        self.width = width

    @classmethod
    def from_pretrained(
        cls,
        weights: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> Midas:
        model = MiDaS_small(pretrained=False)
        weights_dict = load_torch(weights)
        model.load_state_dict(weights_dict)
        return cls(model, height, width)

    def get_input_spec(
        self,
        batch_size: int = 1,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, self.height, self.width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Runs the model on an image tensor and returns a tensor of depth estimates

        Parameters
        ----------
        image
            A [1, 3, H, W] image.
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1] if self.normalize_input, else ~[-2.5, 2.5]
            3-channel Color Space: RGB

        Returns
        -------
        depth_estimates : torch.Tensor
            Tensor of depth estimates of size [1, H, W].
        """
        if self.normalize_input:
            image = normalize_image_torchvision(image)
        return self.model(image)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"
