# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from omegaconf import DictConfig
from safetensors.torch import load_file
from typing_extensions import Self

from qai_hub_models.models.depth_anything_v3.external_repos.depth_anything_3.src.depth_anything_3.cfg import (
    create_object,
    load_config,
)
from qai_hub_models.models.depth_anything_v3.external_repos.depth_anything_3.src.depth_anything_3.registry import (
    MODEL_REGISTRY,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.depth_estimation.model import DepthEstimationModel
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://huggingface.co/depth-anything/DA3-SMALL/resolve/main/model.safetensors",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "model.safetensors",
)


class DepthAnythingV3(DepthEstimationModel):
    """Exportable DepthAnythingV3 Depth Estimation, end-to-end."""

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        super().__init__(model=model)

    @classmethod
    def from_pretrained(cls, ckpt: str | None = None) -> Self:
        """Load DepthAnythingV3 from a weightfile from Huggingface/Transfomers."""
        cfg = load_config(MODEL_REGISTRY["da3-small"])
        assert isinstance(cfg, DictConfig)
        model = create_object(cfg)

        if ckpt is None:
            state_dict = load_file(DEFAULT_WEIGHTS.fetch())
        else:
            state_dict = load_file(ckpt)

        if any(key.startswith("model.") for key in state_dict):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run DepthAnythingV3 on `image`, and produce a predicted depth.

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        depth : torch.Tensor
            Shape [batch, 1, 518, 518]
        """
        image = normalize_image_torchvision(image).unsqueeze(1)
        out = self.model(image)
        return out["depth"]

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 518,
        width: int = 518,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }
