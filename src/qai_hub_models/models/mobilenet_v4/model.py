# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.models.mobilenet_v4.external_repos.MobileNetV4_pytorch.mobilenet.mobilenetv4 import (
    MobileNetV4 as _MobileNetV4,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
NUM_CLASSES = 1000
EMBEDDING_DIM = 1280

VARIANTS = {
    "small": "MobileNetV4ConvSmall",
    "medium": "MobileNetV4ConvMedium",
    "large": "MobileNetV4ConvLarge",
}
DEFAULT_VARIANT = "medium"


class MobileNetV4(ImagenetClassifier):
    def __init__(self, net: nn.Module) -> None:
        super().__init__(net, normalize_input=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(EMBEDDING_DIM, NUM_CLASSES, bias=True)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = normalize_image_torchvision(image_tensor)
        # backbone returns list [x1..x5]; x[-1] is the pooled 1280-dim feature
        features = self.net(image_tensor)
        return self.linear(self.flatten(features[-1]))

    @classmethod
    def from_pretrained(
        cls, variant: str = DEFAULT_VARIANT, ckpt: str | None = None
    ) -> Self:
        if variant not in VARIANTS:
            raise ValueError(
                f"Unknown MobileNetV4 variant '{variant}'. "
                f"Valid variants: {', '.join(VARIANTS)}"
            )
        model_name = VARIANTS[variant]
        weights_filename = ckpt if ckpt is not None else model_name + ".pth"
        model = cls(_MobileNetV4(model_name))
        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, weights_filename
        ).fetch()
        state_dict = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=True
        )
        # Checkpoints store backbone weights under "backbone.*"; remap to "net.*"
        # to match the attribute name used by ImagenetClassifier.
        state_dict = {
            k.replace("backbone.", "net.", 1): v for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)
        return model
