# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from PIL.Image import Image
from transformers import MobileViTFeatureExtractor

from qai_hub_models.models.mobile_vit.model import (
    DEFAULT_WEIGHTS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MobileVIT,
)


class MobileVITApp:
    """Encapsulates the logic for running inference on a MobileVIT model."""

    def __init__(
        self,
        model: MobileVIT,
        feature_extractor: MobileViTFeatureExtractor | None = None,
    ) -> None:
        self.model = model
        if feature_extractor is None:
            # Traced and on-device models do not expose the feature extractor.
            feature_extractor = getattr(model, "feature_extractor", None)
        self.feature_extractor = (
            feature_extractor
            if feature_extractor is not None
            else self._default_feature_extractor()
        )

    @staticmethod
    def _default_feature_extractor() -> MobileViTFeatureExtractor:
        feature_extractor = MobileViTFeatureExtractor.from_pretrained(DEFAULT_WEIGHTS)
        feature_extractor.size = {"height": INPUT_HEIGHT, "width": INPUT_WIDTH}
        return feature_extractor

    def predict(self, image: Image) -> torch.Tensor:
        feature = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(feature.pixel_values)
        return torch.softmax(logits[0], dim=0)
