# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from typing_extensions import Self

from qai_hub_models.models.templates.super_resolution.model import (
    DEFAULT_SCALE_FACTOR,
    SuperResolutionModel,
    validate_scale_factor,
)
from qai_hub_models.models.xlsr.external_repos.aimet_zoo.aimet_zoo_torch.common.super_resolution.models import (
    XLSRRelease,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
BASE_ASSET_URL = "https://github.com/quic/aimet-model-zoo/releases/download/phase_2_february_artifacts/xlsr_{scale_factor}x_checkpoint_float32.pth.tar"


class XLSR(SuperResolutionModel):
    """Exportable XLSR super resolution model, end-to-end."""

    @classmethod
    def from_pretrained(cls, scale_factor: int = DEFAULT_SCALE_FACTOR) -> Self:
        validate_scale_factor(scale_factor)
        model = XLSRRelease(scaling_factor=scale_factor)

        url = BASE_ASSET_URL.format(scale_factor=scale_factor)
        checkpoint_asset = CachedWebModelAsset(
            url,
            MODEL_ID,
            MODEL_ASSET_VERSION,
            Path(url).name,
        )
        checkpoint = load_torch(checkpoint_asset)
        model.load_state_dict(checkpoint["state_dict"])

        return cls(model, scale_factor)
