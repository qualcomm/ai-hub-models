# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.crestereo.app import CREStereoApp
from qai_hub_models.models.crestereo.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CREStereo,
)
from qai_hub_models.models.templates.stereo_depth.demo import stereo_disparity_demo
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

DEFAULT_LEFT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "left.png"
)
DEFAULT_RIGHT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "right.png"
)


def main(is_test: bool = False) -> None:
    stereo_disparity_demo(
        CREStereo,
        CREStereoApp,
        MODEL_ID,
        DEFAULT_LEFT_IMAGE,
        DEFAULT_RIGHT_IMAGE,
        "crestereo_disparity.png",
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
