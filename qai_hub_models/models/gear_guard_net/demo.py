# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.gear_guard_net.demo import gear_guard_demo
from qai_hub_models.models.gear_guard_net.app import GearGuardNetApp
from qai_hub_models.models.gear_guard_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GearGuardNet,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_image.jpg"
)


def main(is_test: bool = False) -> None:
    gear_guard_demo(
        model_type=GearGuardNet,
        model_id=MODEL_ID,
        app_type=GearGuardNetApp,
        default_image=INPUT_IMAGE_ADDRESS,
        default_score_threshold=0.9,
        default_iou_threshold=0.5,
        output_filename="gear_guard_demo_output.png",
        is_test=is_test,
    )


if __name__ == "__main__":
    main()
