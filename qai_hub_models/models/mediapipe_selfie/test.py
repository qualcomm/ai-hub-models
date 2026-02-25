# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.selfie_segmentation.test import (
    selfie_model_and_app_test_e2e,
)
from qai_hub_models.models.mediapipe_selfie.demo import IMAGE_ADDRESS
from qai_hub_models.models.mediapipe_selfie.demo import main as demo_main
from qai_hub_models.models.mediapipe_selfie.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MediapipeSelfie,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "selfie_output.png"
)


def test_output() -> None:
    selfie_model_and_app_test_e2e(
        MediapipeSelfie.from_pretrained(),
        load_image(IMAGE_ADDRESS),
        load_image(OUTPUT_IMAGE_ADDRESS),
    )


def test_demo() -> None:
    demo_main(is_test=True)
