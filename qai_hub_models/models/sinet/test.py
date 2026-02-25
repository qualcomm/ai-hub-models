# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.selfie_segmentation.test import (
    selfie_model_and_app_test_e2e,
)
from qai_hub_models.models.sinet.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.sinet.demo import main as demo_main
from qai_hub_models.models.sinet.model import MODEL_ASSET_VERSION, MODEL_ID, SINet
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.testing import skip_clone_repo_check

OUTPUT_IMAGE_LOCAL_PATH = "sinet_demo_output.png"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, OUTPUT_IMAGE_LOCAL_PATH
)


@skip_clone_repo_check
def test_task() -> None:
    selfie_model_and_app_test_e2e(
        SINet.from_pretrained(),
        load_image(INPUT_IMAGE_ADDRESS),
        load_image(OUTPUT_IMAGE_ADDRESS),
    )


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
