# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.crestereo.app import CREStereoApp
from qai_hub_models.models.crestereo.demo import DEFAULT_LEFT_IMAGE, DEFAULT_RIGHT_IMAGE
from qai_hub_models.models.crestereo.demo import main as demo_main
from qai_hub_models.models.crestereo.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CREStereo,
)
from qai_hub_models.models.templates.stereo_depth.test_utils import (
    run_stereo_disparity_test,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "output.png"
)


def test_task() -> None:
    model = CREStereo.from_pretrained()
    input_spec = CREStereo.from_pretrained().get_input_spec(height=720, width=1280)
    app = CREStereoApp(model, input_spec=input_spec)
    run_stereo_disparity_test(
        app, DEFAULT_LEFT_IMAGE, DEFAULT_RIGHT_IMAGE, OUTPUT_IMAGE_ADDRESS
    )


def test_demo() -> None:
    demo_main(is_test=True)
