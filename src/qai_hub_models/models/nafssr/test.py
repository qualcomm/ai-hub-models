# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import warnings

import numpy as np

from qai_hub_models.models.nafssr.app import NAFSSRApp
from qai_hub_models.models.nafssr.demo import main as demo_main
from qai_hub_models.models.nafssr.model import (
    LR_IMAGE_L,
    LR_IMAGE_R,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NAFSSR,
    SCALING_FACTOR,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

SR_IMAGE_L = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/sr_img_l.png"
)
SR_IMAGE_R = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/sr_img_r.png"
)


def test_task() -> None:
    warnings.filterwarnings("ignore")
    l_img = load_image(LR_IMAGE_L)
    r_img = load_image(LR_IMAGE_R)
    output_image_l = load_image(SR_IMAGE_L)
    output_image_r = load_image(SR_IMAGE_R)

    model = NAFSSR.from_pretrained()
    app = NAFSSRApp(model, model.get_input_spec(), SCALING_FACTOR)
    app_output_image_l, app_output_image_r = app.restore_image(l_img, r_img)

    np.testing.assert_allclose(
        np.asarray(preprocess_PIL_image(app_output_image_l), dtype=np.float32),
        np.asarray(preprocess_PIL_image(output_image_l), dtype=np.float32),
        rtol=0.2,
        atol=0.01,
    )

    np.testing.assert_allclose(
        np.asarray(preprocess_PIL_image(app_output_image_r), dtype=np.float32),
        np.asarray(preprocess_PIL_image(output_image_r), dtype=np.float32),
        rtol=0.2,
        atol=0.01,
    )


def test_demo() -> None:
    demo_main(is_test=True)
