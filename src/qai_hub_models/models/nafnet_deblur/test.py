# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import warnings

import numpy as np

from qai_hub_models.models.nafnet_deblur.demo import main as demo_main
from qai_hub_models.models.nafnet_deblur.model import (
    IMAGE_ADDRESS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NafNetDeBlur,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.nafnet.app import NAFNetApp
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/deblurred_image.png"
)


def test_task() -> None:
    warnings.filterwarnings("ignore")
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)

    model = NafNetDeBlur.from_pretrained()
    app = NAFNetApp(model, model.get_input_spec())

    restored_image = app.restore_image(image)

    np.testing.assert_allclose(
        np.asarray(preprocess_PIL_image(restored_image), dtype=np.float32),
        np.asarray(preprocess_PIL_image(output_image), dtype=np.float32),
        rtol=0.2,
        atol=0.01,
    )


def test_demo() -> None:
    demo_main(is_test=True)
