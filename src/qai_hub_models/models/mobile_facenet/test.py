# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.mobile_facenet.app import MobileFaceNetApp
from qai_hub_models.models.mobile_facenet.demo import (
    INPUT_IMAGE_ADDRESS_1,
    INPUT_IMAGE_ADDRESS_2,
)
from qai_hub_models.models.mobile_facenet.demo import (
    main as demo_main,
)
from qai_hub_models.models.mobile_facenet.model import (
    DEFAULT_THRESHOLD,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MobileFaceNet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "results.npy"
)


def _assert_matches_expected(preds: tuple[bool, float, float]) -> None:
    expected = load_numpy(OUTPUT.fetch())
    same, angle, cosine = preds
    assert bool(same) == bool(expected[0])
    np.testing.assert_allclose(angle, expected[1], atol=0.5)
    np.testing.assert_allclose(cosine, expected[2], atol=0.005)


def test_task() -> None:
    model = MobileFaceNet.from_pretrained()
    app = MobileFaceNetApp.from_pretrained(model)
    app.threshold = DEFAULT_THRESHOLD

    img1 = load_image(INPUT_IMAGE_ADDRESS_1.fetch())
    img2 = load_image(INPUT_IMAGE_ADDRESS_2.fetch())

    _assert_matches_expected(app.predict_features(img1, img2))


def test_demo() -> None:
    demo_main(is_test=True)
