# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.netvlad.app import NetVLADApp
from qai_hub_models.models.netvlad.demo import QUERY_IMAGE, REFERENCE_IMAGE
from qai_hub_models.models.netvlad.demo import main as demo_main
from qai_hub_models.models.netvlad.model import NetVLAD
from qai_hub_models.utils.asset_loaders import load_image


def test_task() -> None:
    model = NetVLAD.from_pretrained()
    input_spec = model.get_input_spec()
    app = NetVLADApp(model, input_spec)

    query_img = load_image(QUERY_IMAGE)
    ref_img = load_image(REFERENCE_IMAGE)

    descriptors = app.predict_features(query_img, ref_img)
    assert descriptors.shape[0] == 2
    # VLAD output should be L2-normalized (norm ≈ 1).
    norms = np.linalg.norm(descriptors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


def test_demo() -> None:
    demo_main(is_test=True)
