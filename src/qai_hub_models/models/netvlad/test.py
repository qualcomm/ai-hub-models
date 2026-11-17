# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.netvlad.app import NetVLADApp
from qai_hub_models.models.netvlad.demo import QUERY_IMAGE, REFERENCE_IMAGE
from qai_hub_models.models.netvlad.demo import main as demo_main
from qai_hub_models.models.netvlad.model import NetVLAD
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DISTRACTOR_IMAGE = CachedWebModelAsset.from_asset_store("osnet", 1, "person_b_0.jpg")


def test_task() -> None:
    model = NetVLAD.from_pretrained()
    input_spec = model.get_input_spec()
    app = NetVLADApp(model, input_spec)

    query_img = load_image(QUERY_IMAGE)
    ref_img = load_image(REFERENCE_IMAGE)
    distractor_img = load_image(DISTRACTOR_IMAGE)

    descriptors = app.predict_features(query_img, ref_img, distractor_img)
    assert descriptors.shape[0] == 3
    # VLAD output should be L2-normalized (norm ≈ 1).
    norms = np.linalg.norm(descriptors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # Reference shows the same scene as the query; the distractor does not.
    # Random or broken weights give ~1.0 similarity for every pair.
    sim_ref = float(descriptors[0] @ descriptors[1])
    sim_distractor = float(descriptors[0] @ descriptors[2])
    assert sim_ref > sim_distractor + 0.02
    assert sim_distractor < 0.95


def test_demo() -> None:
    demo_main(is_test=True)
