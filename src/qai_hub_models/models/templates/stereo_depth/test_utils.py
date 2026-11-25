# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.templates.stereo_depth.app import StereoDisparityApp
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import assert_most_close


def run_stereo_disparity_test(
    app: StereoDisparityApp,
    left_image: CachedWebModelAsset,
    right_image: CachedWebModelAsset,
    expected_output: CachedWebModelAsset,
) -> None:
    """Shared test helper for stereo disparity models.

    Parameters
    ----------
    app
        StereoDisparityApp subclass instance wrapping the model.
    left_image
        Left stereo image asset.
    right_image
        Right stereo image asset.
    expected_output
        Expected disparity colormap image asset.
    """
    left = load_image(left_image)
    right = load_image(right_image)
    exp_disp = load_image(expected_output)

    disp = app.predict_disparity(left, right, raw_output=False)
    assert_most_close(
        np.asarray(disp),
        np.asarray(exp_disp),
        diff_tol=5e-3,
        rtol=1e-3,
        atol=1e-3,
    )
