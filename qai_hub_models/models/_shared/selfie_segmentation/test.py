# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
from PIL import Image

from qai_hub_models.models._shared.selfie_segmentation.app import SelfieSegmentationApp
from qai_hub_models.models._shared.selfie_segmentation.model import SelfieSegmentor


def selfie_model_and_app_test_e2e(
    model: SelfieSegmentor,
    input_image: np.ndarray | Image.Image,
    expected_output_image: np.ndarray | Image.Image,
) -> None:
    (_, _, height, width) = model.get_input_spec()["image"][0]
    app = SelfieSegmentationApp(model, (height, width), model.MASK_THRESHOLD)
    output_img = app.predict(input_image)[0]
    output = np.asarray(output_img, dtype=np.float32)
    expected_output = np.asarray(expected_output_image, np.float32)
    np.testing.assert_allclose(output, expected_output, rtol=0.1, atol=0.1)
