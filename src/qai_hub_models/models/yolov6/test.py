# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.yolo.utils import detect_postprocess
from qai_hub_models.models.yolov6.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov6.demo import main as demo_main
from qai_hub_models.models.yolov6.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    WEIGHTS_PATH,
    YoloV6,
    _load_yolov6_source_model_from_weights,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image


def test_task() -> None:
    model_path = f"{WEIGHTS_PATH}{DEFAULT_WEIGHTS}"
    asset = CachedWebModelAsset(
        model_path, MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
    )

    # source model
    source_model = _load_yolov6_source_model_from_weights(asset)

    # Qualcomm AI Hub Model
    qaihm_model = YoloV6.from_pretrained()

    with torch.no_grad():
        # source model output
        processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
        source_detect_out = source_model(processed_sample_image)
        source_out_postprocessed = detect_postprocess(source_detect_out)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


def test_demo() -> None:
    demo_main(is_test=True)
