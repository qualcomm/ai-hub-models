# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import torch
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import DetectionModel

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.ultralytics.detect_patches import (
    patch_ultralytics_detection_head,
)
from qai_hub_models.models.templates.yolo.model import yolo_detect_postprocess
from qai_hub_models.models.yolov5.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov5.demo import main as demo_main
from qai_hub_models.models.yolov5.model import (
    DEFAULT_WEIGHTS,
    YoloV5,
)
from qai_hub_models.utils.asset_loaders import load_image


def test_task() -> None:
    # source model
    source_model = cast(DetectionModel, ultralytics_YOLO(DEFAULT_WEIGHTS).model)
    patch_ultralytics_detection_head(source_model)

    # Qualcomm AI Hub Model
    qaihm_model = YoloV5.from_pretrained()

    with torch.no_grad():
        # source model output
        processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
        source_detect_out = source_model(processed_sample_image)
        source_out_postprocessed = yolo_detect_postprocess(*source_detect_out)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


def test_demo() -> None:
    demo_main(is_test=True)
