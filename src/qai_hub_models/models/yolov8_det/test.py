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
from qai_hub_models.models.templates.yolo.model import yolo_detect_postprocess
from qai_hub_models.models.yolov8_det.app import YoloV8DetectionApp
from qai_hub_models.models.yolov8_det.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov8_det.demo import main as demo_main
from qai_hub_models.models.yolov8_det.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV8Detector,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.set_env import set_temp_env

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image_class_dependent_nms.png"
)
WEIGHTS = "yolov8n.pt"


def test_numerical() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    # Set the environment variable to force torch.load to use weights_only=False
    # This is needed for PyTorch 2.8+ where the default changed to weights_only=True
    with set_temp_env({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}):
        source_model = cast(DetectionModel, ultralytics_YOLO(WEIGHTS).model)
    qaihm_model = YoloV8Detector.from_pretrained(WEIGHTS)

    with torch.no_grad():
        # original model output
        source_detect_out, *_ = source_model(processed_sample_image)
        boxes, scores = torch.split(source_detect_out, [4, 80], 1)
        source_out_postprocessed = yolo_detect_postprocess(boxes, scores)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    app = YoloV8DetectionApp(YoloV8Detector.from_pretrained(WEIGHTS))
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


def test_demo() -> None:
    demo_main(is_test=True)
