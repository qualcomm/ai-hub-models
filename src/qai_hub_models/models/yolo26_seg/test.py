# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import torch
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.yolo.app import YoloSegmentationApp
from qai_hub_models.models.templates.yolo.model import yolo_segment_postprocess
from qai_hub_models.models.yolo26_seg.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolo26_seg.demo import main as demo_main
from qai_hub_models.models.yolo26_seg.model import Yolo26Segmentor
from qai_hub_models.utils.asset_loaders import load_image

WEIGHTS = "yolo26n-seg.pt"


def test_task() -> None:
    qaihm_model = Yolo26Segmentor.from_pretrained(WEIGHTS)
    qaihm_app = YoloSegmentationApp(qaihm_model, nms_score_threshold=0.25)
    source_model = cast(SegmentationModel, ultralytics_YOLO(WEIGHTS).model)

    # YOLO26 has end2end=True by default, which applies NMS
    # We need to disable it to get raw outputs for comparison
    source_model.model[-1].end2end = False

    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    processed_sample_image = qaihm_app.preprocess_input(processed_sample_image)

    with torch.no_grad():
        # original model output
        source_out = source_model(processed_sample_image)[0]
        source_out_postprocessed = yolo_segment_postprocess(
            source_out[0], qaihm_model.num_classes
        )

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
