# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import DetectionModel

from qai_hub_models.models.lpdm.app import LicensePlateDetectorApp
from qai_hub_models.models.lpdm.demo import IMAGE_ADDRESS
from qai_hub_models.models.lpdm.demo import main as demo_main
from qai_hub_models.models.lpdm.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NMS_IOU_THRESHOLD,
    SCORE_THRESHOLD,
    LicensePlateDetector,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.yolo.model import yolo_detect_postprocess
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

NUM_CLASSES = 51

GOLDEN_OUTPUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_files/lpdm_output.npz"
)


def test_numerical() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    weights_path = str(DEFAULT_WEIGHTS.fetch())
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))

    source_model = cast(DetectionModel, ultralytics_YOLO(weights_path).model)
    qaihm_model = LicensePlateDetector.from_pretrained()

    with torch.no_grad():
        source_detect_out, *_ = source_model(processed_sample_image)
        boxes, scores = torch.split(source_detect_out, [4, NUM_CLASSES], 1)
        source_out_postprocessed = yolo_detect_postprocess(boxes, scores)

        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


def test_task() -> None:
    model = LicensePlateDetector.from_pretrained()
    image = load_image(IMAGE_ADDRESS)
    app = LicensePlateDetectorApp(
        model,
        nms_score_threshold=SCORE_THRESHOLD,
        nms_iou_threshold=NMS_IOU_THRESHOLD,
        input_spec=model.get_input_spec(),
    )
    boxes_list, scores_list, classes_list = app.predict_boxes_from_image(
        image, raw_output=True
    )

    golden = np.load(GOLDEN_OUTPUT.fetch())
    assert np.allclose(boxes_list[0].numpy(), golden["boxes"], atol=1e-3)
    assert np.allclose(scores_list[0].numpy(), golden["scores"], atol=1e-4)
    assert np.array_equal(classes_list[0].numpy(), golden["classes"])


def test_demo() -> None:
    demo_main(is_test=True)
