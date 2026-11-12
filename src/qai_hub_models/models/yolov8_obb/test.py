# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import torch
from ultralytics.models import YOLO as ultralytics_YOLO
from ultralytics.nn.tasks import OBBModel

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.ultralytics.obb_patches import (
    patch_ultralytics_obb_head,
)
from qai_hub_models.models.yolov8_obb.app import YoloV8OBBApp
from qai_hub_models.models.yolov8_obb.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov8_obb.demo import main as demo_main
from qai_hub_models.models.yolov8_obb.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV8OBB,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.set_env import set_temp_env

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image_obb.png"
)
WEIGHTS = "yolov8n-obb.pt"


def test_numerical() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    # YOLOv8-OBB standard input is 640*640
    image = load_image(IMAGE_ADDRESS)
    processed_sample_image = preprocess_PIL_image(image)

    # Set the environment variable to force torch.load to use weights_only=False
    with set_temp_env({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}):
        source_model = cast(OBBModel, ultralytics_YOLO(WEIGHTS).model)
    patch_ultralytics_obb_head(source_model)

    qaihm_model = YoloV8OBB.from_pretrained(
        WEIGHTS, include_postprocessing=False, split_output=True
    )

    with torch.no_grad():
        # Collect model source output
        source_out = source_model(processed_sample_image)

        if isinstance(source_out, (tuple, list)):
            src_boxes, src_angles, src_scores = source_out
        else:
            # Split Source Output: [Batch, 4+1+C, Anchors]
            # Box (4), Angle (1), Classes (15 for DOTAv1)
            num_classes = source_out.shape[1] - 5
            src_boxes, src_angles, src_scores = torch.split(
                source_out, [4, 1, num_classes], 1
            )

        # 2. Qualcomm AI Hub Model output
        # Returns tuple: (boxes, angles, scores)
        qaihm_boxes, qaihm_angles, qaihm_scores = qaihm_model(processed_sample_image)

        # 3. Compare Raw Outputs
        # We compare the split tensors directly
        # np.allclose keeps its default rtol=1e-5 unless rtol is passed explicitly,
        # so set both bounds here instead of letting a hidden default define them.
        assert np.allclose(src_boxes, qaihm_boxes, atol=2e-5, rtol=1e-4)
        assert np.allclose(src_angles, qaihm_angles, atol=2e-5, rtol=1e-4)
        assert np.allclose(src_scores, qaihm_scores, atol=2e-5, rtol=1e-4)


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    app = YoloV8OBBApp(YoloV8OBB.from_pretrained(WEIGHTS))
    results = app.predict_obb_from_image(image)

    assert len(results) > 0
    assert results[0] is not None

    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    result_array = np.asarray(results[0], dtype=np.int16)
    output_array = np.asarray(output_image, dtype=np.int16)
    diff = np.abs(result_array - output_array)
    assert float(diff.mean()) < 1.0
    assert float(np.count_nonzero(diff) / diff.size) < 0.01


def test_demo() -> None:
    demo_main(is_test=True)
