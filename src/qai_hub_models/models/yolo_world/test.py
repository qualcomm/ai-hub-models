# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import torch
from ultralytics.models import YOLOWorld
from ultralytics.nn.tasks import WorldModel

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.models.templates.yolo.model import yolo_detect_postprocess
from qai_hub_models.models.yolo_world.app import YoloWorldDetectionApp
from qai_hub_models.models.yolo_world.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolo_world.demo import main as demo_main
from qai_hub_models.models.yolo_world.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloWorld,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.test_helpers import assert_most_close

WEIGHTS = DEFAULT_WEIGHTS
PROMPT_TEXT = "bus,person"

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/bus_output.png"
)


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)
    prompt_list = [t.strip() for t in PROMPT_TEXT.split(",") if t.strip()]
    wrapper = YoloWorld.from_pretrained(WEIGHTS)
    app = YoloWorldDetectionApp(wrapper.detector, wrapper.text_encoder)
    assert_most_close(
        app.predict_boxes_from_image(image, prompt_text=prompt_list)[0],
        np.asarray(output_image),
        0.05,
        0.05,
        1,
    )


def test_numerical() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    _, nchw = app_to_net_image_inputs(load_image(IMAGE_ADDRESS))
    processed_sample_image, _, _ = resize_pad(nchw, (640, 640))
    prompt_list = [t.strip() for t in PROMPT_TEXT.split(",") if t.strip()]
    num_classes = len(prompt_list)
    yolo_world = YOLOWorld(WEIGHTS)
    yolo_world.set_classes(prompt_list)
    source_model = cast(WorldModel, yolo_world.model)
    txt_feats = source_model.get_text_pe(prompt_list).to(torch.float32)

    wrapper = YoloWorld.from_pretrained(WEIGHTS)
    qaihm_detector = wrapper.detector

    with torch.no_grad():
        # original model output
        source_detect_out, *_ = source_model(processed_sample_image)
        boxes, scores = torch.split(source_detect_out, [4, num_classes], 1)
        source_out_postprocessed = yolo_detect_postprocess(boxes, scores)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_detector(processed_sample_image, txt_feats)
        for i in range(len(source_out_postprocessed)):
            # Avoid strict equality check which fails here for einsum patch
            # allowing rtol=1e-4 for relative tolerance
            assert np.allclose(
                source_out_postprocessed[i],
                qaihm_out_postprocessed[i],
                rtol=1e-4,
                atol=1e-4,
            ), f"Output mismatch at index {i}"


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
