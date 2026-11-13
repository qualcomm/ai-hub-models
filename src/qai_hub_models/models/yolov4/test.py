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
from qai_hub_models.models.yolov4.app import YoloV4TinyDetectionApp
from qai_hub_models.models.yolov4.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov4.demo import main as demo_main
from qai_hub_models.models.yolov4.external_repos.yolov4.nets.yolo import YoloBody
from qai_hub_models.models.yolov4.external_repos.yolov4.utils.utils_bbox import (
    DecodeBox,
)
from qai_hub_models.models.yolov4.model import (
    ANCHORS,
    ANCHORS_MASK,
    DEFAULT_HEIGHT,
    DEFAULT_WEIGHTS,
    DEFAULT_WIDTH,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NUM_OF_CLASSES,
    YoloV4Tiny,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/yolo_demo_output.png"
)


def test_numerical() -> None:
    """Verify QAI Hub decoding matches the upstream YOLOv4 decoder numerically."""
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    qaihm_model = YoloV4Tiny.from_pretrained(
        DEFAULT_WEIGHTS, include_postprocessing=False
    )
    source_model = YoloBody(ANCHORS_MASK, NUM_OF_CLASSES)
    source_model.load_state_dict(
        torch.load(DEFAULT_WEIGHTS.fetch(), map_location="cpu")
    )
    source_model.eval()

    with torch.no_grad():
        raw_outputs = source_model(processed_sample_image)

        # Decode with the upstream implementation for an independent reference.
        anchors = np.asarray(ANCHORS, dtype=np.float32)
        source_decoder = DecodeBox(
            anchors, NUM_OF_CLASSES, (DEFAULT_HEIGHT, DEFAULT_WIDTH), ANCHORS_MASK
        )
        source_detector_output = torch.cat(
            source_decoder.decode_box(raw_outputs), dim=1
        )
        source_postprocessed = detect_postprocess(source_detector_output)

        # YoloV4Tiny uses the same raw heads but exposes pixel-space xy/wh.
        qaihm_detector_output = qaihm_model(processed_sample_image).clone()
        qaihm_detector_output[:, :, :4] /= DEFAULT_WIDTH
        qaihm_postprocessed = detect_postprocess(qaihm_detector_output)

        for source, qaihm in zip(
            source_postprocessed, qaihm_postprocessed, strict=True
        ):
            np.testing.assert_allclose(
                source.numpy(), qaihm.numpy(), rtol=1e-5, atol=1e-5
            )


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS).convert("RGB")
    model = YoloV4Tiny.from_pretrained(DEFAULT_WEIGHTS)
    app = YoloV4TinyDetectionApp(model, input_spec=model.get_input_spec())
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


def test_demo() -> None:
    demo_main(is_test=True)
