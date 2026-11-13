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
from qai_hub_models.models.yolov7.app import YoloV7DetectionApp
from qai_hub_models.models.yolov7.demo import IMAGE_ADDRESS
from qai_hub_models.models.yolov7.demo import main as demo_main
from qai_hub_models.models.yolov7.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV7,
    _load_yolov7_source_model_from_weights,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "yolov7_demo_640_output.png"
)
WEIGHTS = "yolov7-tiny.pt"


def test_numerical() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    source_model = _load_yolov7_source_model_from_weights(WEIGHTS)
    qaihm_model = YoloV7.from_pretrained(WEIGHTS)

    with torch.no_grad():
        # original model output
        source_model.model[-1].training = False  # type: ignore[index, union-attr]
        source_model.model[-1].export = False  # type: ignore[index, union-attr]
        source_detect_out = source_model(processed_sample_image)[0]
        source_out_postprocessed = detect_postprocess(source_detect_out)

        # Qualcomm AI Hub Model output
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(source_out_postprocessed[i], qaihm_out_postprocessed[i])


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS).convert("RGB")
    app = YoloV7DetectionApp(YoloV7.from_pretrained(WEIGHTS))
    assert np.allclose(app.predict_boxes_from_image(image)[0], np.asarray(output_image))


def test_demo() -> None:
    demo_main(is_test=True)


def test_app_with_larger_image() -> None:
    """Test that app resizes larger image to model size and transforms boxes back."""
    model = YoloV7.from_pretrained(WEIGHTS)
    app = YoloV7DetectionApp(model, input_spec=model.get_input_spec())

    # Resize image to be LARGER than model expects (1280x1280 > 640x640)
    image = load_image(IMAGE_ADDRESS).resize((1280, 1280))

    boxes, _, _ = app.predict_boxes_from_image(image, raw_output=True)

    # Boxes should be in 1280x1280 space (values > 640)
    expected_boxes = np.array(
        [
            [356.8037, 739.486, 673.6941, 953.0402],
            [17.9385, 704.0192, 378.3746, 987.5313],
            [1050.6981, 554.0536, 1279.8295, 890.3339],
            [692.2549, 489.1848, 959.5836, 916.5651],
            [512.4363, 382.6761, 879.8276, 746.683],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(boxes[0].numpy(), expected_boxes, rtol=1e-3)


def test_app_with_non_square_image() -> None:
    """Test that app correctly pads non-square images."""
    model = YoloV7.from_pretrained(WEIGHTS)
    app = YoloV7DetectionApp(model, input_spec=model.get_input_spec())

    # Non-square image (wider than tall) - requires padding
    image = load_image(IMAGE_ADDRESS).resize((1280, 640))

    boxes, _, _ = app.predict_boxes_from_image(image, raw_output=True)

    # Boxes should be in 1280x640 space
    expected_boxes = np.array(
        [
            [36.6308, 357.6668, 324.3835, 495.7377],
            [375.1452, 368.0038, 672.8099, 475.5621],
            [700.2011, 350.0212, 927.4102, 459.5838],
            [1058.0432, 297.4793, 1281.248, 422.134],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(boxes[0].numpy(), expected_boxes, rtol=1e-3)
