# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.easyocr.app import EasyOCRApp
from qai_hub_models.models.easyocr.dataset import CocoTextCalibrationDataset
from qai_hub_models.models.easyocr.demo import INPUT_IMAGE_ADDRESS
from qai_hub_models.models.easyocr.demo import main as demo_main
from qai_hub_models.models.easyocr.model import (
    LANG_LIST,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    EasyOCR,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "gracewood_output.png"
)
CHINESE_INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "chinese.jpg"
)
ENGLISH_INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "english.png"
)


def test_cocotext_metadata_validation() -> None:
    dataset = CocoTextCalibrationDataset.__new__(CocoTextCalibrationDataset)
    dataset.metadata = {"anns": {}, "imgs": {}}
    with pytest.raises(ValueError, match="No eligible"):
        dataset._get_image_metadata()

    dataset.metadata = {
        "anns": {
            "1": {
                "image_id": 1,
                "legibility": "legible",
                "language": "english",
                "utf8_string": "text",
            }
        },
        "imgs": {"1": {"id": 1, "set": "train", "file_name": "../image.jpg"}},
    }
    assert dataset._get_image_metadata()[0]["file_name"] == "image.jpg"


def test_task() -> None:
    model = EasyOCR.from_pretrained(LANG_LIST)
    detector_shape = tuple(model.detector.get_input_spec()["image"][0][2:4])
    recognizer_shape = tuple(model.recognizer.get_input_spec()["image"][0][2:4])
    assert len(detector_shape) == 2 and len(recognizer_shape) == 2
    app = EasyOCRApp(
        model.detector,
        model.recognizer,
        detector_shape,
        recognizer_shape,
        model.lang_list,
    )
    image = load_image(INPUT_IMAGE_ADDRESS)
    results = app.predict_text_from_image(image)
    output_image = load_image(OUTPUT_IMAGE_ADDRESS)

    # Checking bounding box
    np.testing.assert_allclose(
        np.asarray(results[0][0]), np.asarray(output_image), atol=3
    )

    # Checking recognition text
    assert results[0][1][0] == "GRACEWOOD LN"
    assert results[0][1][1] == "PRIVATE ROAD"

    en_image = load_image(ENGLISH_INPUT_IMAGE_ADDRESS)
    en_results = app.predict_text_from_image(en_image)
    assert en_results[0][1][0] == "Reduce your risk of coronavirus infection:"
    assert en_results[0][1][1] == "Clean hands with soap and water"
    assert en_results[0][1][2] == "or alcohol-based hand rub"
    assert en_results[0][1][3] == "Cover nose and mouth when coughing and"
    assert en_results[0][1][4] == "sneezing with tissue or flexed elbow"
    assert en_results[0][1][5] == "Avoid close contact with anyone with"
    assert en_results[0][1][6] == "cold or flu-like symptoms"
    assert en_results[0][1][7] == "Thoroughly cook meat and eggs"
    assert en_results[0][1][8] == "No unprotected contact with live wild"
    assert en_results[0][1][9] == "or farm animals"
    assert en_results[0][1][10] == "World Health"
    assert en_results[0][1][11] == "Organization"

    ch_model = EasyOCR.from_pretrained(["ch_sim", "en"])
    app = EasyOCRApp(
        ch_model.detector,
        ch_model.recognizer,
        detector_shape,
        recognizer_shape,
        ch_model.lang_list,
    )
    image = load_image(CHINESE_INPUT_IMAGE_ADDRESS)
    results = app.predict_text_from_image(image)

    # Checking recognition text
    assert results[0][1][2] == "西"

    # If you do zero padding instead of background color padding,
    # the model predicts "Yuyuanl", where the "l" character comes from
    # The blackspace padding being interpreted as another character
    assert results[0][1][5] == "Yuyuan"


def test_demo() -> None:
    demo_main(is_test=True)
