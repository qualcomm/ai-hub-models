# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np
import torch
from ultralytics.models import YOLOE as ultralytics_YOLOE
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.yolo.model import yolo_segment_postprocess
from qai_hub_models.models.yoloe_seg.app import YoloESegmentationApp
from qai_hub_models.models.yoloe_seg.demo import IMAGE_ADDRESS
from qai_hub_models.models.yoloe_seg.demo import main as demo_main
from qai_hub_models.models.yoloe_seg.model import (
    DEFAULT_WEIGHTS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloESegmentor,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

WEIGHTS = DEFAULT_WEIGHTS
PROMPT_TEXT = "bus,person"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/out_bus_person_with_mask.png"
)


def test_task() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    qaihm_model = YoloESegmentor.from_pretrained(PROMPT_TEXT, WEIGHTS)
    qaihm_app = YoloESegmentationApp(qaihm_model, nms_score_threshold=0.25)

    # Load source model and set the same text prompts
    source_yoloe = ultralytics_YOLOE(WEIGHTS)
    source_yoloe.set_classes([t.strip() for t in PROMPT_TEXT.split(",")])
    source_model = cast(SegmentationModel, source_yoloe.model)

    processed_sample_image = preprocess_PIL_image(load_image(IMAGE_ADDRESS))
    processed_sample_image = qaihm_app.preprocess_input(processed_sample_image)

    with torch.no_grad():
        # Original model output: source_model returns ((predictions, protos), ...)
        # source_out[0] is the raw detector tensor of shape [batch, 4+num_classes+32, num_anchors]
        source_out = source_model(processed_sample_image)[0]
        source_out_postprocessed = yolo_segment_postprocess(
            source_out[0], qaihm_model.num_classes
        )

        # Qualcomm AI Hub Model output: returns (boxes, scores, mask_coeffs, class_idx, mask_protos)
        qaihm_out_postprocessed = qaihm_model(processed_sample_image)

        # Compare boxes, scores, mask_coeffs, and class_idx (first 4 outputs)
        for i in range(len(source_out_postprocessed)):
            assert np.allclose(
                source_out_postprocessed[i].numpy(),
                qaihm_out_postprocessed[i].numpy(),
                rtol=1e-3,
                atol=1e-4,
            ), f"Output mismatch at index {i}"


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)


def test_app_with_larger_image() -> None:
    """Test that app resizes larger image to model size and resizes masks back."""
    model = YoloESegmentor.from_pretrained(PROMPT_TEXT, WEIGHTS)
    app = YoloESegmentationApp(
        model,
        nms_score_threshold=0.25,
        input_spec=model.get_input_spec(),
    )

    # Resize image LARGER than model expects (1280x1280 > 640x640)
    image = load_image(IMAGE_ADDRESS).resize((1280, 1280))

    _, _, masks, _ = cast(
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ],
        app.predict_segmentation_from_image_yoloe(image, raw_output=True),
    )

    # Masks should be resized back to original input dimensions.
    assert masks[0].shape == (5, 1280, 1280)

    # Per-detection mask area (sum of mask values) — sensitive to correct resize.
    expected_areas = np.array(
        [499340.0, 90312.0, 63024.0, 41176.0, 17192.0], dtype=np.float32
    )
    np.testing.assert_allclose(
        masks[0].sum(dim=(1, 2)).numpy(), expected_areas, rtol=1e-3
    )


def test_app_with_non_square_image() -> None:
    """Test that app correctly handles non-square images."""
    model = YoloESegmentor.from_pretrained(PROMPT_TEXT, WEIGHTS)
    app = YoloESegmentationApp(
        model,
        nms_score_threshold=0.25,
        input_spec=model.get_input_spec(),
    )

    # Non-square image (wider than tall) - requires padding
    image = load_image(IMAGE_ADDRESS).resize((1280, 640))

    _, _, masks, _ = cast(
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ],
        app.predict_segmentation_from_image_yoloe(image, raw_output=True),
    )

    # Masks should be resized back to original input dimensions [H, W].
    assert masks[0].shape == (5, 640, 1280)

    expected_areas = np.array(
        [249670.0, 45008.0, 31714.0, 20584.0, 8482.0], dtype=np.float32
    )
    np.testing.assert_allclose(
        masks[0].sum(dim=(1, 2)).numpy(), expected_areas, rtol=1e-3
    )
