# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

from qai_hub_models.models.maskrcnn.app import MaskRCNNApp
from qai_hub_models.models.maskrcnn.demo import IMAGE_ADDRESS
from qai_hub_models.models.maskrcnn.demo import main as demo_main
from qai_hub_models.models.maskrcnn.model import MaskRCNN
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)


def run_source_model() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the source torchvision model and return predictions."""
    source_model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
    source_model.eval()

    # Modify parameters to match AI Hub model defaults
    source_model.roi_heads.score_thresh = 0.9
    source_model.roi_heads.nms_thresh = 0.5
    source_model.roi_heads.detections_per_img = 200
    source_model.rpn.nms_thresh = 0.7
    source_model.rpn._post_nms_top_n["testing"] = 200
    source_model.rpn._pre_nms_top_n["testing"] = 6000

    img = load_image(IMAGE_ADDRESS)
    processed_image = preprocess_PIL_image(img)

    with torch.no_grad():
        outputs = source_model(processed_image)

    # Extract predictions from first image in batch
    predictions = outputs[0]
    exp_boxes = predictions["boxes"]
    exp_scores = predictions["scores"]
    exp_labels = predictions["labels"]
    exp_masks = predictions["masks"]

    return exp_boxes, exp_scores, exp_labels, exp_masks


def test_task() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and torchvision) networks are similar."""
    exp_boxes, exp_scores, exp_labels, exp_masks = run_source_model()

    wrapper = MaskRCNN.from_pretrained()
    proposal_generator, roi_head = wrapper.proposal_generator, wrapper.roi_head
    img = load_image(IMAGE_ADDRESS)
    input_spec = wrapper.proposal_generator.get_input_spec()

    app = MaskRCNNApp(
        proposal_generator,
        roi_head,
        boxes_score_threshold=0.9,
        input_spec=input_spec,
    )
    result = app.predict(img, raw_output=True)
    assert isinstance(result, tuple), "Expected tuple output when raw_output=True"
    boxes, scores, labels, masks = result

    # Compare detections (boxes, scores, labels, masks)
    assert_most_close(
        np.asarray(exp_boxes, dtype=np.float32),
        np.asarray(boxes[0], dtype=np.float32),
        diff_tol=0.01,
        rtol=0.01,
        atol=0.01,
    )
    assert_most_close(
        np.asarray(exp_scores, dtype=np.float32),
        np.asarray(scores[0], dtype=np.float32),
        diff_tol=0.01,
        rtol=0.01,
        atol=0.01,
    )
    assert_most_close(
        np.asarray(exp_labels, dtype=np.float32),
        np.asarray(labels[0], dtype=np.float32),
        diff_tol=0.01,
        rtol=0.01,
        atol=0.01,
    )
    # Compare masks
    assert_most_close(
        np.asarray(exp_masks, dtype=np.float32),
        np.asarray(masks[0], dtype=np.float32),
        diff_tol=0.2,
        rtol=0.01,
        atol=0.01,
    )


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
