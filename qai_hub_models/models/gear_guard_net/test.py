# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest

from qai_hub_models.models.gear_guard_net.app import GearGuardNetApp
from qai_hub_models.models.gear_guard_net.demo import main as demo_main
from qai_hub_models.models.gear_guard_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GearGuardNet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_raw_file,
)
from qai_hub_models.utils.bounding_box_processing import get_iou
from qai_hub_models.utils.testing import skip_clone_repo_check

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_image.jpg"
)
GROUND_TRUTH_RESULT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ground_truth.txt"
)


@skip_clone_repo_check
def test_task() -> None:
    app = GearGuardNetApp(GearGuardNet.from_pretrained(), nms_score_threshold=0.9)
    image = load_image(INPUT_IMAGE_ADDRESS.fetch())
    boxes, _, class_idx = app.predict_boxes_from_image(image, raw_output=True)
    boxes_pd, class_idx_pd = (
        boxes[0][1].numpy().astype(int),
        class_idx[0][1].numpy().astype(int),
    )
    gt = np.array(load_raw_file(GROUND_TRUTH_RESULT).split(), dtype=int)
    boxes_gt, class_idx_gt = gt[1:5], gt[0]
    assert class_idx_pd == class_idx_gt
    assert get_iou(boxes_pd, boxes_gt) > 0.5
    assert len(boxes[0]) == 2


@pytest.mark.trace
@skip_clone_repo_check
def test_trace() -> None:
    app = GearGuardNetApp(
        GearGuardNet.from_pretrained().convert_to_torchscript(), nms_score_threshold=0.9
    )
    image = load_image(INPUT_IMAGE_ADDRESS.fetch())
    boxes, _, class_idx = app.predict_boxes_from_image(image, raw_output=True)
    boxes_pd, class_idx_pd = (
        boxes[0][1].numpy().astype(int),
        class_idx[0][1].numpy().astype(int),
    )
    gt = np.array(load_raw_file(GROUND_TRUTH_RESULT).split(), dtype=int)
    boxes_gt, class_idx_gt = gt[1:5], gt[0]
    assert class_idx_pd == class_idx_gt
    assert get_iou(boxes_pd, boxes_gt) > 0.5


@skip_clone_repo_check
def test_demo() -> None:
    demo_main(is_test=True)
