# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.rtmdet.app import RTMDetApp
from qai_hub_models.models.rtmdet.demo import main as demo_main
from qai_hub_models.models.rtmdet.model import MODEL_ASSET_VERSION, MODEL_ID, RTMDet
from qai_hub_models.models.templates.corevision.bounding_box_processing import get_iou
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store("yolov7", 1, "yolov7_demo_640.jpg")

GT_BOXES = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "rtmdet_boxes.npy"
)


def test_task() -> None:
    image = load_image(IMAGE_ADDRESS)
    app = RTMDetApp(RTMDet.from_pretrained(), nms_score_threshold=0.5)
    boxes = app.predict_boxes_from_image(image, raw_output=True)[0][0].numpy()
    boxes_gt = load_numpy(GT_BOXES)
    boxes = sorted(boxes, key=lambda box: box[0])  # type: ignore[assignment]
    boxes_gt = sorted(boxes_gt, key=lambda box: box[0])
    assert len(boxes) == len(boxes_gt)
    ious = [get_iou(box, box_gt) for box, box_gt in zip(boxes, boxes_gt, strict=False)]
    for iou in ious:
        assert iou > 0.95


def test_demo() -> None:
    demo_main(is_test=True)
