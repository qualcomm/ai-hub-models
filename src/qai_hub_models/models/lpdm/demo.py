# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, cast

import numpy as np
from PIL import Image as PILImage

from qai_hub_models.models.lpdm.app import LicensePlateDetectorApp
from qai_hub_models.models.lpdm.evaluator import decompose_plate
from qai_hub_models.models.lpdm.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    NMS_IOU_THRESHOLD,
    SCORE_THRESHOLD,
    LicensePlateDetector,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.corevision.draw import draw_box_from_xyxy
from qai_hub_models.models.templates.yolo.app import YoloObjectDetectionApp
from qai_hub_models.models.templates.yolo.demo import yolo_detection_demo
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "car.jpg"
)


def main(is_test: bool = False) -> None:
    def _output(app: YoloObjectDetectionApp, image: Any, args: Any) -> None:
        lpdm_app = cast(LicensePlateDetectorApp, app)
        boxes_list, _, classes_list = lpdm_app.predict_boxes_from_image(
            image, raw_output=True
        )

        plate = decompose_plate(boxes_list[0], classes_list[0])
        rendered: np.ndarray = np.array(image.convert("RGB"))
        for box in boxes_list[0]:
            draw_box_from_xyxy(
                rendered, box[0:2].int(), box[2:4].int(), color=(0, 255, 0), size=2
            )

        if not is_test:
            print(f"Plate Category : {plate.category_number or 'N/A'}")
            print(f"Plate Number   : {plate.plate_number or 'N/A'}")
            print(f"Plate State    : {plate.emirate or 'N/A'}")
            display_or_save_image(
                PILImage.fromarray(rendered), args.output_dir, "lpdm_demo_output.png"
            )

    yolo_detection_demo(
        model_type=LicensePlateDetector,
        model_id=MODEL_ID,
        app_type=LicensePlateDetectorApp,
        default_image=IMAGE_ADDRESS,
        default_score_threshold=SCORE_THRESHOLD,
        default_iou_threshold=NMS_IOU_THRESHOLD,
        is_test=is_test,
        output_fn=_output,
    )


if __name__ == "__main__":
    main()
