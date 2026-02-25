# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.display import display_or_save_image


def gear_guard_demo(
    model_type: type[BaseModel],
    model_id: str,
    app_type: Callable,
    default_image: str | CachedWebAsset,
    default_score_threshold: float,
    default_iou_threshold: float,
    output_filename: str,
    is_test: bool = False,
) -> None:
    """
    Shared demo function for GearGuardNet models.

    Parameters
    ----------
    model_type
        The model class (e.g., GearGuardNet, GearGuardNetV2).
    model_id
        The model ID string.
    app_type
        The app class to use (e.g., GearGuardNetApp, PPEDetectionApp).
    default_image
        Default image path or URL for the demo.
    default_score_threshold
        Default score threshold for NMS.
    default_iou_threshold
        Default IoU threshold for NMS.
    output_filename
        Name of the output file to save.
    is_test
        Whether this is a test run.
    """
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=default_score_threshold,
        help="Score threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=default_iou_threshold,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)

    app = app_type(
        model,
        nms_score_threshold=args.score_threshold,
        nms_iou_threshold=args.iou_threshold,
    )

    print("Model Loaded")
    image = load_image(args.image)
    pred_images = app.predict_boxes_from_image(image)
    assert isinstance(pred_images[0], np.ndarray)
    out = Image.fromarray(pred_images[0])
    if not is_test:
        display_or_save_image(out, args.output_dir, output_filename)
