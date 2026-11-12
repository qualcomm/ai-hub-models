# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast

from PIL import Image

from qai_hub_models import TargetRuntime
from qai_hub_models.models.grounding_dino_t.app import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_NMS_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    GroundingDINOTCollectionApp,
)
from qai_hub_models.models.grounding_dino_t.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    GroundingDINOTCollection,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.display import display_or_save_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "demo.jpg"
)

DEFAULT_TEXT = "Mirror. plants. books"


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(GroundingDINOTCollection)
    parser = get_on_device_demo_parser(
        parser,
        available_target_runtimes=[
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.QNN_DLC,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
            TargetRuntime.TFLITE,
        ],
        add_output_dir=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="Path or URL to input image.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT,
        help='Grounding text prompt. Separate categories with " . ", e.g. "cat . dog ."',
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=DEFAULT_BOX_THRESHOLD,
        help=f"Minimum per-box confidence score to keep a detection (default {DEFAULT_BOX_THRESHOLD}).",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help=f"Minimum per-token score used when decoding the predicted phrase label (default {DEFAULT_TEXT_THRESHOLD}).",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=DEFAULT_NMS_THRESHOLD,
        help=f"IoU threshold above which overlapping detections are suppressed (default {DEFAULT_NMS_THRESHOLD}).",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    _, components = demo_model_components_from_cli_args(
        GroundingDINOTCollection, MODEL_ID, args
    )
    app = cast(
        "GroundingDINOTCollectionApp",
        GroundingDINOTCollectionApp.from_components(list(components), args.text),
    )

    image = load_image(args.image)
    annotated_frames, _, _, _ = app.predict(
        image,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
    )

    pred_image = Image.fromarray(annotated_frames[0])

    if not is_test:
        display_or_save_image(pred_image, args.output_dir)


if __name__ == "__main__":
    main()
