# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from PIL import Image

from qai_hub_models.models.owlv2.app import OwlV2CollectionApp
from qai_hub_models.models.owlv2.model import (
    HF_MODEL_ID,
    IMAGE_SIZE,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    OwlV2,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_TEXT_QUERIES = ["a photo of a cup"]
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(OwlV2)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    parser.add_argument(
        "--image-path",
        type=str,
        default=IMAGE_ADDRESS,
        help=(
            "Path or URL to the input image. "
            "Defaults to the bundled owlvit sample image."
        ),
    )
    parser.add_argument(
        "--text-queries",
        type=str,
        default=",".join(DEFAULT_TEXT_QUERIES),
        help='Comma-separated text queries, e.g. "a photo of a cat,a photo of a remote".',
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Minimum confidence score to display a detection (default: 0.1).",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    text_queries = [q.strip() for q in args.text_queries.split(",") if q.strip()]

    (
        _,
        (
            vision_encoder,
            text_detector,
        ),
    ) = demo_model_components_from_cli_args(OwlV2, MODEL_ID, args)

    app = OwlV2CollectionApp(
        vision_encoder=vision_encoder,  # type: ignore[arg-type]
        text_detector=text_detector,  # type: ignore[arg-type]
        hf_model_id=HF_MODEL_ID,
        model_image_height=IMAGE_SIZE,
        model_image_width=IMAGE_SIZE,
        nms_score_threshold=args.score_threshold,
        nms_iou_threshold=args.iou_threshold,
    )

    print(f"Running detection on: {args.image_path}")
    print(f"Text queries: {text_queries}")
    image = load_image(args.image_path)

    pred_images = app.predict_boxes_from_image(image, text_queries, False)
    pred_image = Image.fromarray(np.asarray(pred_images[0]))

    if is_test:
        assert isinstance(pred_image, Image.Image)
    else:
        display_or_save_image(pred_image, args.output_dir)


if __name__ == "__main__":
    main()
