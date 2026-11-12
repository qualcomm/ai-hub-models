# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.owl.app import OwlApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


def owl_detection_demo(
    model_type: type[BaseModel],
    model_id: str,
    hf_model_id: str,
    default_image: str | CachedWebAsset,
    default_text_queries: list[str],
    is_test: bool = False,
    default_score_threshold: float = 0.1,
) -> None:
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)

    parser.add_argument(
        "--image-path",
        type=str,
        default=default_image,
        help=(
            "Path or URL to the input image. "
            "Defaults to the bundled girl.png sample image."
        ),
    )
    parser.add_argument(
        "--text-queries",
        type=str,
        default=",".join(default_text_queries),
        help='Comma-separated text queries, e.g. "a photo of a cat,a photo of a remote".',
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
        default=0.7,
        help="Intersection over Union (IoU) threshold for NonMaximumSuppression",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=len(default_text_queries),
        help=(
            "Number of text queries the model is compiled for. "
            "Only used when --eval-mode on-device and no --hub-model-id is given "
            "(i.e. the model is compiled on-the-fly). "
            "Must equal the number of comma-separated entries in --text-queries."
        ),
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    text_queries = [q.strip() for q in args.text_queries.split(",") if q.strip()]

    model = demo_model_from_cli_args(model_type, model_id, args)

    # Run inference
    print(f"Running detection on: {args.image_path}")
    print(f"Text queries: {text_queries}")
    image = load_image(args.image_path)

    app = OwlApp(
        model=model,  # type: ignore[arg-type]
        hf_model_id=hf_model_id,
        nms_score_threshold=args.score_threshold,
        nms_iou_threshold=args.iou_threshold,
    )
    pred_images = app.predict_boxes_from_image(image, text_queries, False)
    pred_image = Image.fromarray(np.asarray(pred_images[0]))

    if is_test:
        assert isinstance(pred_image, Image.Image)
    else:
        display_or_save_image(pred_image, args.output_dir)
