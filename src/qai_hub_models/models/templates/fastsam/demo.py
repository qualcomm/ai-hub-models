# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.fastsam.app import FastSAMApp
from qai_hub_models.models.templates.fastsam.model import Fast_SAM
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebAsset,
    load_image,
)


def fastsam_demo(
    model_type: type[Fast_SAM],
    model_id: str,
    image_path: str | CachedWebAsset,
    is_test: bool,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=image_path,
        help="image file path or URL.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_type, model_id, args)
    app = FastSAMApp(model)  # type: ignore[arg-type]

    image = load_image(args.image)
    pred_image = cast(list[Image.Image], app.segment_image(image))[0]
    if not is_test:
        display_or_save_image(pred_image, args.output_dir, "output.jpg")
