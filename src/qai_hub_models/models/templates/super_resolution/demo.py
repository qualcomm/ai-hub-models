# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import sys

from qai_hub_models import TargetRuntime
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.super_resolution.app import SuperResolutionApp
from qai_hub_models.models.templates.super_resolution.model import IMAGE_ADDRESS
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_model_input_spec_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


# Run Super Resolution end-to-end on a sample image.
# The demo will display both the input image and the higher resolution output.
def super_resolution_demo(
    model_cls: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset = IMAGE_ADDRESS,
    is_test: bool = False,
    available_target_runtimes: list[TargetRuntime] | None = None,
) -> None:
    # Demo parameters
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    parser = get_model_cli_parser(model_cls)
    parser = get_model_input_spec_parser(model_cls, parser)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        available_target_runtimes=available_target_runtimes,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    inference_model = demo_model_from_cli_args(
        model_cls,
        model_id,
        args,
    )
    image = load_image(args.image)
    input_spec = input_spec_from_cli_args(inference_model, args)

    # Make sure the input image is consistent with the model.
    # Since we are demonstrating super-resolution, we do not want to do any
    # implicit resampling.
    img_width, img_height = image.size
    input_img_shape = (img_height, img_width)
    model_img_shape = input_spec["image"][0][2:4]
    if input_img_shape != model_img_shape:
        print(
            f"Error: The input image is required to be {model_img_shape[1]}x{model_img_shape[0]} for the on-device demo ({img_width}x{img_height} provided)"
        )
        sys.exit(1)

    app = SuperResolutionApp(inference_model)  # type: ignore[arg-type]
    print("Model Loaded")
    pred_images = app.upscale_image(image)
    if not is_test:
        display_or_save_image(
            image, args.output_dir, "original_image.png", "original image"
        )
        display_or_save_image(
            pred_images[0], args.output_dir, "upscaled_image.png", "upscaled image"
        )
