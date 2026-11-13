# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models import TargetRuntime
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.repaint.app import RepaintMaskApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


# Run repaint app end-to-end on a sample image.
# The demo will display the predicted image in a window.
def repaint_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset,
    default_mask: str | CachedWebAsset,
    is_test: bool = False,
    available_target_runtimes: list[TargetRuntime] | None = None,
) -> None:
    # Demo parameters
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=available_target_runtimes, add_output_dir=True
    )
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="test image file path or URL",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=default_mask,
        help="test mask file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    model = demo_model_from_cli_args(model_type, model_id, args)
    image = load_image(args.image)
    mask = load_image(args.mask)
    print("Model Loaded")

    # Run app
    app = RepaintMaskApp(model)  # type: ignore[arg-type]
    out = app.paint_mask_on_image(image, mask)[0]

    if not is_test:
        display_or_save_image(image, args.output_dir, "input_image.png", "input image")
        display_or_save_image(out, args.output_dir, "output_image.png", "output image")
