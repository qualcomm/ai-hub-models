# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import numpy as np
from PIL import Image

from qai_hub_models.models.eyegaze.app import EyeGazeApp
from qai_hub_models.models.eyegaze.model import MODEL_ASSET_VERSION, MODEL_ID, EyeGaze
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "input_image.png"
)


def eyegaze_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: CachedWebModelAsset,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL. Must be a grayscale eye crop.",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="left",
        choices=["left", "right"],
        help="eye side; if 'right', yaw is flipped per source evaluation",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_type, model_id, args)
    validate_on_device_demo_args(args, model_id)

    # Load and preprocess image
    (_, height, width) = model.get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    image = orig_image.resize((width, height))

    image_np = np.array(image.convert("L"))

    # Initialize app and run inference
    app = EyeGazeApp(model)  # type: ignore[arg-type]
    print("Model Loaded")
    output = cast(Image.Image, app.predict_gaze_angle(image_np, side=args.side))

    if not is_test:
        image_annotated = output.resize(orig_image.size)
        display_or_save_image(image_annotated, args.output_dir)


def main(is_test: bool = False) -> None:
    eyegaze_demo(EyeGaze, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
