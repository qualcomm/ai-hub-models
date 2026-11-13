# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image

from qai_hub_models.models.stereonet.app import StereoNetApp
from qai_hub_models.models.stereonet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    StereoNet,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_LEFT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "left_image.png"
)
DEFAULT_RIGHT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "right_image.png"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(StereoNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--stereo-left",
        type=str,
        default=str(DEFAULT_LEFT_IMAGE),
        help="Path of the left-view input image.",
    )
    parser.add_argument(
        "--stereo-right",
        type=str,
        default=str(DEFAULT_RIGHT_IMAGE),
        help="Path of the right-view input image.",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=0.9,
        help="center-crop percentage to apply to disparity width",
    )
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, MODEL_ID)
    model = demo_model_from_cli_args(StereoNet, MODEL_ID, args)
    print("Model Loaded")

    left = load_image(args.stereo_left)
    right = load_image(args.stereo_right)

    h, w = StereoNet.from_pretrained().get_input_spec()["image"][0][2:]
    app = StereoNetApp(model, height=h, width=w)  # type: ignore[arg-type]
    out_image = app.predict_disparity(left, right, crop=args.crop)
    assert isinstance(out_image, Image.Image)

    if not is_test:
        display_or_save_image(out_image, args.output_dir, "stereonet_disparity.png")


if __name__ == "__main__":
    main()
