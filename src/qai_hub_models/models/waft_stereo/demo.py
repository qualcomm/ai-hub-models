# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.waft_stereo.app import WaftStereoApp
from qai_hub_models.models.waft_stereo.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    WaftStereo,
    WaftStereoClassification,
    WaftStereoEncoder,
    WaftStereoRegression,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_LEFT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "left.jpg"
)
DEFAULT_RIGHT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "right.jpg"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(WaftStereo)
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
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, MODEL_ID)

    inference_model, (encoder, classification, regression) = (
        demo_model_components_from_cli_args(WaftStereo, MODEL_ID, args)
    )
    input_spec = inference_model.encoder.get_input_spec()
    if not is_test:
        print("Model Loaded")

    left_orig = load_image(args.stereo_left)
    right_orig = load_image(args.stereo_right)

    app = WaftStereoApp(
        encoder=cast(WaftStereoEncoder, encoder),
        classification=cast(WaftStereoClassification, classification),
        regression=cast(WaftStereoRegression, regression),
        input_spec=input_spec,
    )

    out_image = app.predict_disparity(left_orig, right_orig)
    assert isinstance(out_image, Image.Image)

    if not is_test:
        display_or_save_image(out_image, args.output_dir, "waft_stereo_disparity.png")


if __name__ == "__main__":
    main()
