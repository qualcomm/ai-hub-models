# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image

from qai_hub_models.models.crestereo.app import CREStereoApp
from qai_hub_models.models.crestereo.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    CREStereo,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_model_input_spec_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_LEFT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "left.png"
)
DEFAULT_RIGHT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "right.png"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(CREStereo)
    parser = get_model_input_spec_parser(CREStereo, parser)
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

    model = demo_model_from_cli_args(CREStereo, MODEL_ID, args)
    input_spec = input_spec_from_cli_args(model, args)
    print("Model Loaded")

    left_orig = load_image(args.stereo_left)
    right_orig = load_image(args.stereo_right)

    app = CREStereoApp(model=model, input_spec=input_spec)  # type: ignore[arg-type]

    out_image = app.predict_disparity(left_orig, right_orig)
    assert isinstance(out_image, Image.Image)

    if not is_test:
        display_or_save_image(out_image, args.output_dir, "crestereo_disparity.png")


if __name__ == "__main__":
    main()
