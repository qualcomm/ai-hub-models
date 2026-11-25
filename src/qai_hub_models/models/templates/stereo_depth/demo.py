# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.stereo_depth.app import StereoDisparityApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_model_input_spec_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


def stereo_disparity_demo(
    model_cls: type[BaseModel],
    app_cls: type[StereoDisparityApp],
    model_id: str,
    default_left_image: CachedWebModelAsset,
    default_right_image: CachedWebModelAsset,
    output_filename: str,
    is_test: bool = False,
) -> None:
    """Shared demo for stereo disparity models.

    Parameters
    ----------
    model_cls
        Model class to instantiate.
    app_cls
        StereoDisparityApp subclass to use.
    model_id
        Model ID string.
    default_left_image
        Default left stereo image asset.
    default_right_image
        Default right stereo image asset.
    output_filename
        Output filename for the disparity image.
    is_test
        If True, skip argument parsing and output saving.
    """
    parser = get_model_cli_parser(model_cls)
    parser = get_model_input_spec_parser(model_cls, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--stereo-left",
        type=str,
        default=str(default_left_image),
        help="Path of the left-view input image.",
    )
    parser.add_argument(
        "--stereo-right",
        type=str,
        default=str(default_right_image),
        help="Path of the right-view input image.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_cls, model_id, args)
    input_spec = input_spec_from_cli_args(model, args)

    left_orig = load_image(args.stereo_left)
    right_orig = load_image(args.stereo_right)

    app = app_cls(model=model, input_spec=input_spec)
    out_image = app.predict_disparity(left_orig, right_orig)
    assert isinstance(out_image, Image.Image)

    if not is_test:
        display_or_save_image(out_image, args.output_dir, output_filename)
