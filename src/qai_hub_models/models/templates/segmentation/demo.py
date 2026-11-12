# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.segmentation.app import SegmentationApp
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


def segmentation_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: CachedWebModelAsset,
    is_test: bool = False,
    pad_mode: str = "constant",
    app_cls: type[SegmentationApp] = SegmentationApp,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_model_input_spec_parser(model_type, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_type, model_id, args)
    validate_on_device_demo_args(args, model_id)

    input_spec = input_spec_from_cli_args(model, args)

    app = app_cls(
        model,  # type: ignore[arg-type]
        input_spec=input_spec,
    )
    print("Model Loaded")

    image = load_image(args.image)
    output = app.segment_image(image, pad_mode=pad_mode)[0]

    if not is_test:
        display_or_save_image(output, args.output_dir)
