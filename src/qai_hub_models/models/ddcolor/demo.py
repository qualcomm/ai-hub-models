# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.ddcolor.app import DDColorApp
from qai_hub_models.models.ddcolor.model import MODEL_ASSET_VERSION, MODEL_ID, DDColor
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

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ddcolor_sample.jpeg"
)


def ddcolor_demo(
    model_cls: type[DDColor],
    default_image: CachedWebModelAsset,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_model_input_spec_parser(model_cls, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    input_spec = input_spec_from_cli_args(model, args)
    app = DDColorApp(model, input_spec=input_spec)  # type: ignore[arg-type]
    print("Model Loaded")

    image = load_image(args.image)

    output = app.colorize(image)

    if not is_test:
        display_or_save_image(output, args.output_dir)


def main(is_test: bool = False) -> None:
    ddcolor_demo(DDColor, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
