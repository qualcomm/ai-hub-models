# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from PIL.Image import Image

from qai_hub_models.models.fcn_resnet50.app import FCN_ResNet50App
from qai_hub_models.models.fcn_resnet50.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FCN_ResNet50,
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

# Demo image comes from https://github.com/pytorch/hub/raw/master/images/deeplab1.png
# and has had alpha channel removed for use as input
INPUT_IMAGE_LOCAL_PATH = "fcn_demo.png"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, INPUT_IMAGE_LOCAL_PATH
)


def fcn_resnet50_demo(model_cls: type[FCN_ResNet50], is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_model_input_spec_parser(model_cls, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    input_spec = input_spec_from_cli_args(model, args)

    app = FCN_ResNet50App(model, input_spec=input_spec)  # type: ignore[arg-type]
    print("Model Loaded")

    image = load_image(args.image)
    output = app.predict(image, False)
    assert isinstance(output, Image)

    if not is_test:
        display_or_save_image(output, args.output_dir, "fcn_demo_output.png")


def main(is_test: bool = False) -> None:
    return fcn_resnet50_demo(FCN_ResNet50, is_test=is_test)


if __name__ == "__main__":
    main()
