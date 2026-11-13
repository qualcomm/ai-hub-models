# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL.Image import Image

from qai_hub_models.models.posenet_mobilenet.app import PosenetApp
from qai_hub_models.models.posenet_mobilenet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    PosenetMobilenet,
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

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "posenet_demo.jpg"
)


# The demo will display a image with the predicted keypoints.
def posenet_demo(model_cls: type[PosenetMobilenet], is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_model_input_spec_parser(model_cls, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image & model
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    input_spec = input_spec_from_cli_args(model, args)
    image = load_image(args.image)
    print("Model Loaded")

    # OnDeviceModel is underspecified to have a fully-baked type here
    app = PosenetApp(model, input_spec=input_spec)  # type: ignore[arg-type]
    keypoints = app.predict_pose_keypoints(image)
    assert isinstance(keypoints, Image)
    if not is_test:
        display_or_save_image(
            keypoints, args.output_dir, "posenet_demo_output.png", "keypoints"
        )


def main(is_test: bool = False) -> None:
    return posenet_demo(PosenetMobilenet, is_test)


if __name__ == "__main__":
    main()
