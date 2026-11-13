# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.foot_track_net.app import FootTrackNet_App
from qai_hub_models.models.foot_track_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FootTrackNet,
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

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test1.jpg"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(FootTrackNet)
    parser = get_model_input_spec_parser(FootTrackNet, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(FootTrackNet, MODEL_ID, args)
    input_spec = input_spec_from_cli_args(model, args)
    # Load image
    print("Model Loaded")
    app = FootTrackNet_App(
        model,  # type: ignore[arg-type]
        input_spec=input_spec,
    )
    image_out = app.predict_and_draw_bbox_landmarks(load_image(args.image))
    if not is_test:
        display_or_save_image(
            image_out, args.output_dir, "FootTrackNet_demo_output.png"
        )


if __name__ == "__main__":
    main()
