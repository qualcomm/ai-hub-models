# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.hrnet_pose.app import HRNetPoseApp
from qai_hub_models.models.hrnet_pose.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HRNetPose,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hrnet_pose_demo.png"
)


# The demo will display a image with the predicted keypoints.
def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(HRNetPose)
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
    model = demo_model_from_cli_args(HRNetPose, MODEL_ID, args)
    image = load_image(args.image)
    print("Model Loaded")

    app = HRNetPoseApp(model)  # type: ignore[arg-type]
    keypoints = app.predict_pose_keypoints(image)[0]
    if not is_test:
        display_or_save_image(
            keypoints, args.output_dir, "hrnetpose_demo_output.png", "keypoints"
        )


if __name__ == "__main__":
    main()
