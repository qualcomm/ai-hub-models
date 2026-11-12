# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.litehrnet.app import LiteHRNetApp
from qai_hub_models.models.litehrnet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    LiteHRNet,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IA_HELP_MSG = "More inferencer architectures for litehrnet can be found at https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/topdown_heatmap/coco"
IMAGE_LOCAL_PATH = "litehrnet_demo.png"
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, IMAGE_LOCAL_PATH
)


# Run LiteHRNet end-to-end on a sample image.
# The demo will display a image with the predicted keypoints.
def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(LiteHRNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    litehrnet_model = model_from_cli_args(LiteHRNet, args)
    hub_model = demo_model_from_cli_args(LiteHRNet, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image & model
    image = load_image(args.image)
    print("Model Loaded")

    app = LiteHRNetApp(
        hub_model,  # type: ignore[arg-type]
        litehrnet_model.inferencer,
    )
    keypoints = app.predict_pose_keypoints(image)[0]
    if not is_test:
        display_or_save_image(keypoints, args.output_dir, "litehrnet_demo_output.png")


if __name__ == "__main__":
    main()
