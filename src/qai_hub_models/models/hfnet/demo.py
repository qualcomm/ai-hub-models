# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.hfnet.app import HFNetApp
from qai_hub_models.models.hfnet.model import MODEL_ASSET_VERSION, MODEL_ID, HFNet
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "query2.jpg",
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(HFNet)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE,
        help="Path or URL to input image.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=300,
        help="Number of top keypoints to draw.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)
    model = demo_model_from_cli_args(HFNet, MODEL_ID, args)
    image_shape = model.get_input_spec()["image"].shape
    input_size = (int(image_shape[2]), int(image_shape[1]))
    app = HFNetApp(model, input_size=input_size)

    input_image = load_image(args.image)

    output_image = app.predict_keypoint_overlay(input_image, topk=args.topk)

    if not is_test:
        out_name = f"{MODEL_ID}_query_keypoint_overlay.png"
        display_or_save_image(output_image, args.output_dir, out_name)


if __name__ == "__main__":
    main()
