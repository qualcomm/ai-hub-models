# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL.Image import Image

from qai_hub_models.models.nafssr.app import NAFSSRApp
from qai_hub_models.models.nafssr.model import (
    LR_IMAGE_L,
    LR_IMAGE_R,
    MODEL_ID,
    NAFSSR,
    SCALING_FACTOR,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


def nafssr_demo(
    model_cls: type[BaseModel],
    model_id: str,
    l_image: str | CachedWebAsset = LR_IMAGE_L,
    r_image: str | CachedWebAsset = LR_IMAGE_R,
    is_test: bool = False,
) -> None:
    # Demo parameters

    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
    )
    parser.add_argument(
        "--l_image",
        type=str,
        default=l_image,
        help="Left view image file path or URL.",
    )
    parser.add_argument(
        "--r_image",
        type=str,
        default=r_image,
        help="Right view image file path or URL.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    inference_model = demo_model_from_cli_args(
        model_cls,
        model_id,
        args,
    )
    input_spec = input_spec_from_cli_args(inference_model, args)

    app = NAFSSRApp(inference_model, input_spec, SCALING_FACTOR)  # type: ignore[arg-type]

    l_img = load_image(args.l_image)
    r_img = load_image(args.r_image)
    assert l_img.size == r_img.size, (
        "Both input images (left and right) should be of same shape"
    )

    # Run inference
    pred_images = app.restore_image(l_img, r_img)
    assert isinstance(pred_images[0], Image)
    assert isinstance(pred_images[1], Image)

    if not is_test:
        display_or_save_image(
            pred_images[0],
            args.output_dir,
            "upscaled_image_left.png",
            "upscaled left image",
        )

        display_or_save_image(
            pred_images[1],
            args.output_dir,
            "upscaled_image_right.png",
            "upscaled right image",
        )


def main(is_test: bool = False) -> None:
    nafssr_demo(NAFSSR, MODEL_ID, LR_IMAGE_L, LR_IMAGE_R, is_test=is_test)


if __name__ == "__main__":
    main()
