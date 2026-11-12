# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Literal

from PIL.Image import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.nafnet.app import NAFNetApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


def nafnet_demo(
    model_cls: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset,
    is_test: bool = False,
    task: Literal["denoise", "deblur"] = "denoise",
) -> None:
    # Demo parameters

    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    # Load image & model
    orig_image = load_image(args.image)

    inference_model = demo_model_from_cli_args(
        model_cls,
        model_id,
        args,
    )
    input_spec = input_spec_from_cli_args(inference_model, args)

    # Run inference
    app = NAFNetApp(inference_model, input_spec)  # type: ignore[arg-type]

    restored_image = app.restore_image(orig_image)
    assert isinstance(restored_image, Image)

    if not is_test:
        if task == "deblur":
            display_or_save_image(
                restored_image,
                args.output_dir,
                "deblurred_image.png",
                "deblurred image",
            )
        elif task == "denoise":
            display_or_save_image(
                restored_image, args.output_dir, "denoised_image.png", "denoised image"
            )
