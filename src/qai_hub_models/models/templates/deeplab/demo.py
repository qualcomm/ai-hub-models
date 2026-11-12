# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.deeplab.app import DeepLabV3App
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_model_input_spec_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


def deeplabv3_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: str | CachedWebAsset,
    num_classes: int,
    is_test: bool,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_model_input_spec_parser(model_type, parser)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    inference_model = demo_model_from_cli_args(model_type, model_id, args)
    input_spec = input_spec_from_cli_args(inference_model, args)

    app = DeepLabV3App(
        inference_model,  # type: ignore[arg-type]
        num_classes=num_classes,
        input_spec=input_spec,
    )
    print("Model Loaded")

    image = load_image(args.image)
    image_annotated = app.predict(image, False)
    assert isinstance(image_annotated, Image.Image)

    if not is_test:
        display_or_save_image(
            image_annotated, args.output_dir, "annotated_image.png", "predicted image"
        )
