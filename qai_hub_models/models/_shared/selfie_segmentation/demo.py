# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models._shared.selfie_segmentation.app import SelfieSegmentationApp
from qai_hub_models.models._shared.selfie_segmentation.model import SelfieSegmentor
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_input_spec_kwargs,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.display import display_or_save_image


def selfie_segmentation_demo(
    model_cls: type[SelfieSegmentor],
    model_id: str,
    default_image: str,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL.",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, model_id, args)
    validate_on_device_demo_args(args, model_id)

    # load image and model
    image = load_image(args.image)
    input_image = image.convert("RGB")
    input_spec = (
        model.get_input_spec(**get_input_spec_kwargs(model, args_dict=args.__dict__))
        if isinstance(model, SelfieSegmentor)
        else model.get_input_spec()
    )
    (_, _, height, width) = input_spec["image"][0]

    # Run app and display/save output
    app = SelfieSegmentationApp(model, (height, width), model_cls.MASK_THRESHOLD)  # type: ignore[arg-type]
    output = app.predict(input_image)[0]
    if not is_test:
        display_or_save_image(output, args.output_dir, "segmentation_demo_output.png")
