# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.u2net_segmentation.app import SegmentationU2NetApp
from qai_hub_models.models.u2net_segmentation.model import (
    IMAGE_ADDRESS,
    MODEL_ID,
    SegmentationU2Net,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SegmentationU2Net)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="Image file path or URL to run salient object segmentation on.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(SegmentationU2Net, MODEL_ID, args)
    input_spec = model.get_input_spec()
    app = SegmentationU2NetApp(model, input_spec)  # type: ignore[arg-type]

    if is_test:
        return

    image = load_image(args.image)
    output = app.predict(image)
    assert isinstance(output, Image.Image)
    display_or_save_image(output, args.output_dir)


if __name__ == "__main__":
    main()
