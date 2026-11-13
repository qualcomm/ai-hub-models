# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image

from qai_hub_models.models.centernet_2d.app import CenterNet2DApp
from qai_hub_models.models.centernet_2d.model import IMAGE, MODEL_ID, CenterNet2D
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(CenterNet2D)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    args = parser.parse_args([] if is_test else None)

    inference_model = model_from_cli_args(CenterNet2D, args)
    model = demo_model_from_cli_args(CenterNet2D, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    (_, _, height, width) = inference_model.get_input_spec()["image"][0]
    # Load
    app = CenterNet2DApp(
        model,  # type: ignore[arg-type]
        inference_model.decode,
        height,
        width,
    )

    image = load_image(IMAGE.fetch())

    output_image = app.predict_2d_boxes_from_image(image)

    if not is_test:
        assert isinstance(output_image, Image.Image)
        # visualize images
        display_or_save_image(output_image, args.output_dir, "centernet_bbox.png")


if __name__ == "__main__":
    main()
