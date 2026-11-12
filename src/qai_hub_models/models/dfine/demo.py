# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image

from qai_hub_models.models.dfine.app import DFineApp
from qai_hub_models.models.dfine.model import (
    IMAGE_ADDRESS,
    MODEL_ID,
    DFine,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


# Run D-FINE app end-to-end on a sample image.
def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(DFine)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="test image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load image & model
    model = demo_model_from_cli_args(DFine, MODEL_ID, args)
    input_spec = model.get_input_spec()
    (h, w) = input_spec["image"][0][2:]

    img = load_image(args.image)
    app = DFineApp(model, h, w)  # type: ignore[arg-type]
    pred_images, _, _, _ = app.predict(img)
    pred_image = Image.fromarray(pred_images[0])

    if is_test:
        assert isinstance(pred_image, Image.Image)
    else:
        display_or_save_image(pred_image, args.output_dir)


if __name__ == "__main__":
    main()
