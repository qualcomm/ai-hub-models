# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from PIL import Image

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.depth_estimation.app import DepthEstimationApp
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel


# The demo will display a heatmap of the estimated depth at each point in the image.
def depth_estimation_demo(
    model_cls: type[BaseModel],
    model_id: str,
    default_image: CachedWebModelAsset,
    is_test: bool = False,
) -> None:
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_cls, model_id, args)
    validate_on_device_demo_args(args, model_id)

    # Load image
    (_, _, height, width) = model.get_input_spec()["image"][0]
    image = load_image(args.image)
    print("Model Loaded")

    app = DepthEstimationApp(
        model,  # type: ignore[arg-type]
        height,
        width,
    )
    heatmap_image = app.estimate_depth(image)
    assert isinstance(heatmap_image, Image.Image)

    if not is_test:
        # Resize / unpad annotated image
        display_or_save_image(
            heatmap_image, args.output_dir, "out_heatmap.png", "heatmap"
        )
