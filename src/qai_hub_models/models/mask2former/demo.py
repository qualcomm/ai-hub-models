# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from qai_hub_models.models.mask2former.app import Mask2FormerApp
from qai_hub_models.models.mask2former.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Mask2Former,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_model import BaseModel

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/input_image.jpg"
)


def mask2former_demo(
    model_type: type[BaseModel],
    model_id: str,
    default_image: CachedWebModelAsset,
    is_test: bool = False,
) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=default_image,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(model_type, model_id, args)
    validate_on_device_demo_args(args, model_id)

    (_, _, height, width) = model.get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    image = orig_image.resize((width, height))

    app = Mask2FormerApp(model)  # type: ignore[arg-type]
    print("Model Loaded")

    output = app.segment_image(image)[0]

    if not is_test:
        image_annotated = output.resize(orig_image.size)
        display_or_save_image(image_annotated, args.output_dir)


def main(is_test: bool = False) -> None:
    mask2former_demo(Mask2Former, MODEL_ID, INPUT_IMAGE_ADDRESS, is_test)


if __name__ == "__main__":
    main()
