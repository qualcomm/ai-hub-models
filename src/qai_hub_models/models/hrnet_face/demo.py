# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from PIL.Image import Image

from qai_hub_models.models.hrnet_face.app import HRNetFaceApp
from qai_hub_models.models.hrnet_face.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    HRNetFace,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "hrnet_face_demo.jpg"
)


def hrnet_face_demo(model_cls: type[HRNetFace], is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)
    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)

    # Load image
    (_, _, height, width) = model_cls.from_pretrained().get_input_spec()["image"][0]
    orig_image = load_image(args.image)
    image = orig_image.resize((width, height))

    app = HRNetFaceApp(model)  # type: ignore[arg-type]
    output = app.predict_face_keypoints(image)[0]
    assert isinstance(output, Image)

    if not is_test:
        # Resize / unpad annotated image
        image_annotated = output.resize(orig_image.size)
        display_or_save_image(
            image_annotated, args.output_dir, "hrnetpose_demo_output.png", "keypoints"
        )


def main(is_test: bool = False) -> None:
    hrnet_face_demo(HRNetFace, is_test=is_test)


if __name__ == "__main__":
    main()
