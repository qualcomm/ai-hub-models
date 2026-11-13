# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.yolov5_face.app import YoloV5FaceApp
from qai_hub_models.models.yolov5_face.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    YoloV5Face,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    input_spec_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face.jpeg"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(YoloV5Face)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=IMAGE_ADDRESS,
        help="image file path or URL",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(YoloV5Face, MODEL_ID, args)
    input_spec = input_spec_from_cli_args(model, args)

    image = load_image(args.image)

    app = YoloV5FaceApp(model, input_spec=input_spec)  # type: ignore[arg-type]
    detections, annotated_image = app.predict(image)

    if not is_test:
        display_or_save_image(
            annotated_image,
            args.output_dir,
            "yolov5_face_output.png",
            "face detections",
        )
        print(f"Detected {len(detections)} face(s).")
        for i, det in enumerate(detections):
            print(
                f"  Face {i + 1}: box={det['box']}, score={det['score']:.3f}, "
                f"landmarks={det['landmarks']}"
            )


if __name__ == "__main__":
    main()
