# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.mobile_facenet.app import MobileFaceNetApp
from qai_hub_models.models.mobile_facenet.model import (
    DEFAULT_THRESHOLD,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    MobileFaceNet,
)
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS_1 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "mobilefacenet_demo_input_1.jpg"
)
INPUT_IMAGE_ADDRESS_2 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "mobilefacenet_demo_input_2.jpg"
)


def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(MobileFaceNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image1",
        type=str,
        default=INPUT_IMAGE_ADDRESS_1.fetch(),
        help="image file path or URL",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=INPUT_IMAGE_ADDRESS_2.fetch(),
        help="second image file path or URL",
    )
    parser.add_argument(
        "--already_aligned",
        action="store_true",
        help="whether the image is already aligned or not",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Cosine-angle threshold in degrees (default: 74.18). Below = same person.",
    )

    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    _, (face_detector, face_recognizer) = demo_model_components_from_cli_args(
        MobileFaceNet, MODEL_ID, args
    )
    app = MobileFaceNetApp(
        face_detector=face_detector,
        face_recognizer=face_recognizer,
        threshold=args.threshold,
    )
    print("Model loaded")

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    same, angle, cosine = app.predict_features(img1, img2, args.already_aligned)

    print("\n--- Result ---")
    print(f"Cosine similarity : {cosine:+.4f}  (range -1 to +1, higher = more similar)")
    print(f"Angle             : {angle:.2f}°  (lower = more similar)")
    print(f"Threshold         : {args.threshold:.1f}°")
    print(f"Verdict           : {'SAME PERSON' if same else 'DIFFERENT PERSON'}")


if __name__ == "__main__":
    main()
