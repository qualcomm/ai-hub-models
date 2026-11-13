# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.easyocr.app import EasyOCRApp
from qai_hub_models.models.easyocr.model import MODEL_ASSET_VERSION, MODEL_ID, EasyOCR
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "gracewood.png"
)


def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(EasyOCR)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    if not isinstance(args.lang_list, list):
        raise TypeError("The 'lang_list' parameter must be a list.")
    validate_on_device_demo_args(args, MODEL_ID)

    # Load app and image
    image = load_image(args.image)
    python_model, (detector, recognizer) = demo_model_components_from_cli_args(
        EasyOCR, MODEL_ID, args
    )

    app = EasyOCRApp(
        detector,  # type: ignore[arg-type]
        recognizer,  # type: ignore[arg-type]
        tuple(python_model.detector.get_input_spec()["image"][0][2:4]),
        tuple(python_model.recognizer.get_input_spec()["image"][0][2:4]),
        python_model.lang_list,
    )
    print("Model Loaded")

    results = app.predict_text_from_image(image)[0]

    if not is_test:
        display_or_save_image(results[0], args.output_dir)
        print("Predicted texts & confidence:")
        for i in range(len(results[1])):
            print(f"{results[1][i]}   {results[2][i]}")


if __name__ == "__main__":
    main()
