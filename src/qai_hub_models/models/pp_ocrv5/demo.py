# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.pp_ocrv5.app import PPOCRv5App
from qai_hub_models.models.pp_ocrv5.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    PPOCRv5,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "signage.jpg"
)


def main(is_test: bool = False) -> None:
    # Demo parameters
    parser = get_model_cli_parser(PPOCRv5)
    parser.add_argument(
        "--image",
        type=str,
        default=INPUT_IMAGE_ADDRESS,
        help="image file path or URL",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "ko", "en", "latin", "eslav", "thai", "greek"],
        help="recognition language component to use",
    )
    args = parser.parse_args([] if is_test else None)

    # Load model and image
    model = model_from_cli_args(PPOCRv5, args)
    image = load_image(args.image)

    recognizer = getattr(model, f"{args.lang}_rec")
    app = PPOCRv5App(
        model.det,
        recognizer,
        tuple(model.det.get_input_spec()["image"][0][2:4]),
        tuple(recognizer.get_input_spec()["image"][0][2:4]),
        charlist=recognizer.get_charlist(),
    )
    print("Model Loaded")

    # `results` holds the recognized text (one string per detected region),
    # produced by greedy CTC decoding the recognizer logits.
    results = app.predict_text_from_image(image)

    if not is_test:
        print(f"Detected {len(results)} text region(s).")
        for i, text in enumerate(results):
            print(f"region {i}: {text}")
        print("\nRecognized text:")
        print("\n".join(text for text in results if text))


if __name__ == "__main__":
    main()
