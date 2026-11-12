# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.siglip2.app import SigLIP2App
from qai_hub_models.models.siglip2.model import MODEL_ASSET_VERSION, MODEL_ID, SigLIP2
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

DEFAULT_DEMO_IMAGES = ["image1.jpg", "image2.jpg", "image3.jpg"]
DEFAULT_TEXT_PROMPT = "camping under the stars"


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SigLIP2)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image-paths",
        type=str,
        default="",
        help="Comma-separated image file paths or URLs.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help="Text prompt for zero-shot image search.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    wrapper, (image_encoder, text_encoder) = demo_model_components_from_cli_args(
        SigLIP2, MODEL_ID, args
    )

    app = SigLIP2App(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        logit_scale=wrapper.logit_scale,
        logit_bias=wrapper.logit_bias,
    )

    image_paths: list[str] = []
    if args.image_paths:
        image_paths = [p.strip() for p in args.image_paths.split(",")]
    else:
        for fname in DEFAULT_DEMO_IMAGES:
            asset = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, fname
            )
            asset.fetch()
            image_paths.append(str(asset.path))

    predictions = app.predict_similarity(image_paths, [args.text]).flatten()

    print(f"Searching images by prompt: {args.text}")
    for i, path in enumerate(image_paths):
        print(f"  {path}: similarity score = {predictions[i]:.4f}")

    print("Displaying the most relevant image")
    best_idx = int(np.argmax(predictions.detach().numpy()))
    most_relevant = load_image(image_paths[best_idx])
    if not is_test:
        display_or_save_image(most_relevant, args.output_dir)


if __name__ == "__main__":
    main()
