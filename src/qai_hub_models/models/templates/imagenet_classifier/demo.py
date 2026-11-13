# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from torchvision import transforms

from qai_hub_models import TargetRuntime
from qai_hub_models.models.templates.corevision.image_processing import (
    IMAGENET_TRANSFORM,
)
from qai_hub_models.models.templates.imagenet_classifier.app import (
    ImagenetClassifierApp,
)
from qai_hub_models.models.templates.imagenet_classifier.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    ImagenetClassifier,
)
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)

IMAGENET_LABELS_ASSET = CachedWebModelAsset(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "imagenet_labels.json",
)
GROUP_NAME = "imagenet_classifier"
TEST_IMAGENET_IMAGE = CachedWebModelAsset.from_asset_store(
    GROUP_NAME, MODEL_ASSET_VERSION, "dog.jpg"
)


# Run Imagenet Classifier end-to-end on a sample image.
# The demo will print the predicted class to terminal.
def imagenet_demo(
    model_cls: type[ImagenetClassifier],
    model_id: str,
    is_test: bool = False,
    available_target_runtimes: list[TargetRuntime] | None = None,
    transform: transforms.Compose = IMAGENET_TRANSFORM,
) -> None:
    # Demo parameters
    if available_target_runtimes is None:
        available_target_runtimes = list(TargetRuntime.__members__.values())
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(
        parser, available_target_runtimes=available_target_runtimes
    )
    parser.add_argument(
        "--image",
        type=str,
        default=TEST_IMAGENET_IMAGE,
        help="test image file path or URL",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, model_id)

    model = demo_model_from_cli_args(model_cls, model_id, args)
    app = ImagenetClassifierApp(model, transform=transform)
    print("Model Loaded")

    image = load_image(args.image)
    # Run app
    probabilities = app.predict(image)
    top5 = torch.topk(probabilities, 5)
    if not is_test:
        labels = load_json(IMAGENET_LABELS_ASSET)
        print("Top 5 predictions for image:\n")
        for i in range(5):
            print(f"{labels[top5.indices[i]]}: {100 * top5.values[i]:.3g}%\n")
