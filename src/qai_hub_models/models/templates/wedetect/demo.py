# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import numpy as np
from PIL import Image

from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.models.templates.wedetect import WeDetectApp
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebAsset, load_image
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

from .app import resolve_class_labels

COCO_LABELS = str(QAIHM_PACKAGE_ROOT / "labels" / "coco_labels.txt")


def wedetect_detection_demo(
    model_type: type[WorkbenchModelCollection],
    model_id: str,
    app_type: type[WeDetectApp],
    image: str | CachedWebAsset,
    class_labels: str | list[str] = COCO_LABELS,
    is_test: bool = False,
) -> None:
    """
    Shared demo function for WeDetect open-vocabulary object detection models.

    Parameters
    ----------
    model_type
        A ``WorkbenchModelCollection`` subclass (detector + text encoder).
    model_id
        Human-readable model identifier.
    app_type
        Application class — must be ``WeDetectApp`` or a subclass.
    image
        Default image path or URL.
    class_labels
        Class names to detect. Comma-separated string, path to ``.txt``, or list.
    is_test
        When ``True``, parses an empty argument list. Useful for CI testing.
    """
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=image,
        help="Input image file path or URL.",
    )
    parser.add_argument(
        "--class-labels",
        type=str,
        default=class_labels,
        help="Comma separated list of class names, or path to a .txt file.",
    )

    args = parser.parse_args([] if is_test else None)
    requested_classes = resolve_class_labels(args.class_labels)
    validate_on_device_demo_args(args, model_id)

    _, components = demo_model_components_from_cli_args(model_type, model_id, args)
    components_list = cast(list[ExecutableModelProtocol], list(components))
    app = app_type.from_components(components_list)

    print("Model Loaded")
    input_image = load_image(args.image)
    pred_images = app.predict_boxes_from_image(
        input_image, class_labels=requested_classes
    )
    out = Image.fromarray(np.array(pred_images[0]))
    if not is_test:
        display_or_save_image(out, args.output_dir, "wedetect_demo_output.png")
