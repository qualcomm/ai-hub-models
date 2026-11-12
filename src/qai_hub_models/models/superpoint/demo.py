# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from PIL import Image

from qai_hub_models.models.superpoint.app import DEFAULT_TOP_K, SuperPointApp
from qai_hub_models.models.superpoint.dataset import get_sample_image_pair_paths
from qai_hub_models.models.superpoint.model import MODEL_ID, SuperPoint
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image

DEFAULT_DETECTION_THRESHOLD = 0.001


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SuperPoint)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="First image file path or URL. Defaults to a real HPatches "
        "viewpoint-change pair.",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=None,
        help="Second image file path or URL. Defaults to a real HPatches "
        "viewpoint-change pair.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=DEFAULT_DETECTION_THRESHOLD,
        help="Minimum detection score for a pixel to be kept as a keypoint.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum number of keypoints to keep per image.",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(SuperPoint, MODEL_ID, args)
    input_spec = model.get_input_spec()

    image1_path, image2_path = args.image1, args.image2
    if image1_path is None or image2_path is None:
        default1, default2 = get_sample_image_pair_paths()
        image1_path = image1_path or default1
        image2_path = image2_path or default2

    image_a = load_image(image1_path)
    image_b = load_image(image2_path)
    print("Model loaded")

    app = SuperPointApp(
        model,  # type: ignore[arg-type]
        input_spec,
        args.detection_threshold,
        args.top_k,
    )
    canvas = app.predict_matches(image_a, image_b)
    assert isinstance(canvas, np.ndarray)

    if not is_test:
        print(
            "\nDetected keypoints and mutual descriptor matches between the two images."
        )
        display_or_save_image(
            Image.fromarray(canvas),
            args.output_dir,
            "superpoint_matches.png",
            "keypoint match visualization",
        )


if __name__ == "__main__":
    main()
