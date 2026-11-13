# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from PIL import Image
from skimage.data import astronaut

from qai_hub_models.models.sixd_repnet.app import SixDRepNetApp
from qai_hub_models.models.sixd_repnet.model import MODEL_ID, SixDRepNet
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import load_image


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(SixDRepNet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="file path or URL of a face image",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    _, (face_detector, pose_estimator) = demo_model_components_from_cli_args(
        SixDRepNet, MODEL_ID, args
    )

    if args.image:
        image = load_image(args.image).convert("RGB")
    else:
        image = Image.fromarray(astronaut())  # 512x512 RGB face photo

    app = SixDRepNetApp(
        face_detector=face_detector,  # type: ignore[arg-type]
        pose_estimator=pose_estimator,  # type: ignore[arg-type]
    )
    output = app.predict_pose(image)

    if not is_test:
        display_or_save_image(output, args.output_dir, "sixdrepnet_output.png")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
