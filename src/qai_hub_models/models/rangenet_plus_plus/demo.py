# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np

from qai_hub_models.models.rangenet_plus_plus.app import (
    RangeNetApp,
    project_points_to_range_image,
)
from qai_hub_models.models.rangenet_plus_plus.model import (
    MODEL_ID,
    SAMPLE_POINT_CLOUD_ADDRESS,
    RangeNetPlusPlus,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(RangeNetPlusPlus)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    parser.add_argument(
        "--point-cloud",
        type=str,
        default=None,
        help=(
            "Path to a KITTI binary .bin point cloud file to segment "
            "(raw float32 array with columns [x, y, z, intensity]). "
            "Example: --point-cloud /path/to/velodyne/000000.bin  "
            "If omitted, the bundled sample scan is downloaded automatically."
        ),
    )
    parser.add_argument(
        "--range-image",
        type=str,
        default=None,
        help=(
            "Path to a pre-projected range image saved as a .npy file "
            "with shape (1, 5, H, W) float32. "
            "When provided, skips point cloud projection and runs inference directly. "
            "Example: --range-image /path/to/range_image.npy"
        ),
    )
    args = parser.parse_args([] if is_test else None)
    if args.model_dir:
        args.model_dir = os.path.abspath(args.model_dir)
    validate_on_device_demo_args(args, MODEL_ID)

    # Load model
    model = RangeNetPlusPlus.from_pretrained(model_dir=args.model_dir)
    app = RangeNetApp(model)

    if args.range_image:
        # User supplied a pre-projected range image — run inference directly.
        range_image_arr = np.load(args.range_image)
        seg_image = app.segment_range_image(range_image_arr)
        bev_image = None
    else:
        # Load point cloud — KITTI binary format: raw float32 [x, y, z, intensity]
        point_cloud_path = args.point_cloud or str(SAMPLE_POINT_CLOUD_ADDRESS.fetch())
        points = np.fromfile(point_cloud_path, dtype=np.float32).reshape(-1, 4)
        # Run inference
        range_image_arr, _, _ = project_points_to_range_image(points)
        seg_image, bev_image = app.segment_and_bev(points, range_image=range_image_arr)

    if not is_test:
        display_or_save_image(
            seg_image, args.output_dir, "segmentation.png", "Segmentation"
        )
        if bev_image is not None:
            display_or_save_image(
                bev_image, args.output_dir, "bird_eye_view.png", "Bird's-Eye View"
            )


if __name__ == "__main__":
    main()
