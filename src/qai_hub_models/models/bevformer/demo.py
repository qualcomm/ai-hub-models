# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models.models.bevformer.app import (
    CLASS_NAMES,
    SCORE_THRESHOLD,
    BEVFormerApp,
    visualize_bev,
    visualize_on_cameras,
)
from qai_hub_models.models.bevformer.model import (
    MODEL_ID,
    BEVFormer,
    load_frame_inputs,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(BEVFormer)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    args = parser.parse_args([] if is_test else None)
    model = demo_model_from_cli_args(BEVFormer, MODEL_ID, args)
    validate_on_device_demo_args(args, MODEL_ID)

    app = BEVFormerApp(model, score_threshold=SCORE_THRESHOLD)  # type: ignore[arg-type]

    spec = BEVFormer.get_input_spec()
    image, can_bus, lidar2img = load_frame_inputs(spec)

    result = app.predict_3d_boxes(image, can_bus, lidar2img)

    if not isinstance(result, dict) or len(result["scores"]) == 0:
        print("No detections above threshold.")
        return

    detections: dict = result
    n = len(detections["scores"])
    print(f"{n} detections found")

    if is_test:
        return

    topk = min(10, n)
    idx = detections["scores"].topk(topk).indices
    print(f"\nTop-{topk} detections:")
    for rank, ii in enumerate(idx):
        ii = ii.item()
        cls_idx = detections["labels"][ii].item()
        print(
            f"  [{rank}] {CLASS_NAMES[cls_idx]:20s} "
            f"score={detections['scores'][ii]:.3f}  "
            f"xy=({detections['x'][ii]:.1f},{detections['y'][ii]:.1f})m"
        )

    display_or_save_image(
        visualize_bev(detections), args.output_dir, "bev_detections.png"
    )
    display_or_save_image(
        visualize_on_cameras(image, lidar2img, detections),
        args.output_dir,
        "camera_detections.png",
    )


if __name__ == "__main__":
    main()
