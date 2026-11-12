# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from PIL import Image

from qai_hub_models.models.cvt.app import CVTApp
from qai_hub_models.models.cvt.model import CVT, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    model_from_cli_args,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)

# Sample 6-camera images for demo
CAMERAS = {
    "CAM_FRONT_LEFT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "CAM_FRONT_LEFT.jpg"
    ),
    "CAM_FRONT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "CAM_FRONT.jpg"
    ),
    "CAM_FRONT_RIGHT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "CAM_FRONT_RIGHT.jpg"
    ),
    "CAM_BACK_LEFT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK_LEFT.jpg"
    ),
    "CAM_BACK": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK.jpg"
    ),
    "CAM_BACK_RIGHT": CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK_RIGHT.jpg"
    ),
}
CAM_METADATA = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inputs.json"
)


def cvt_demo(
    model_type: type[CVT],
    model_id: str,
    cameras: dict[str, CachedWebModelAsset],
    cam_metadata: CachedWebModelAsset,
    is_test: bool = False,
) -> list[Image.Image] | torch.Tensor:
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser)
    add_output_dir_arg(parser)
    parser.add_argument(
        "--ckpt", type=str, default="both", choices=["both", "vehicles", "road"]
    )
    args = parser.parse_args([] if is_test else None)

    # Load images and inputs
    images = [load_image(str(img.fetch())) for img in cameras.values()]
    camera_metadata = load_json(cam_metadata.fetch())

    # Process model inference
    h, w = model_from_cli_args(model_type, args).get_input_spec()["image"][0][3:]

    def process_model(ckpt_name: str) -> list[Image.Image] | torch.Tensor:
        args.ckpt_name = ckpt_name
        model = demo_model_from_cli_args(model_type, model_id, args)
        validate_on_device_demo_args(args, model_id)
        app = CVTApp(
            model,  # type: ignore[arg-type]
            ckpt_name=ckpt_name,
            target_height=h,
            target_width=w,
        )
        return app.predict_from_images(images, camera_metadata, raw_output=is_test)

    if args.ckpt == "both":
        vehicle_maps = process_model("vehicles_50k")
        road_maps = process_model("road_75k")
        if not is_test:
            for v_img, r_img in zip(vehicle_maps, road_maps, strict=False):
                combined = Image.blend(v_img, r_img, alpha=0.5)
                display_or_save_image(combined, args.output_dir, "cvt_bev_combined.png")
                display_or_save_image(v_img, args.output_dir, "cvt_bev_vehicle.png")
                display_or_save_image(r_img, args.output_dir, "cvt_bev_road.png")
        return vehicle_maps
    ckpt_name = "vehicles_50k" if args.ckpt == "vehicles" else "road_75k"
    maps = process_model(ckpt_name)
    if not is_test:
        for i, img in enumerate(iterable=maps):
            display_or_save_image(img, args.output_dir, f"cvt_bev_{args.ckpt}_{i}.png")
    return maps


def main(is_test: bool = False) -> None:
    cvt_demo(CVT, MODEL_ID, CAMERAS, CAM_METADATA, is_test)


if __name__ == "__main__":
    main()
