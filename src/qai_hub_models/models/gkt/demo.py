# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

from PIL.Image import Image

from qai_hub_models.models.gkt.app import GKTApp
from qai_hub_models.models.gkt.model import GKT, MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    add_output_dir_arg,
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


def gkt_demo(
    model_type: type[GKT],
    model_id: str,
    cameras: dict[str, CachedWebModelAsset],
    cam_metadata: CachedWebModelAsset,
    is_test: bool = False,
) -> list[Image]:
    parser = get_model_cli_parser(model_type)
    parser = get_on_device_demo_parser(parser)
    add_output_dir_arg(parser)
    args = parser.parse_args([] if is_test else None)

    # Load images and inputs
    cam_paths = dict(cameras.items())
    images = [load_image(str(img.fetch())) for img in cam_paths.values()]
    camera_metadata = load_json(cam_metadata.fetch())

    model = demo_model_from_cli_args(model_type, model_id, args)
    validate_on_device_demo_args(args, model_id)
    h, w = GKT.from_pretrained().get_input_spec()["image"][0][3:]
    app = GKTApp(
        model,  # type: ignore[arg-type]
        ckpt_name="vehicle",
        target_height=h,
        target_width=w,
    )
    maps = cast(
        list[Image],
        app.predict_from_images(images, camera_metadata, raw_output=is_test),
    )

    if not is_test:
        for i, img in enumerate(maps):
            display_or_save_image(img, args.output_dir, f"gkt_bev_{i}.png")
    return maps


def main(is_test: bool = False) -> None:
    gkt_demo(GKT, MODEL_ID, CAMERAS, CAM_METADATA, is_test)


if __name__ == "__main__":
    main()
