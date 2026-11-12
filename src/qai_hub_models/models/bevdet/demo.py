# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import cast

import numpy as np

from qai_hub_models.models.bevdet.app import BEVDetApp
from qai_hub_models.models.bevdet.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    BEVDet,
    BEVDetDecoder,
    BEVDetEncoder,
    BEVDetPooler,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    transform_to_matrix,
)
from qai_hub_models.models.templates.corevision.display import display_or_save_image
from qai_hub_models.utils.args import (
    demo_model_components_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
)

# These assets are source from the nuscene dataset correspond to
# 'scene-0103' with sample_token '3e8750f331d7499e9b5123e9eb70f2e2'
CAM_FRONT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_FRONT.jpg"
)
CAM_FRONT_RIGHT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_FRONT_RIGHT.jpg"
)
CAM_FRONT_LEFT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_FRONT_LEFT.jpg"
)
CAM_BACK = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK.jpg"
)
CAM_BACK_LEFT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK_LEFT.jpg"
)
CAM_BACK_RIGHT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK_RIGHT.jpg"
)
CAM_BACK_RIGHT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "CAM_BACK_RIGHT.jpg"
)
INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "inputs.json"
)


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(BEVDet)
    parser = get_on_device_demo_parser(parser, add_output_dir=True)
    args = parser.parse_args([] if is_test else None)

    validate_on_device_demo_args(args, MODEL_ID)

    inference_model, (encoder, pooler, decoder) = demo_model_components_from_cli_args(
        BEVDet, MODEL_ID, args
    )

    shape = inference_model.encoder.get_input_spec()["image"][0]
    app = BEVDetApp(
        cast(BEVDetEncoder, encoder),
        cast(BEVDetPooler, pooler),
        cast(BEVDetDecoder, decoder),
        inference_model.bboxcoder,
        model_input_shape=(shape[-2], shape[-1]),
    )

    cam_paths = {
        "CAM_FRONT_LEFT": str(CAM_FRONT_LEFT.fetch()),
        "CAM_FRONT": str(CAM_FRONT.fetch()),
        "CAM_FRONT_RIGHT": str(CAM_FRONT_RIGHT.fetch()),
        "CAM_BACK_LEFT": str(CAM_BACK_LEFT.fetch()),
        "CAM_BACK": str(CAM_BACK.fetch()),
        "CAM_BACK_RIGHT": str(CAM_BACK_RIGHT.fetch()),
    }

    inputs = load_json(INPUTS.fetch())

    images_list = []
    intrins_list = []
    sensor2egos_list = []
    ego2globals_list = []

    for cam_name, cam_img_path in cam_paths.items():
        images_list.append(load_image(cam_img_path))
        intrin = np.array(inputs[cam_name]["intrins"], dtype=np.float32)
        intrins_list.append(intrin)
        sensor2ego = transform_to_matrix(
            inputs[cam_name]["sensor2ego_translation"],
            inputs[cam_name]["sensor2ego_rotation"],
        )
        sensor2egos_list.append(sensor2ego)
        ego2global = transform_to_matrix(
            inputs[cam_name]["ego2global_translation"],
            inputs[cam_name]["ego2global_rotation"],
        )
        ego2globals_list.append(ego2global)

    output_images = app.predict_3d_boxes_from_images(
        images_list, intrins_list, sensor2egos_list, ego2globals_list
    )

    if not is_test:
        # visualize images
        for i, cam_name in enumerate(cam_paths):
            display_or_save_image(
                output_images[i], args.output_dir, f"bevdet_{cam_name}.png"
            )


if __name__ == "__main__":
    main()
