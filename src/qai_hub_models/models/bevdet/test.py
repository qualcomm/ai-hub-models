# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import importlib
import importlib.metadata

import numpy as np
from packaging.version import parse as parse_version

from qai_hub_models.models.bevdet.app import BEVDetApp
from qai_hub_models.models.bevdet.demo import (
    CAM_BACK,
    CAM_BACK_LEFT,
    CAM_BACK_RIGHT,
    CAM_FRONT,
    CAM_FRONT_LEFT,
    CAM_FRONT_RIGHT,
    INPUTS,
)
from qai_hub_models.models.bevdet.demo import main as demo_main
from qai_hub_models.models.bevdet.model import MODEL_ASSET_VERSION, MODEL_ID, BEVDet
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    transform_to_matrix,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_json,
    load_numpy,
)

OUTPUT_PYTORCH_34MINUS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "corners.npy"
)
OUTPUT_PYTORCH_35PLUS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "corners_pt35_plus.npy"
)
OUTPUT_PYTORCH_211PLUS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "corners_pt211_plus.npy"
)


def _get_expected_output() -> np.ndarray:
    torch_version = parse_version(importlib.metadata.distribution("torch").version)
    if torch_version >= parse_version("2.11.0"):
        # pytorch/pytorch#154983 changed LU factorization on CPU to avoid using
        # transposed matrices, which subtly changes torch.inverse numerics for
        # the 4x4 ego2global matrices (~3e-5 diff). This propagates through
        # the network and drops 3 borderline detections (67 -> 64 boxes).
        return load_numpy(OUTPUT_PYTORCH_211PLUS.fetch())
    if torch_version >= parse_version("2.5.0"):
        # The numerics of torch.inverse subtly changed in pytorch 2.5.
        # This changes the network input slightly, which results in 10% different output.
        return load_numpy(OUTPUT_PYTORCH_35PLUS.fetch())
    return load_numpy(OUTPUT_PYTORCH_34MINUS.fetch())


def test_task() -> None:
    model = BEVDet.from_pretrained()
    app = BEVDetApp(
        model.encoder,
        model.pooler,
        model.decoder,
        model.bboxcoder,
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

    corners = app.predict_3d_boxes_from_images(
        images_list,
        intrins_list,
        sensor2egos_list,
        ego2globals_list,
        raw_output=True,
    )

    np.testing.assert_allclose(
        np.array(corners), _get_expected_output(), rtol=1e-03, atol=5e-04
    )


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
