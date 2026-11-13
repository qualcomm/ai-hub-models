# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json

import numpy as np

from qai_hub_models.models.foot_track_net.app import FootTrackNet_App
from qai_hub_models.models.foot_track_net.demo import main as demo_main
from qai_hub_models.models.foot_track_net.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    FootTrackNet,
)
from qai_hub_models.models.templates.corevision.image_processing import pil_resize_pad
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_path,
)
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test6.jpg"
)
OUTPUT_RST_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test6_rst.json"
)


# Verify that the output from Torch is as expected. bbox, landmark, visibility
def test_task() -> None:
    # error tolerance
    diff_tol = 0.1
    atol = 5
    rtol = 0.1

    model = FootTrackNet.from_pretrained()
    app = FootTrackNet_App(model)
    (_, _, height, width) = model.get_input_spec()["image"][0]
    image, *_ = pil_resize_pad(load_image(INPUT_IMAGE_ADDRESS), (height, width))
    objs_face, objs_person = app.predict_bbox_landmarks(image)

    pth = load_path(OUTPUT_RST_ADDRESS, "tmp")
    with open(pth) as f:
        rst = json.load(f)

    # extract the key detection result
    assert objs_person[0].landmark is not None
    persons_landmark = np.array(
        [
            [
                objs_person[0].landmark[0],
                objs_person[0].landmark[15],
                objs_person[0].landmark[16],
            ]
        ]
    )
    assert objs_person[0].vis is not None
    persons_visibility = np.array([[objs_person[0].vis[15], objs_person[0].vis[16]]])

    # assert, face_bbox, person_bbox, person_landmark and person_landmark_visibility
    assert_most_close(
        np.array(rst["faces_bbox"]),
        np.array([objs_face[0].box]),
        diff_tol=diff_tol,
        atol=atol,
        rtol=rtol,
    )
    assert_most_close(
        np.array(rst["persons_bbox"]),
        np.array([objs_person[0].box]),
        diff_tol=diff_tol,
        atol=atol,
        rtol=rtol,
    )
    assert_most_close(
        np.array(rst["persons_landmark"]),
        np.asarray(persons_landmark),
        diff_tol=diff_tol,
        atol=atol,
        rtol=rtol,
    )
    assert_most_close(
        np.array(rst["persons_visibility"]),
        np.asarray(persons_visibility),
        diff_tol=diff_tol,
        atol=0.2,
        rtol=rtol,
    )


def test_demo() -> None:
    demo_main(is_test=True)
