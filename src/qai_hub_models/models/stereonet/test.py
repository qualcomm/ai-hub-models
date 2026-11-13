# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from PIL import Image
from stereonet.model import StereoNet as StereoNetModel

from qai_hub_models.models.stereonet.app import StereoNetApp
from qai_hub_models.models.stereonet.demo import DEFAULT_LEFT_IMAGE, DEFAULT_RIGHT_IMAGE
from qai_hub_models.models.stereonet.demo import main as demo_main
from qai_hub_models.models.stereonet.model import DEFAULT_CKPT, StereoNet
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.test_helpers import (
    assert_most_close,
)


def run_source_model(
    left: Image.Image, right: Image.Image, height: int, width: int
) -> np.ndarray:
    ckpt_path = str(DEFAULT_CKPT.fetch())
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

    source_model = StereoNetModel(in_channels=1)
    source_model.load_state_dict(state_dict, strict=False)
    source_model.eval()

    _, left_t = app_to_net_image_inputs(left, image_layout="L")
    _, right_t = app_to_net_image_inputs(right, image_layout="L")
    left_t, _, _ = resize_pad(left_t, (height, width))
    right_t, _, _ = resize_pad(right_t, (height, width))
    stack = torch.cat([left_t, right_t], dim=1)
    mean = (torch.tensor([111.5684, 113.6528]) / 255).view(1, 2, 1, 1)
    std = (torch.tensor([61.9625, 62.0313]) / 255).view(1, 2, 1, 1)

    with torch.no_grad():
        pred = source_model((stack - mean) / std)

    return pred.squeeze().numpy().astype(np.float32)


def _run_test(model: StereoNet, spec_model: StereoNet) -> None:
    *_, height, width = spec_model.get_input_spec()["image"][0]
    left = load_image(DEFAULT_LEFT_IMAGE)
    right = load_image(DEFAULT_RIGHT_IMAGE)
    exp_disp = run_source_model(left, right, height=height, width=width)
    disp = StereoNetApp(model, height=height, width=width).predict_disparity(
        left, right, raw_output=True
    )
    assert isinstance(disp, np.ndarray)
    assert_most_close(
        exp_disp,
        disp,
        diff_tol=1e-3,
        rtol=1e-3,
        atol=1e-3,
    )


def test_task() -> None:
    model = StereoNet.from_pretrained()
    _run_test(model, model)


def test_demo() -> None:
    demo_main(is_test=True)
