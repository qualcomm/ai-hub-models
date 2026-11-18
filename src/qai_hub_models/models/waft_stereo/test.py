# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    undo_resize_pad,
)
from qai_hub_models.models.waft_stereo.app import WaftStereoApp
from qai_hub_models.models.waft_stereo.dataset import resize_pad_rgb
from qai_hub_models.models.waft_stereo.demo import (
    DEFAULT_LEFT_IMAGE,
    DEFAULT_RIGHT_IMAGE,
)
from qai_hub_models.models.waft_stereo.demo import main as demo_main
from qai_hub_models.models.waft_stereo.evaluator import WaftStereoD1Evaluator
from qai_hub_models.models.waft_stereo.model import (
    WaftStereo,
    WaftStereoClassification,
    WaftStereoEncoder,
    WaftStereoRegression,
    load_source_model,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.input_spec import TensorSpec
from qai_hub_models.utils.test_helpers import assert_most_close


def run_source_model(
    left: Image.Image, right: Image.Image, height: int, width: int
) -> np.ndarray:
    source_model = load_source_model()
    source_model.eval()

    orig_img, left_t = app_to_net_image_inputs(left, image_layout="RGB")
    _, right_t = app_to_net_image_inputs(right, image_layout="RGB")
    # resize_pad_rgb (ImageNet-mean pad) rather than plain resize_pad (black
    # pad): the encoder normalizes post-pad, so mean-padding pre-normalize is
    # what actually reproduces upstream's pad-with-0-after-normalize Padder.
    left_t, scale, padding = resize_pad_rgb(left_t, (height, width))
    right_t, _, _ = resize_pad_rgb(right_t, (height, width))

    with torch.no_grad():
        output = source_model({"img1": left_t * 255.0, "img2": right_t * 255.0})

    disp_pred = output["disp_pred"].unsqueeze(1)
    h, w = orig_img[0].shape[:2]
    disp_pred = undo_resize_pad(disp_pred, (w, h), scale, padding)
    disp_pred = disp_pred / scale
    return disp_pred.squeeze().numpy().astype(np.float32)


def _run_test(
    encoder: WaftStereoEncoder,
    classification: WaftStereoClassification,
    regression: WaftStereoRegression,
    spec_model: WaftStereo,
    left: Image.Image | None = None,
    right: Image.Image | None = None,
) -> None:
    *_, height, width = spec_model.encoder.get_input_spec()["img1"][0]
    left = left or load_image(DEFAULT_LEFT_IMAGE)
    right = right or load_image(DEFAULT_RIGHT_IMAGE)
    exp_disp = run_source_model(left, right, height=height, width=width)
    input_spec = spec_model.encoder.get_input_spec()
    disp = WaftStereoApp(
        encoder, classification, regression, input_spec
    ).predict_disparity(left, right, raw_output=True)
    assert isinstance(disp, np.ndarray)
    assert_most_close(
        exp_disp,
        disp,
        diff_tol=1e-3,
        rtol=1e-3,
        atol=1e-3,
    )


def test_task() -> None:
    model = WaftStereo.from_pretrained()
    _run_test(model.encoder, model.classification, model.regression, model)


def test_demo() -> None:
    demo_main(is_test=True)


def test_predict_disparity_nondefault_aspect_ratio() -> None:
    """The default demo pair's aspect ratio is close enough to the network's
    that resize_pad's scale is ~1, which would let a broken `/ scale` rescale
    in predict_disparity pass unnoticed. A deliberately mismatched aspect
    ratio forces a non-trivial scale to actually exercise that rescale.
    """
    model = WaftStereo.from_pretrained()
    rng = np.random.default_rng(0)
    # 300x350 has aspect ratio 1.17 (network target is 320x240 -> 1.33), giving
    # scale=0.8 and only a 20px letterbox border out of 320 - enough to force a
    # real rescale without the border (padded differently by each pipeline)
    # dominating the frame and swamping the comparison.
    left = Image.fromarray(rng.integers(0, 256, (300, 350, 3), dtype=np.uint8))
    right = Image.fromarray(rng.integers(0, 256, (300, 350, 3), dtype=np.uint8))
    _run_test(model.encoder, model.classification, model.regression, model, left, right)


def test_run_model_for_eval() -> None:
    """Nothing else here calls run_model_for_eval; this exercises its
    sync-tensor path (plain callables, not AsyncOnDeviceResult) end-to-end.
    """
    fmap1, fmap2, net_state = (
        torch.ones(1, 2),
        torch.full((1, 2), 2.0),
        torch.full((1, 2), 3.0),
    )
    disp = torch.full((1, 2), 4.0)
    reg_output = torch.full((1, 2), 5.0)

    def encoder(
        img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fmap1, fmap2, net_state

    def classification(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        assert torch.equal(f1, fmap1) and torch.equal(f2, fmap2)
        return disp

    def regression(
        f1: torch.Tensor, f2: torch.Tensor, state: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        assert torch.equal(f1, fmap1) and torch.equal(f2, fmap2)
        assert torch.equal(state, net_state) and torch.equal(d, disp)
        return reg_output

    app = WaftStereoApp(
        encoder,
        classification,
        regression,
        input_spec={"img1": TensorSpec(shape=(1, 3, 240, 320))},
    )
    img1 = img2 = torch.zeros(1, 3, 240, 320)
    gen = app.run_model_for_eval((img1, img2), model_batch_size=1)
    yielded = []
    try:
        while True:
            yielded.append(next(gen))
    except StopIteration as stop:
        final_output = stop.value

    enc_output = cast(tuple[torch.Tensor, ...], yielded[0])
    cls_output = cast(tuple[torch.Tensor, ...], yielded[1])
    assert torch.equal(enc_output[0], fmap1)
    assert torch.equal(cls_output[0], disp)
    assert torch.equal(cast(torch.Tensor, yielded[2]), reg_output)
    assert torch.equal(cast(torch.Tensor, final_output), reg_output)


def test_evaluator_add_batch_hand_computed() -> None:
    """Hand-computed native-resolution D1 counts, with one invalid (gt<=0)
    pixel and both a within-threshold and an over-threshold error, to pin
    down WaftStereoD1Evaluator's bad/valid pixel accounting.
    """
    # gt=0.0 is invalid and excluded regardless of its paired prediction.
    disp_native = torch.tensor([[[5.0, 0.0], [10.0, 100.0]]])
    scale = torch.tensor([1.0])
    pad_left = torch.tensor([0])
    pad_top = torch.tensor([0])

    # (0,0): error 2.0, below the 3.0 abs threshold -> not bad.
    # (1,0): error 4.0 > 3.0 abs and > 0.05*10=0.5 rel -> bad.
    # (1,1): error 10.0 > 3.0 abs and > 0.05*100=5.0 rel -> bad.
    output = torch.tensor([[[[7.0, 999.0], [14.0, 90.0]]]])

    evaluator = WaftStereoD1Evaluator()
    evaluator.add_batch(output, (disp_native, scale, pad_left, pad_top))

    assert evaluator._valid_count == 3.0
    assert evaluator._bad_count == 2.0
    assert evaluator.get_accuracy_score() == pytest.approx(100 * 2 / 3)


def test_input_spec_consistency_at_nondefault_size() -> None:
    """classification/regression's fmap spec must track height/width so a CLI
    --height/--width override (forwarded to every component by param name)
    reaches them the same way it reaches the encoder.
    """
    height, width = 480, 640
    encoder = WaftStereoEncoder(net=cast(Any, None))
    classification = WaftStereoClassification(net=cast(Any, None))
    regression = WaftStereoRegression(net=cast(Any, None))

    enc_shape = encoder.get_input_spec(height=height, width=width)["img1"][0]
    cls_fmap_shape = classification.get_input_spec(height=height, width=width)["fmap1"][
        0
    ]
    reg_fmap_shape = regression.get_input_spec(height=height, width=width)["fmap1"][0]

    expected_fmap_shape = (1, 48, height // 2, width // 2)
    assert enc_shape == (1, 3, height, width)
    assert cls_fmap_shape == expected_fmap_shape
    assert reg_fmap_shape == expected_fmap_shape
