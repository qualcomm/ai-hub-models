# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.ops import deform_conv2d

from qai_hub_models.models.basicvsr_pp.app import BasicVSRPlusPlusApp
from qai_hub_models.models.basicvsr_pp.demo import main as demo_main
from qai_hub_models.models.basicvsr_pp.mmcv_shim import (
    grouped_modulated_deform_conv2d,
)
from qai_hub_models.models.basicvsr_pp.model import (
    DEFAULT_HEIGHT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_WIDTH,
    MID_CHANNELS,
    MODEL_ASSET_VERSION,
    MODEL_ID,
    BasicVSRPlusPlus,
    _patch_deform_align_conv_offset,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def _make_random_clip(
    num_frames: int = DEFAULT_NUM_FRAMES,
    h: int = DEFAULT_HEIGHT,
    w: int = DEFAULT_WIDTH,
) -> torch.Tensor:
    return torch.rand(num_frames, 3, h, w)


def test_conv_offset_weight_patch() -> None:
    """Weight insertion at channel 192 must not disturb the flow weight columns."""
    torch.manual_seed(0)
    out_ch, kH, kW = 32, 3, 3
    conv = nn.Conv2d(196, out_ch, (kH, kW), padding=1)

    # Save a reference copy before patching.
    conv_ref = nn.Conv2d(196, out_ch, (kH, kW), padding=1)
    conv_ref.weight = nn.Parameter(conv.weight.clone())
    assert conv.bias is not None
    conv_ref.bias = nn.Parameter(conv.bias.clone())

    _patch_deform_align_conv_offset(conv)
    assert conv.in_channels == 200

    # Original input: [cond(192), flows(4)]; patched adds zeros at position 192.
    x_cond = torch.randn(1, 192, 8, 8)
    x_flow = torch.randn(1, 4, 8, 8)
    x_orig = torch.cat([x_cond, x_flow], dim=1)
    x_patch = torch.cat([x_cond, torch.zeros(1, 4, 8, 8), x_flow], dim=1)

    with torch.no_grad():
        out_ref = conv_ref(x_orig)
        out_patched = conv(x_patch)

    torch.testing.assert_close(out_ref, out_patched, rtol=1e-5, atol=1e-5)


def test_grouped_modulated_deform_conv2d() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 4, 5, 6)
    weight = torch.randn(3, 4, 3, 3)
    bias = torch.randn(3)
    offset = torch.randn(1, 36, 5, 6) * 0.1
    mask = torch.sigmoid(torch.randn(1, 18, 5, 6))

    expected = deform_conv2d(
        x,
        offset,
        weight,
        bias,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        mask=mask,
    )
    actual = grouped_modulated_deform_conv2d(
        x,
        weight,
        bias,
        offset,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        deform_groups=2,
        mask=mask,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)


def test_flownet() -> None:
    model = BasicVSRPlusPlus.from_pretrained()
    flownet = model.flownet
    ref = torch.rand(1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH)
    supp = torch.rand(1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH)
    with torch.no_grad():
        flow = flownet(ref, supp)
    assert flow.shape == (1, 2, DEFAULT_HEIGHT, DEFAULT_WIDTH), (
        f"Unexpected flow shape: {flow.shape}"
    )
    assert torch.isfinite(flow).all(), "Flow contains non-finite values"


def test_feat_extract() -> None:
    model = BasicVSRPlusPlus.from_pretrained()
    feat_extract = model.feat_extract
    frame = torch.rand(1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH)
    with torch.no_grad():
        feat = feat_extract(frame)
    assert feat.shape == (1, MID_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH), (
        f"Unexpected feat shape: {feat.shape}"
    )
    assert torch.isfinite(feat).all()


def test_prop_step_shapes() -> None:
    model = BasicVSRPlusPlus.from_pretrained()
    h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
    feat = torch.rand(1, MID_CHANNELS, h, w)
    zeros_feat = torch.zeros(1, MID_CHANNELS, h, w)
    zeros_flow = torch.zeros(1, 2, h, w)

    # backward_1: no priors
    with torch.no_grad():
        out = model.prop_backward_1(feat, zeros_feat, feat, zeros_flow, zeros_flow)
    assert out.shape == (1, MID_CHANNELS, h, w)
    assert torch.isfinite(out).all()

    # forward_1: 1 prior
    with torch.no_grad():
        out = model.prop_forward_1(feat, zeros_feat, feat, zeros_flow, zeros_flow, out)
    assert out.shape == (1, MID_CHANNELS, h, w)
    assert torch.isfinite(out).all()


def test_reconstruction_shape() -> None:
    model = BasicVSRPlusPlus.from_pretrained()
    h, w = DEFAULT_HEIGHT, DEFAULT_WIDTH
    feats = {
        k: torch.rand(1, MID_CHANNELS, h, w) for k in ["s", "b1", "f1", "b2", "f2"]
    }
    lq = torch.rand(1, 3, h, w)
    with torch.no_grad():
        hr = model.reconstruction(
            feats["s"], feats["b1"], feats["f1"], feats["b2"], feats["f2"], lq
        )
    assert hr.shape == (1, 3, h * 4, w * 4), f"Unexpected HR shape: {hr.shape}"


def test_task() -> None:
    """End-to-end pipeline: fixed 2-frame clip; frame 1 compared against saved golden."""
    torch.manual_seed(42)
    lq = _make_random_clip(num_frames=2)
    model = BasicVSRPlusPlus.from_pretrained()
    app = BasicVSRPlusPlusApp.from_model(model)
    with torch.no_grad():
        hr_frames = app._run_inference(lq)

    assert len(hr_frames) == 2
    out = hr_frames[1]
    assert out.shape == (1, 3, DEFAULT_HEIGHT * 4, DEFAULT_WIDTH * 4), (
        f"Unexpected shape {out.shape}"
    )
    assert torch.isfinite(out).all(), "Non-finite values in output"

    try:
        golden_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, "test_files/basicvsrpp_golden_output.npy"
        ).fetch()
    except (OSError, ValueError) as exc:
        pytest.skip(f"Golden output unavailable: {exc}")

    expected = torch.from_numpy(np.load(str(golden_path)))
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-4)


def test_demo() -> None:
    demo_main(is_test=True)
