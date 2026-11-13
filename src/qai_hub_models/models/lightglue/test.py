# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from PIL import Image

from qai_hub_models.models.lightglue.demo import main as demo_main
from qai_hub_models.models.lightglue.model import (
    IMAGE0_ADDRESS,
    LightGlue,
)
from qai_hub_models.models.lightglue.postprocess import (
    MAX_KEYPOINTS,
    extract_padded_features,
)
from qai_hub_models.models.superpoint.model import SuperPoint
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import load_image


def test_task() -> None:
    """
    Matching an image against itself is a golden output: every keypoint must
    match the keypoint at its own index, since both inputs are identical.
    """
    extractor = SuperPoint.from_pretrained()
    model = LightGlue.from_pretrained()

    image = (
        load_image(IMAGE0_ADDRESS)
        .convert("RGB")
        .resize((model.width, model.height), Image.BILINEAR)
    )
    _, image_t = app_to_net_image_inputs(image)

    kp0, desc0, n0, kp1, desc1, n1 = extract_padded_features(
        extractor, image_t, image_t
    )
    with torch.no_grad():
        matches0, matches1, _mscores0, _mscores1 = model(kp0, desc0, n0, kp1, desc1, n1)

    n = int(n0[0].item())
    assert n > 0
    expected = torch.arange(n, dtype=matches0.dtype)
    torch.testing.assert_close(matches0[0, :n], expected)
    torch.testing.assert_close(matches1[0, :n], expected)


def test_padding_is_masked_out() -> None:
    """
    Padded keypoint slots must never affect real matches, and must always
    come back as matches=-1/scores=0 themselves. Pads the same synthetic
    keypoints/descriptors to two different lengths and checks both runs
    agree on the real (unpadded) prefix -- this is the case test_task's
    self-match golden output can't cover, since there both images share
    identical padding. Uses synthetic (not SuperPoint-extracted) features
    so the real keypoint count is fixed, rather than depending on how many
    keypoints a real image happens to produce on a given platform (which
    can saturate MAX_KEYPOINTS regardless of detection_threshold).
    """
    model = LightGlue.from_pretrained()
    d = model.conf.descriptor_dim
    generator = torch.Generator().manual_seed(0)
    size = torch.tensor([model.width, model.height], dtype=torch.float32)

    def _padded(
        n: int, max_keypoints: int, kp_real: torch.Tensor, desc_real: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kp = kp_real.new_zeros(1, max_keypoints, 2)
        desc = desc_real.new_zeros(1, max_keypoints, d)
        kp[0, :n] = kp_real
        desc[0, :n] = desc_real
        return kp, desc, torch.tensor([n], dtype=torch.int32)

    n0, n1 = 30, 40
    kp0_real = torch.rand(n0, 2, generator=generator) * size
    desc0_real = torch.nn.functional.normalize(
        torch.randn(n0, d, generator=generator), dim=-1
    )
    kp1_real = torch.rand(n1, 2, generator=generator) * size
    desc1_real = torch.nn.functional.normalize(
        torch.randn(n1, d, generator=generator), dim=-1
    )

    kp0, desc0, num_kp0 = _padded(n0, MAX_KEYPOINTS, kp0_real, desc0_real)
    kp1, desc1, num_kp1 = _padded(n1, MAX_KEYPOINTS, kp1_real, desc1_real)

    tight_k = max(n0, n1) + 5
    kp0_t, desc0_t, num_kp0_t = _padded(n0, tight_k, kp0_real, desc0_real)
    kp1_t, desc1_t, num_kp1_t = _padded(n1, tight_k, kp1_real, desc1_real)

    with torch.no_grad():
        matches0, matches1, mscores0, mscores1 = model(
            kp0, desc0, num_kp0, kp1, desc1, num_kp1
        )
        matches0_t, matches1_t, mscores0_t, mscores1_t = model(
            kp0_t, desc0_t, num_kp0_t, kp1_t, desc1_t, num_kp1_t
        )

    # The padded tail is always -1 / 0, regardless of padding length.
    assert torch.all(matches0[0, n0:] == -1)
    assert torch.all(mscores0[0, n0:] == 0)
    assert torch.all(matches0_t[0, n0:] == -1)
    assert torch.all(mscores0_t[0, n0:] == 0)

    # The real (unpadded) prefix doesn't depend on how much padding follows it.
    torch.testing.assert_close(matches0[0, :n0], matches0_t[0, :n0])
    torch.testing.assert_close(mscores0[0, :n0], mscores0_t[0, :n0])
    torch.testing.assert_close(matches1[0, :n1], matches1_t[0, :n1])
    torch.testing.assert_close(mscores1[0, :n1], mscores1_t[0, :n1])


def test_demo() -> None:
    """Run demo and verify it does not crash."""
    demo_main(is_test=True)
