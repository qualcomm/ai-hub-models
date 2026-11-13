# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.superpoint.dataset import (
    HPATCHES_ASSET,
    _find_sequences_root,
)
from qai_hub_models.models.superpoint.demo import main as demo_main
from qai_hub_models.models.superpoint.model import SuperPoint
from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.utils.asset_loaders import load_image


def test_task() -> None:
    """
    End-to-end sanity check of the torch model + evaluator pipeline: matching
    an image against itself under the identity homography should yield
    near-perfect Repeatability and Homography Correctness, since every
    keypoint is trivially its own correspondence. This exercises
    the same forward -> extract_keypoints -> match/warp path used for real
    HPatches pairs, without depending on a separately-hosted reference
    fixture.
    """
    model = SuperPoint.from_pretrained()
    height, width = model.get_input_spec()["image_a"][0][2:]

    HPATCHES_ASSET.fetch(extract=True)
    seq_root = _find_sequences_root(HPATCHES_ASSET.extracted_path)
    seq_dir = sorted(d for d in seq_root.iterdir() if d.is_dir())[0]
    image = load_image(str(seq_dir / "1.ppm")).convert("RGB").resize((width, height))
    image_t = numpy_image_to_torch(np.array(image))

    with torch.no_grad():
        output = model(image_t, image_t)

    evaluator = model.get_evaluator()
    identity_homography = torch.eye(3).unsqueeze(0)
    evaluator.add_batch(output, identity_homography)

    assert evaluator.get_accuracy_score() > 90.0
    assert evaluator.get_repeatability_score() > 90.0
    assert evaluator.get_homography_correctness_score() > 90.0


def test_demo() -> None:
    demo_main(is_test=True)
