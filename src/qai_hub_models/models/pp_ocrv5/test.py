# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch

from qai_hub_models.models.pp_ocrv5.demo import main as demo_main
from qai_hub_models.models.pp_ocrv5.model import (
    DETECTION_RESOLUTION,
    RECOGNITION_HEIGHT,
    RECOGNITION_WIDTH,
    PPOCRv5,
    PPOCRv5Detector,
    PPOCRv5Recognizer,
)
from qai_hub_models.scorecard.utils.testing import skip_clone_repo_check


@skip_clone_repo_check
def test_task() -> None:
    model = PPOCRv5.from_pretrained()

    # Detection produces a per-pixel probability map.
    height, width = DETECTION_RESOLUTION
    det_input = torch.rand(1, 3, height, width)
    prob_map = np.asarray(model.det(det_input))
    assert prob_map.shape[0] == 1
    assert prob_map.shape[1] == 1

    # All recognition components produce 3D CTC logits [batch, T, vocab].
    rec_input = torch.rand(1, 3, RECOGNITION_HEIGHT, RECOGNITION_WIDTH)
    for recognizer in (
        model.zh_rec,
        model.ko_rec,
        model.en_rec,
        model.latin_rec,
        model.eslav_rec,
        model.thai_rec,
        model.greek_rec,
    ):
        logits = np.asarray(recognizer(rec_input))
        assert logits.ndim == 3
        assert logits.shape[0] == 1


@skip_clone_repo_check
def test_components_load() -> None:
    det = PPOCRv5Detector.from_pretrained()
    assert det is not None
    for lang in ("zh", "ko", "en", "latin", "eslav", "thai", "greek"):
        assert PPOCRv5Recognizer.from_pretrained(lang) is not None


def test_demo() -> None:
    demo_main(is_test=True)
