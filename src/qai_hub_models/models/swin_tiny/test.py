# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
import torchvision.models as tv_models

from qai_hub_models.models.swin_tiny.demo import main as demo_main
from qai_hub_models.models.swin_tiny.model import MODEL_ID, SwinTiny
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.imagenet_classifier.test_utils import (  # noqa: F401
    imagenet_sample_torch,
    run_imagenet_classifier_test,
)


def test_numerical(imagenet_sample_torch: torch.Tensor) -> None:  # noqa: F811
    # Ensure that the optimized SwinTiny matches the original one numerically
    x = imagenet_sample_torch
    model_opt = SwinTiny.from_pretrained().eval()
    model_orig = tv_models.swin_t(weights="IMAGENET1K_V1").eval()
    np.testing.assert_allclose(
        model_opt(x).detach().numpy(),
        model_orig(normalize_image_torchvision(x)).detach().numpy(),
        atol=1e-5,
        rtol=1e-3,
    )


def test_task() -> None:
    run_imagenet_classifier_test(
        SwinTiny.from_pretrained(), MODEL_ID, probability_threshold=0.53
    )


def test_demo() -> None:
    # Verify demo does not crash
    demo_main(is_test=True)
