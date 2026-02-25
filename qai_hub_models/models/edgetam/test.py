# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.sam2.test_utils import run_sam2_numerical_test
from qai_hub_models.models.edgetam.demo import IMAGE_ADDRESS
from qai_hub_models.models.edgetam.demo import main as demo_main
from qai_hub_models.models.edgetam.model import (
    DEFAULT_MODEL_TYPE,
    EdgeTAM,
    EdgeTAMLoader,
)


def test_e2e_numerical() -> None:
    """Verify our driver produces the correct segmentation as source PyTorch model"""
    run_sam2_numerical_test(
        EdgeTAM,
        EdgeTAMLoader,
        DEFAULT_MODEL_TYPE,
        IMAGE_ADDRESS,
        mask_comparison_atol=0.05,
        squeeze_mask=True,
    )


def test_demo() -> None:
    demo_main(is_test=True)
