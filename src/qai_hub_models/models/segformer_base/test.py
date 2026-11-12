# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from qai_hub_models.models.segformer_base.demo import main as demo_main
from qai_hub_models.models.segformer_base.model import (
    INPUT_IMAGE_ADDRESS,
    SegformerBase,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_transform,
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import load_image

WEIGHTS = "nvidia/segformer-b0-finetuned-ade-512-512"


def test_task() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    source_model = SegformerForSemanticSegmentation.from_pretrained(WEIGHTS)
    qaihm_model = SegformerBase.from_pretrained(WEIGHTS)
    processed_sample_image = preprocess_PIL_image(load_image(INPUT_IMAGE_ADDRESS))

    with torch.no_grad():
        # original model output
        source_out = source_model(normalize_image_transform()(processed_sample_image))

        # Qualcomm AI Hub Model output
        qaihm_out = qaihm_model(processed_sample_image)
        for i in range(len(source_out)):
            assert np.allclose(source_out[i], qaihm_out[i])


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
