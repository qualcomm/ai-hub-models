# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import torch
from torch.nn import functional as F
from transformers import Mask2FormerForUniversalSegmentation

from qai_hub_models.models.mask2former.demo import (
    INPUT_IMAGE_ADDRESS,
)
from qai_hub_models.models.mask2former.demo import main as demo_main
from qai_hub_models.models.mask2former.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    Mask2Former,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

WEIGHTS = "facebook/mask2former-swin-tiny-coco-panoptic"
OUTPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/output_image.png"
)


def test_task() -> None:
    """Verify that raw (numeric) outputs of both (QAIHM and non-qaihm) networks are the same."""
    source_model = Mask2FormerForUniversalSegmentation.from_pretrained(WEIGHTS)
    qaihm_model = Mask2Former.from_pretrained(WEIGHTS)
    processed_sample_image = preprocess_PIL_image(load_image(INPUT_IMAGE_ADDRESS))
    normalized_image = normalize_image_torchvision(processed_sample_image)
    with torch.no_grad():
        # original model output
        source_out = source_model(normalized_image, return_dict=False)
        source_out = (F.softmax(source_out[0], dim=-1).max(-1), source_out[1])
        source_out = [source_out[0][0], source_out[0][1], source_out[1]]

        # Qualcomm AI Hub Model output
        qaihm_out = qaihm_model(processed_sample_image)
        for i in range(len(source_out)):
            assert np.allclose(source_out[i], qaihm_out[i], atol=0.001, rtol=0.001)


def test_demo() -> None:
    # Run demo and verify it does not crash
    demo_main(is_test=True)
