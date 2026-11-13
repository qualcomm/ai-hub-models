# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.templates.cityscapes_segmentation.ffnet_model import (
    _load_ffnet_source_model,
)
from qai_hub_models.models.templates.cityscapes_segmentation.model import (
    TEST_CITYSCAPES_LIKE_IMAGE_ASSET,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_model import BaseModel


def run_test_off_target_numerical(
    ffnet_cls: type[BaseModel],
    variant_name: str,
    relax_numerics: bool = False,
) -> None:
    """Verify that raw (numeric) outputs of both (qaism and non-qaism) networks are the same."""
    source_model = _load_ffnet_source_model(variant_name)
    if "_pre_down" in variant_name:
        source_model.pre_downsampling = False  # type: ignore[assignment]
    qaism_model = ffnet_cls.from_pretrained()
    input_spec = qaism_model.get_input_spec()
    _, _, h, w = input_spec["image"][0]

    pil_image = load_image(TEST_CITYSCAPES_LIKE_IMAGE_ASSET)
    processed_sample_image = preprocess_PIL_image(pil_image.resize((w, h)))
    normalized_image = normalize_image_torchvision(processed_sample_image)

    with torch.no_grad():
        source_out = source_model(normalized_image)
        qaism_out = qaism_model(processed_sample_image)

        if relax_numerics:
            # At least 90% of pixels should have original prediction
            assert (source_out.argmax(1) == qaism_out.argmax(1)).float().mean() > 0.9
        else:
            np.testing.assert_array_almost_equal(source_out, qaism_out)
