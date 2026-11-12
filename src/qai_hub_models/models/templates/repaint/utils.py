# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


def preprocess_inputs(
    pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
    mask_pixel_values_or_image: torch.Tensor | np.ndarray | Image,
) -> dict[str, torch.Tensor]:
    NCHW_fp32_torch_frames = app_to_net_image_inputs(pixel_values_or_image)[1]
    NCHW_fp32_torch_masks = app_to_net_image_inputs(mask_pixel_values_or_image)[1]

    # Broadcast a single mask to match the number of input frames.
    if NCHW_fp32_torch_masks.shape[0] == 1 and NCHW_fp32_torch_frames.shape[0] > 1:
        NCHW_fp32_torch_masks = NCHW_fp32_torch_masks.tile(
            (NCHW_fp32_torch_frames.shape[0], 1, 1, 1)
        )

    assert NCHW_fp32_torch_masks.shape[0] == NCHW_fp32_torch_frames.shape[0], (
        f"Mask batch size {NCHW_fp32_torch_masks.shape[0]} must equal "
        f"frame batch size {NCHW_fp32_torch_frames.shape[0]}"
    )

    # Mask input image
    return {"image": NCHW_fp32_torch_frames, "mask": NCHW_fp32_torch_masks}
