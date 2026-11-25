# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.models.templates.stereo_depth.app import StereoDisparityApp
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.input_spec import InputSpec


class CREStereoApp(StereoDisparityApp):
    """End-to-end pipeline for the CREStereo disparity model."""

    def __init__(self, model: ExecutableModelProtocol, input_spec: InputSpec) -> None:
        super().__init__(model, input_spec, image_layout="BGR", align_divisor=8)

    def _prepare_model_inputs(
        self,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # CREStereo also needs half-resolution inputs
        left_half = F.interpolate(
            left_t, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        right_half = F.interpolate(
            right_t, scale_factor=0.5, mode="bilinear", align_corners=True
        )
        return (left_t, right_t, left_half, right_half)
