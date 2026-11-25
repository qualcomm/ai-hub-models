# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.templates.stereo_depth.app import StereoDisparityApp
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.input_spec import InputSpec


class IGEVPlusPlusApp(StereoDisparityApp):
    """End-to-end pipeline for the IGEV++ stereo disparity model."""

    def __init__(self, model: ExecutableModelProtocol, input_spec: InputSpec) -> None:
        super().__init__(model, input_spec, image_layout="RGB", align_divisor=32)

    def _prepare_model_inputs(
        self,
        left_t: torch.Tensor,
        right_t: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # Pass [0, 1] tensors directly — IGEVPlusPlus.forward() scales
        # to [0, 255] internally before calling IGEVStereo.
        return (left_t, right_t)
