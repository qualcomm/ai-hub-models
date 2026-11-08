# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.templates.sesr.external_repos.aimet_zoo.aimet_zoo_torch.common.super_resolution.models import (
    SESRRelease,
)

# SESR original repo is here: https://github.com/ARM-software/sesr
# But this is all written in TF and Keras. Torch version is in AIMET Zoo


def _load_sesr_source_model(
    scaling_factor: int, num_channels: int, num_lblocks: int
) -> torch.nn.Module:
    return SESRRelease(
        scaling_factor=scaling_factor,
        num_channels=num_channels,
        num_lblocks=num_lblocks,
    )
