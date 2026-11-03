# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch

from qai_hub_models.models.templates.quicksrnet.external_repos.aimet_zoo.aimet_zoo_torch.quicksrnet.model.models import (
    QuickSRNetBase,
)


def _load_quicksrnet_source_model(
    scaling_factor: int,
    num_channels: int,
    num_intermediate_layers: int,
    use_ito_connection: bool,
) -> torch.nn.Module:
    return QuickSRNetBase(
        scaling_factor=scaling_factor,
        num_channels=num_channels,
        num_intermediate_layers=num_intermediate_layers,
        use_ito_connection=use_ito_connection,
    )
