# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.mobile_vit.model import MOBILEVIT_TRANSFORM
from qai_hub_models.models.templates.imagenet_classifier.app import (
    ImagenetClassifierApp,
)
from qai_hub_models.protocols import ExecutableModelProtocol


class MobileVITApp(ImagenetClassifierApp):
    """Encapsulates the logic for running inference on a MobileVIT model."""

    def __init__(self, model: ExecutableModelProtocol) -> None:
        super().__init__(
            model, normalization_in_network=True, transform=MOBILEVIT_TRANSFORM
        )
