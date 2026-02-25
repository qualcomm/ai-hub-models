# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import timm
from timm.models.nasnet import CellStem1, FirstCell
from typing_extensions import Self

from qai_hub_models.models._shared.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.models.common import Precision
from qai_hub_models.models.nasnet.model_patches import (
    CellStem1_forward,
    FirstCell_forward,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "nasnetalarge.tf_in1k"


class NASNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, checkpoint_path: str = DEFAULT_WEIGHTS) -> Self:
        # Make Functional in QNN and reduce inference latency for quantized variant
        CellStem1.forward = CellStem1_forward
        FirstCell.forward = FirstCell_forward

        model = timm.create_model(checkpoint_path, pretrained=True)
        return cls(model, transform_input=True)

    @staticmethod
    def get_hub_litemp_percentage(_: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 1
