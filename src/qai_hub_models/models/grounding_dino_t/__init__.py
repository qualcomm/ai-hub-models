# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from .app import GroundingDINOTCollectionApp as App
from .model import DEFAULT_PRECISION, MODEL_ID
from .model import GroundingDINOTCollection as Model

__all__ = ["DEFAULT_PRECISION", "MODEL_ID", "App", "Model"]
