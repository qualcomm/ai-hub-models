# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models.silero_vad.app import SileroVADApp
from qai_hub_models.models.silero_vad.model import MODEL_ID, SileroVAD

Model = SileroVAD
App = SileroVADApp

__all__ = ["MODEL_ID", "App", "Model", "SileroVAD", "SileroVADApp"]
