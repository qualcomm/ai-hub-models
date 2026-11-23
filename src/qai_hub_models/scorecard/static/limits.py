# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.utils.device import FormFactor

# Number of AI Hub Workbench inference jobs for static models that can be submitted per-BU.
BU_INFERENCE_JOB_LIMIT = 100

# Number of AI Hub Workbench inference jobs for static models that can be submitted for each device form factor per-BU.
BU_INFERENCE_JOB_LIMIT_BY_FORM_FACTOR: dict[FormFactor, int] = {FormFactor.AUTO: 50}
