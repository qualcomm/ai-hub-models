# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# Groot N1.7 constants
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

DEFAULT_EMBODIMENT_TAG: str = "oxe_droid_relative_eef_relative_joint"

VLM_SEQ_LEN: int = 256

NUM_DEEPSTACK: int = 3

# Action head / DiT (config.json: max_state_dim, max_action_dim).
MAX_STATE_DIM: int = 132
MAX_ACTION_DIM: int = 132

NUM_INFERENCE_TIMESTEPS: int = 4
