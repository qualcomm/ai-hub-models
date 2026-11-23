# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

import pytest

from qai_hub_models.models.overlap_transformer import MODEL_ID, Model
from qai_hub_models.scorecard.envvars import ArtifactsDirEnvvar
from qai_hub_models.scripts.run_scorecard_single import Stage, run_scorecard_single
from qai_hub_models.utils.validation import perform_runtime_model_validation


@pytest.mark.compile
def test_runtime_model_validation() -> None:
    perform_runtime_model_validation(Model, MODEL_ID)


@pytest.mark.compile
def test_compile() -> None:
    assert run_scorecard_single(
        MODEL_ID,
        ArtifactsDirEnvvar.get(),
        end_before=Stage.INFERENCE,
    )
