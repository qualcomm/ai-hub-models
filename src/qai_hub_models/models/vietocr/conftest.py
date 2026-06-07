# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

import gc
import warnings

import pytest
import torch.jit._trace

from qai_hub_models.models.vietocr import Model
from qai_hub_models.scorecard.utils.testing import make_cached_from_pretrained_fixture


def pytest_configure(config: pytest.Config) -> None:
    # pytest is unable to figure out how to silence several PyTorch warning types from pyproject.toml settings,
    # so we apply a manual warning filter here instead.
    warnings.filterwarnings(action="ignore", category=torch.jit._trace.TracerWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning, module="torch.*")
    warnings.filterwarnings(action="ignore", category=FutureWarning, module="torch.*")
    warnings.filterwarnings(
        action="ignore", category=DeprecationWarning, module="torch.*"
    )


# Instantiate the model only once for all tests.
# Mock from_pretrained to always return the initialized model.
# This speeds up tests and limits memory leaks.
cached_from_pretrained = make_cached_from_pretrained_fixture(
    Model, skip_clone_repo=True
)


@pytest.fixture(scope="module", autouse=True)
def ensure_gc() -> None:
    gc.collect()
