# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock

from qai_hub_models.utils.external_repo import (
    IS_PIP_PACKAGE,
    setup_external_repos,
)

RECIPE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_REPO_PATHS: dict[str, Path] = {}

if not TYPE_CHECKING:
    with FileLock(Path(__file__).resolve().parent / ".setup.lock"):
        EXTERNAL_REPO_PATHS = setup_external_repos(RECIPE_DIR)
    if IS_PIP_PACKAGE:
        __path__ = [str(p.parent) for p in EXTERNAL_REPO_PATHS.values()]
