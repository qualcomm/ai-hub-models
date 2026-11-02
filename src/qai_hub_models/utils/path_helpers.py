# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import subprocess
from pathlib import Path

from qai_hub_models._version import __version__

MODELS_PACKAGE_NAME = "models"
QAIHM_PACKAGE_NAME = "qai_hub_models"
QAIHM_PACKAGE_ROOT = Path(__file__).parent.parent
QAIHM_PACKAGE_SRC_ROOT = QAIHM_PACKAGE_ROOT.parent
QAIHM_REPO_ROOT = QAIHM_PACKAGE_SRC_ROOT.parent


def is_internal_repo() -> bool:
    """Check if running from the internal repository via git remote URL."""
    try:
        git_config = QAIHM_REPO_ROOT / ".git" / "config"
        return "ai-hub-models-internal" in git_config.read_text()
    except (OSError, ValueError):
        return False


def _get_qaihm_models_root(package_root: Path = QAIHM_PACKAGE_ROOT) -> Path:
    return package_root / MODELS_PACKAGE_NAME


QAIHM_MODELS_ROOT = _get_qaihm_models_root()

# Holds each in-tree model's internal-config.yaml, including its publish status.
SCORECARD_MODELS_ROOT = QAIHM_PACKAGE_ROOT / "scorecard" / "models"


def _get_all_models(models_root: Path = QAIHM_MODELS_ROOT) -> list[str]:
    all_models: list[str] = []
    for subdir in models_root.iterdir():
        if not subdir.is_dir():
            continue
        if (subdir / "manifest.yaml").exists():
            all_models.append(subdir.name)
    return all_models


MODEL_IDS = sorted(_get_all_models())


def get_next_free_path(path: str | os.PathLike, delim: str = "-") -> Path:
    """Adds an incrementing number at the end of the given path until a non-existant filename is found."""
    path_without_ext, ext = os.path.splitext(path)

    counter = 1
    while os.path.exists(path):
        # Create new filename with underscore and counter
        path = f"{path_without_ext}{delim}{counter}{ext}"
        counter += 1

    return Path(path)


def get_git_branch() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=False,
            capture_output=True,
        )
        if "not a git repository" in res.stderr.decode():
            # repo not found, this must be a release
            return f"release_{__version__}"
        if res.stderr:
            # unknown why git failed
            return f"unknown_git_branch_error__{__version__}"
        # return branch name
        return res.stdout.decode()[:-1]
    except FileNotFoundError:
        # git not found, this must be a release
        return f"release_{__version__}"
