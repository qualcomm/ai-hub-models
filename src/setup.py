# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import functools
import os
import pathlib
import re
from typing import Literal

from setuptools import find_packages, setup
from setuptools_scm import get_version

PACKAGE_ROOT = (pathlib.Path(__file__).parent / "qai_hub_models").absolute()
MODELS_ROOT = PACKAGE_ROOT / "models"
SCORECARD_MODELS_ROOT = PACKAGE_ROOT / "scorecard" / "models"
REQS_FILENAME = "requirements.txt"

# Packages under qai_hub_models that are excluded from release builds
RELEASE_EXCLUDED_PACKAGES = ["scripts"]

# Dirs under models/ that are not model packages but should be kept.
RELEASE_WHITELISTED_MODELS_PACKAGES = ["templates", "utils"]

# When QAIHM_RELEASE_BUILD=1, exclude unpublished models and development files.
IS_RELEASE_BUILD = os.environ.get("QAIHM_RELEASE_BUILD", "0").lower() in [
    "1",
    "true",
    "yes",
]


def _get_model_status(
    model_dir: pathlib.Path,
) -> Literal["published", "unpublished", "pending"] | None:
    config_yaml = SCORECARD_MODELS_ROOT / model_dir.name / "internal-config.yaml"
    if not config_yaml.exists():
        return None
    with open(config_yaml) as f:
        for line in f:
            if m := re.match(r"^status:\s*(\S+)", line):
                out = m.group(1)
                assert out in {"published", "unpublished", "pending"}, (
                    f"Unknown model status: {out}"
                )
                return out  # type: ignore[return-type]
    return None


@functools.cache
def _get_unpublished_models() -> list[str]:
    """Return package patterns for unpublished models and non-model dirs."""
    return (
        [
            f"{model_dir.name}"
            for model_dir in MODELS_ROOT.iterdir()
            if model_dir.is_dir()
            and model_dir.name not in RELEASE_WHITELISTED_MODELS_PACKAGES
            and _get_model_status(model_dir) != "published"
        ]
        if IS_RELEASE_BUILD
        else []
    )


def _load_requirements(path: str | os.PathLike) -> list[str]:
    """
    Read requirements from the given path, return a list of pip-parseable requirements.
    Ignore / remove comments.
    """
    with open(path) as file:
        return [
            line.split("#")[0].strip()
            for line in file
            if line.strip()
            and not line.startswith("#")
            and not line.lstrip().startswith(
                "-"
            )  # pip options (e.g. --extra-index-url) are not valid PEP 508 specifiers
        ]


def _get_extras() -> dict[str, list[str]]:
    # Model deps come from `qai-hub-models install <model>`, which also runs the
    # manifest's pip commands that a pip extra can't express.
    with open(PACKAGE_ROOT / "requirements-dev.txt") as reqf:
        return {"dev": [line.split("#")[0].strip() for line in reqf]}


def _get_excluded_package_data() -> dict[str, list[str]]:
    if not IS_RELEASE_BUILD:
        return {}

    # Exclude data files from unpublished models. Setuptools treats files in
    # excluded sub-packages as data files of ancestor packages, so patterns
    # must be added at every level that has package_data globs.
    return {
        "qai_hub_models": [
            *[f"models/{model}/**" for model in _get_unpublished_models()],
            *[f"{package}/**" for package in RELEASE_EXCLUDED_PACKAGES],
            "models/**/release-assets.yaml",
        ],
    }


def _get_excluded_packages() -> list[str]:
    # Patterns are matched with fnmatch, where `*` spans dots -- so a bare
    # `<name>*` suffix would also exclude `<name>2`, `<name>_foo`, etc.
    excluded = ["qai_hub_models.*.external_repos.*"]
    if IS_RELEASE_BUILD:
        for package in RELEASE_EXCLUDED_PACKAGES:
            excluded += [f"qai_hub_models.{package}", f"qai_hub_models.{package}.*"]
        for model in _get_unpublished_models():
            excluded += [
                f"qai_hub_models.models.{model}",
                f"qai_hub_models.models.{model}.*",
            ]
    return excluded


def _get_install_requires() -> list[str]:
    version = get_version(root="..")
    reqs = _load_requirements(PACKAGE_ROOT / REQS_FILENAME)
    reqs.append(f"qai_hub_models_cli=={version}")
    return reqs


setup(
    packages=find_packages(
        include=["qai_hub_models*"], exclude=_get_excluded_packages()
    ),
    install_requires=_get_install_requires(),
    extras_require=_get_extras(),
    exclude_package_data=_get_excluded_package_data(),
)
