# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Guard the release wheel's package include/exclude lists, and what cites them.

Two failure modes, both silent until a user hits them on PyPI.

A published recipe whose source is dropped from the wheel still ships its
metadata (setuptools keeps files of excluded sub-packages as data files of an
ancestor package). The resulting metadata-only folder imports as an implicit
namespace package, so failures surface far downstream as a bare
``AttributeError: module '...' has no attribute 'Model'``.

Separately, shipped code may *name* an excluded package. ``configure_dataset``
lived in ``qai_hub_models/scripts/``, and fourteen shipped dataset modules told
users to run it, so a PyPI user who followed the printed instruction got
``No module named qai_hub_models.scripts``. The reference was inside a string,
so no import ever failed and nothing caught it.
"""

from __future__ import annotations

import os
import re
import runpy
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

SETUP_PY = Path(__file__).parents[2] / "setup.py"
MODELS_ROOT = Path(__file__).parents[1] / "models"
PACKAGE_ROOT = Path(__file__).parents[1]
SCORECARD_MODELS_ROOT = PACKAGE_ROOT / "scorecard" / "models"

# Directories that are themselves excluded from release builds, or are test-only,
# so a reference from inside them is fine.
_SKIP_DIRS = {"scripts", "test"}

# Pre-existing references, recorded so new ones fail. All three are dev-only
# paths that cannot be reached from a release wheel:
#   - cli/proto_api.py -- get_manifest_proto is for dev installs.
#   - cli/generate_files.py -- "generate-files is dev-only".
#   - models/templates/llm/llm_response_evaluator.py -- the reference is a
#     subprocess argv for a separate hand-built `qaihm-dev-grader` venv, not an
#     import here. Without that venv the code raises before spawning anything.
_KNOWN_OFFENDERS = {
    "cli/proto_api.py",
    "cli/generate_files.py",
    "models/templates/llm/llm_response_evaluator.py",
}


def _run_release_setup() -> tuple[dict[str, Any], dict[str, Any]]:
    """Exec setup.py as a release build; return (module globals, setup kwargs).

    ``setup()`` is called unconditionally at module scope, so it must be
    patched out rather than avoided via ``run_name``. ``find_packages`` walks
    the cwd, so this has to run from setup.py's own directory.
    """
    captured: dict[str, Any] = {}
    # setuptools_scm is a build-system requirement, so pip installs it into an
    # isolated build env and it is absent from the runtime venv CI tests in.
    scm = types.ModuleType("setuptools_scm")
    scm.get_version = lambda **kwargs: "0.0.1"  # type: ignore[attr-defined]
    prev_cwd = os.getcwd()
    os.chdir(SETUP_PY.parent)
    try:
        with (
            patch.dict(os.environ, {"QAIHM_RELEASE_BUILD": "1"}),
            patch.dict(sys.modules, {"setuptools_scm": scm}),
            patch("setuptools.setup", lambda **kw: captured.update(kw)),
        ):
            module_globals = runpy.run_path(str(SETUP_PY), run_name="__main__")
    finally:
        os.chdir(prev_cwd)
    return module_globals, captured


def _published_recipes_with_source() -> list[str]:
    out = []
    for d in sorted(MODELS_ROOT.iterdir()):
        manifest = d / "manifest.yaml"
        if not d.is_dir() or not manifest.exists():
            continue
        if not (d / "__init__.py").exists():
            continue
        config = SCORECARD_MODELS_ROOT / d.name / "internal-config.yaml"
        with open(config) as f:
            if (yaml.safe_load(f) or {}).get("status") == "published":
                out.append(d.name)
    return out


@pytest.fixture(scope="module")
def release_setup() -> tuple[dict[str, Any], dict[str, Any]]:
    return _run_release_setup()


@pytest.mark.parametrize("model_id", _published_recipes_with_source())
def test_published_recipe_source_is_in_release_wheel(
    model_id: str, release_setup: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Every published recipe with source must survive the release exclude lists.

    Regression guard: the exclude patterns are fnmatch globs where ``*`` spans
    dots, so a bare ``<name>*`` suffix silently dropped ``sam2``/``sam3`` via
    the unpublished ``sam`` recipe.
    """
    packages = set(release_setup[1]["packages"])
    assert f"qai_hub_models.models.{model_id}" in packages


def test_release_excludes_are_not_prefix_globs(
    release_setup: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """No exclude pattern may end in ``<name>*``, which over-matches siblings."""
    patterns = release_setup[0]["_get_excluded_packages"]()
    offenders = [p for p in patterns if p.endswith("*") and not p.endswith(".*")]
    assert not offenders, (
        f"These patterns match sibling packages by prefix: {offenders}. "
        "Use both `<pkg>` and `<pkg>.*` instead of `<pkg>*`."
    )


def _release_excluded_packages() -> list[str]:
    """Read ``RELEASE_EXCLUDED_PACKAGES`` out of setup.py without importing it.

    Parametrization happens at collection time, so this cannot use the
    ``release_setup`` fixture; ``test_excluded_packages_regex_matches_setup_py``
    checks the two agree.
    """
    text = SETUP_PY.read_text()
    match = re.search(r"^RELEASE_EXCLUDED_PACKAGES = \[(.*?)\]", text, re.MULTILINE)
    assert match, "RELEASE_EXCLUDED_PACKAGES not found in setup.py"
    return re.findall(r"[\"']([^\"']+)[\"']", match.group(1))


def _shipped_python_files() -> list[Path]:
    return [
        p
        for p in PACKAGE_ROOT.rglob("*.py")
        if not _SKIP_DIRS & set(p.relative_to(PACKAGE_ROOT).parts)
    ]


def test_excluded_packages_regex_matches_setup_py(
    release_setup: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """The collection-time regex must agree with the executed setup.py."""
    assert _release_excluded_packages() == release_setup[0]["RELEASE_EXCLUDED_PACKAGES"]


@pytest.mark.parametrize("excluded", _release_excluded_packages())
def test_shipped_code_does_not_reference_excluded_packages(excluded: str) -> None:
    offenders: list[str] = []
    needle = f"qai_hub_models.{excluded}"
    for path in _shipped_python_files():
        rel = path.relative_to(PACKAGE_ROOT).as_posix()
        if rel in _KNOWN_OFFENDERS:
            continue
        for lineno, line in enumerate(
            path.read_text(errors="ignore").splitlines(), start=1
        ):
            if needle in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        f"Shipped code references `{needle}`, which release builds exclude. "
        "Users hitting this get ModuleNotFoundError on a PyPI install. Either "
        "move the code into a shipped package, or add the file to "
        "_KNOWN_OFFENDERS with a reason if the path is genuinely dev-only.\n"
        + "\n".join(offenders)
    )


def test_known_offenders_all_still_exist() -> None:
    """Keeps the allowlist from silently outliving the files it excuses."""
    missing = [rel for rel in _KNOWN_OFFENDERS if not (PACKAGE_ROOT / rel).exists()]
    assert not missing, f"Stale _KNOWN_OFFENDERS entries: {missing}"
