# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Resolve a recipe's pipeline context (model class, display name, manifest)
from its on-disk folder. The single unit of identity throughout the
export/evaluate/install stack is ``source_dir: Path`` — the recipe folder
that contains ``manifest.yaml``. ``model_id`` is a display-only string
derived from ``source_dir.name`` and is only accepted at the CLI top
layer, where :func:`resolve_recipe_dir` converts it into a folder path.
"""

from __future__ import annotations

import difflib
import functools
import importlib
import sys
from pathlib import Path
from typing import Any

from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    QAIHM_MODELS_ROOT,
    SCORECARD_MODELS_ROOT,
)

_LIST_MODEL_IDS_HINT = "Run `qai-hub-models models -q` to list model IDs."


def _normalize_model_name(name: str) -> str:
    """Fold the separators users get wrong: `MobileNet-v2` -> `mobilenet_v2`."""
    return name.lower().replace("-", "_").replace(" ", "_")


@functools.cache
def _display_name_to_id() -> dict[str, str]:
    """Map normalized manifest display names to model ids.

    Reads every manifest, so call this only when a target has already failed
    to resolve.
    """
    mapping: dict[str, str] = {}
    for model_id in MODEL_IDS:
        try:
            name = QAIHMModelManifest.from_model(model_id).name
        except Exception:
            # An unreadable manifest must not mask the caller's real error.
            continue
        if name:
            mapping.setdefault(_normalize_model_name(name), model_id)
    return mapping


def unknown_model_error(target: str) -> str:
    """Build the error text for a target that is neither a model id nor a folder."""
    normalized = _normalize_model_name(target)
    by_display_name = _display_name_to_id()

    if resolved := by_display_name.get(normalized):
        return (
            f"{target!r} is a model display name, not a model ID. "
            f"Use {resolved!r} instead."
        )

    candidates = {_normalize_model_name(m): m for m in MODEL_IDS} | by_display_name
    if matches := difflib.get_close_matches(normalized, candidates, n=1):
        return (
            f"Unknown model {target!r}. Did you mean {candidates[matches[0]]!r}?\n\n"
            f"{_LIST_MODEL_IDS_HINT}"
        )

    return (
        f"Unknown model {target!r}, and no folder of that name exists here.\n\n"
        f"  * {_LIST_MODEL_IDS_HINT}\n"
        "  * To use a local recipe folder, pass a path: ./my_model"
    )


def resolve_recipe_dir(target: str | Path) -> Path:
    """Convert a CLI target (folder path or installed model id) to a folder path.

    Called once at the CLI top layer; everything downstream operates on
    ``Path``. Accepts:

    * A **folder path** (str or Path, absolute or relative). The folder
      must contain a ``manifest.yaml``.
    * A **bare installed model id** (e.g. ``"mobilenet_v2"``) — resolved
      to ``<installed_pkg>/models/<id>/``.
    * A **bare folder name** that matches a directory in the current
      working directory — resolved as a folder path. Installed model ids
      always win over cwd folders of the same name.

    Display names (``"MobileNetV2"``) are NOT accepted.
    """
    if isinstance(target, Path) or looks_like_path(str(target)):
        source_dir = Path(target).resolve()
        if not source_dir.is_dir():
            raise CLIUserError(
                f"{target!r} is not a directory. Point at a recipe folder "
                "that contains manifest.yaml."
            )
    else:
        target_str = str(target)
        if target_str in MODEL_IDS:
            source_dir = QAIHM_MODELS_ROOT / target_str
        elif (cwd_folder := Path(target_str)).is_dir():
            source_dir = cwd_folder.resolve()
        else:
            raise CLIUserError(unknown_model_error(target_str))
    if not (source_dir / "manifest.yaml").exists():
        raise CLIUserError(
            f"{source_dir} does not contain a manifest.yaml — cannot resolve "
            "as a model recipe."
        )
    return source_dir


def looks_like_path(target: str) -> bool:
    """Return True if *target* is meant as a filesystem path, not a bare id."""
    return (
        "/" in target
        or "\\" in target
        or target.startswith((".", "~"))
        or Path(target).is_absolute()
    )


class RecipeSourceUnavailableError(CLIUserError):
    """A recipe folder holds metadata but no Python source."""


def import_recipe_module(source_dir: Path) -> Any:
    """Import the recipe package at *source_dir* and return its module.

    For recipes inside the installed package, this returns the already-loaded
    ``qai_hub_models.models.<id>`` module. For standalone folders, the folder's
    parent is added to ``sys.path`` so ``import <folder_name>`` works — and
    any sub-imports the recipe uses (e.g. ``<folder_name>.external_repos.<repo>``)
    resolve consistently.

    Raises :class:`RecipeSourceUnavailableError` if the folder ships metadata
    only.
    """
    # Without this, a metadata-only folder imports as an implicit namespace
    # package and every attribute lookup fails with a bare AttributeError.
    if not (source_dir / "__init__.py").exists():
        raise RecipeSourceUnavailableError(
            f"Recipe `{source_dir.name}` has no Python source in this "
            f"installation — {source_dir} has no __init__.py, so there is "
            "nothing to export.\n"
            "Some models are listed for their published metrics but ship no "
            "recipe. Those metrics are still available via "
            f"`qai-hub-models info/perf/numerics {source_dir.name}`; to export "
            "this model, contact ai-hub-support@qti.qualcomm.com."
        )

    module_name = source_dir.name
    try:
        rel = source_dir.resolve().relative_to(QAIHM_MODELS_ROOT.resolve())
        if len(rel.parts) == 1:
            module_name = f"qai_hub_models.models.{rel.parts[0]}"
    except ValueError:
        pass

    if module_name == source_dir.name:
        parent = str(source_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)

    return importlib.import_module(module_name)


def resolve_manifest(source_dir: Path) -> QAIHMModelManifest:
    """Load the manifest for the recipe at *source_dir*."""
    return QAIHMModelManifest.from_yaml(source_dir / "manifest.yaml")


def is_in_tree_recipe(source_dir: Path) -> bool:
    """True if *source_dir* is a model folder under ``qai_hub_models/models/``."""
    try:
        source_dir.resolve().relative_to(QAIHM_MODELS_ROOT.resolve())
    except ValueError:
        return False
    return True


def resolve_publish_status(source_dir: Path) -> MODEL_STATUS | None:
    """
    Publish status of the recipe at *source_dir*, or None for external recipes.

    Read from the raw internal-config.yaml so the public CLI never imports the
    internal-only scorecard package. In-tree recipes without one are a repo bug.
    """
    if not is_in_tree_recipe(source_dir):
        return None
    config_path = SCORECARD_MODELS_ROOT / source_dir.name / "internal-config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{source_dir.name} is an in-tree recipe but has no {config_path}."
        )
    return MODEL_STATUS((load_yaml(config_path) or {})["status"])


def resolve_model_cls(source_dir: Path) -> Any:
    """Return the ``Model`` class exported from the recipe's ``__init__.py``."""
    return import_recipe_module(source_dir).Model


def resolve_model_app_cls(source_dir: Path) -> Any | None:
    """Return the ``App`` class exported from the recipe's package, or ``None``."""
    return getattr(import_recipe_module(source_dir), "App", None)


def resolve_model_display_name(source_dir: Path) -> str:
    """Resolve the human-readable model name from ``manifest.yaml``."""
    return resolve_manifest(source_dir).name or source_dir.name


def resolve_model_id(source_dir: Path) -> str:
    """Return the recipe's model id — its folder name."""
    return source_dir.name
