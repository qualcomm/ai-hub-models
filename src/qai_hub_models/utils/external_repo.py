# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import dataclasses
import hashlib
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    import git  # noqa: TID251
except ImportError as e:
    raise ImportError(
        "GitPython is required for external repo support. "
        "Please install git (https://github.com/git-guides/install-git) "
        "and then run: pip install gitpython"
    ) from e

from qai_hub_models_cli.common import LOCAL_STORE_DEFAULT_PATH

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.asset_loaders import (
    EXECUTING_IN_CI_ENVIRONMENT,
    query_yes_no,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

EXTERNAL_REPOS_CACHE_DIR = Path(LOCAL_STORE_DEFAULT_PATH) / "external_repos"
USE_SYMLINK_CACHE = platform.system() != "Windows"
IS_PIP_PACKAGE = "site-packages" in Path(__file__).resolve().parts
MYPY_IGNORE = "# mypy: ignore-errors\n"

# Bump this when postprocess_repo behavior changes (e.g. import rewriting logic).
# This invalidates all cached clones so they get re-cloned with the new postprocessing.
POSTPROCESS_VERSION = "3"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RepoConfig:
    """Configuration for a single external repository dependency."""

    repo_url: str
    commit_sha: str
    patches_filename: str | None = None


# ---------------------------------------------------------------------------
# Config loading and path resolution
# ---------------------------------------------------------------------------


def _load_repo_configs(manifest_path: Path) -> dict[str, RepoConfig]:
    """Read ``external_repos:`` from manifest.yaml.

    Parameters
    ----------
    manifest_path
        Path to the manifest.yaml file.

    Returns
    -------
    dict[str, RepoConfig]
        Mapping of repo name to configuration, or empty dict if no external
        repos are declared or the manifest doesn't exist.
    """
    if not manifest_path.exists():
        return {}
    manifest = QAIHMModelManifest.from_yaml(manifest_path)
    if not manifest.external_repos:
        return {}
    return {
        name: RepoConfig(
            repo_url=cfg.repo_url,
            commit_sha=cfg.commit_sha,
            patches_filename=cfg.patches_filename,
        )
        for name, cfg in manifest.external_repos.items()
    }


def _resolve_external_repos(
    recipe_dir: Path,
) -> tuple[dict[str, RepoConfig], Path]:
    """Resolve repo configs and external_repos path from a recipe folder."""
    external_repos_path = recipe_dir / "external_repos"
    manifest_path = recipe_dir / "manifest.yaml"
    return _load_repo_configs(manifest_path), external_repos_path


# ---------------------------------------------------------------------------
# Hashing and caching
# ---------------------------------------------------------------------------


def compute_content_hash(
    repo_url: str,
    commit_sha: str,
    patch_contents: str = "",
) -> str:
    """Compute a hash of the repo configuration that produces the clone.

    Parameters
    ----------
    repo_url
        Git repository URL.
    commit_sha
        Pinned commit SHA.
    patch_contents
        Contents of the patch file to apply, or empty string.

    Returns
    -------
    str
        A 16-character hex hash.
    """
    data = (
        repo_url
        + "\n"
        + commit_sha
        + "\n"
        + patch_contents
        + "\n"
        + POSTPROCESS_VERSION
    )
    return hashlib.shake_256(data.encode()).hexdigest(8)


def _cache_namespace(external_repos_path: Path) -> str:
    """Derive a cache namespace from the external repos path.

    Parameters
    ----------
    external_repos_path
        Absolute path to the external_repos directory, e.g.
        ``.../models/gkt/external_repos`` or ``.../templates/centernet/external_repos``.

    Returns
    -------
    str
        Cache namespace, e.g. ``"models/gkt"`` or ``"templates/centernet"``.

    Raises
    ------
    ValueError
        If the path does not contain ``external_repos``.
    """
    parts = external_repos_path.parts
    if "external_repos" not in parts:
        raise ValueError(
            f"Cannot derive cache namespace from {external_repos_path}. "
            "Expected path to contain 'external_repos'."
        )
    idx = parts.index("external_repos")
    parent = parts[idx - 1]
    is_template = idx >= 2 and parts[idx - 2] == "templates"
    return str(Path("templates" if is_template else "models", parent))


def get_cache_dir(external_repos_path: Path, repo_name: str, content_hash: str) -> Path:
    """Return the cache directory path for a given repo.

    Parameters
    ----------
    external_repos_path
        Path to the external_repos directory.
    repo_name
        Name of the repo.
    content_hash
        Content hash of the repo configuration.

    Returns
    -------
    Path
        Absolute path to the cache directory.
    """
    return (
        EXTERNAL_REPOS_CACHE_DIR
        / _cache_namespace(external_repos_path)
        / repo_name
        / content_hash
    )


def get_patch_contents(config: RepoConfig, external_repos_path: Path) -> str:
    """Read patch file contents for a repo config.

    Parameters
    ----------
    config
        Repo configuration with optional ``patches_filename``.
    external_repos_path
        Path to the directory containing the patch file.

    Returns
    -------
    str
        Contents of the patch file, or empty string if none.
    """
    if config.patches_filename:
        patch_path = external_repos_path / config.patches_filename
        if not patch_path.exists():
            raise FileNotFoundError(
                f"Patch file '{config.patches_filename}' not found at {patch_path}"
            )
        return patch_path.read_text()
    return ""


def compute_repo_hashes(
    repo_configs: dict[str, RepoConfig],
    external_repos_path: Path,
) -> dict[str, str]:
    """Compute content hash for each configured repo.

    Parameters
    ----------
    repo_configs
        Mapping of repo name to configuration.
    external_repos_path
        Path to the external_repos directory (for reading patch files).

    Returns
    -------
    dict[str, str]
        Mapping of repo name to content hash.
    """
    return {
        repo_name: compute_content_hash(
            config.repo_url,
            config.commit_sha,
            get_patch_contents(config, external_repos_path),
        )
        for repo_name, config in repo_configs.items()
    }


# ---------------------------------------------------------------------------
# Clone, patch, and postprocess
# ---------------------------------------------------------------------------


def _shallow_clone(url: str, commit_sha: str, target: Path) -> None:
    """Shallow-fetch a single commit into a target directory.

    Falls back to a full fetch if the remote doesn't support shallow fetch
    of arbitrary SHAs.

    Parameters
    ----------
    url
        Git repository URL.
    commit_sha
        Commit SHA to fetch.
    target
        Local directory to clone into.
    """
    repo = git.Repo.init(target)
    # External repos are imported as Python source only; git-lfs assets are
    # never needed. The bare `git init` + `git fetch <url>` clone registers no
    # remote, so git-lfs can't resolve an endpoint and fails the checkout with
    # "batch request: missing protocol" in CI. Skip the smudge filter.
    repo.git.update_environment(GIT_LFS_SKIP_SMUDGE="1")
    try:
        repo.git.fetch(url, commit_sha, depth=1)
    except git.GitCommandError:
        repo.git.fetch(url, commit_sha)
    repo.git.checkout("FETCH_HEAD")


def _apply_patch(target: Path, patch_path: Path) -> None:
    """Apply a patch file to a cloned repo.

    Parameters
    ----------
    target
        Path to the cloned repo directory.
    patch_path
        Absolute path to the ``.diff`` patch file.
    """
    repo = git.Repo(target)
    repo.git.apply(patch_path, ignore_whitespace=True)


def _derive_package_path(repo_path: Path) -> str:
    """Derive the full Python package path for a cloned repo.

    Parameters
    ----------
    repo_path
        Absolute path to the repo directory,
        e.g. ``/home/user/src/qai_hub_models/models/gkt/external_repos/gkt``.

    Returns
    -------
    str
        Dotted package path, e.g. ``"qai_hub_models.models.gkt.external_repos.gkt"``.

    Raises
    ------
    ValueError
        If the path structure cannot be parsed.
    """
    parts = repo_path.resolve().parts

    # In-tree: path contains qai_hub_models/models/<model_id>/external_repos/<repo>
    if "qai_hub_models" in parts:
        idx = parts.index("qai_hub_models")
        return ".".join(parts[idx:])

    # Standalone: path is <any_prefix>/<model_id>/external_repos/<repo>
    # Derive package path as <model_id>.external_repos.<repo>
    if "external_repos" in parts:
        idx = parts.index("external_repos")
        if idx > 0:
            model_id = parts[idx - 1]
            remaining = parts[idx:]
            return ".".join([model_id, *remaining])

    raise ValueError(
        f"Cannot derive package path from {repo_path}. "
        "Expected 'qai_hub_models' or 'external_repos' in the path."
    )


def _get_top_level_packages(path: Path) -> set[str]:
    """Return names of top-level directories and .py modules in the repo root.

    Parameters
    ----------
    path
        Path to the cloned repo root.

    Returns
    -------
    set[str]
        Set of top-level package/module names (without ``.py`` extension).
    """
    top_level = set()
    for entry in path.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            top_level.add(entry.name)
        elif entry.suffix == ".py" and entry.name != "__init__.py":
            top_level.add(entry.stem)
    return top_level


def _rewrite_import_line(
    line: str, top_level: set[str], package_path: str
) -> str | None:
    """Rewrite a single import line if it references a repo-internal module.

    Parameters
    ----------
    line
        A single line of source code (with original indentation and newline).
    top_level
        Set of top-level package names in the repo.
    package_path
        Full dotted package path prefix to prepend.

    Returns
    -------
    str | None
        The rewritten line (preserving indentation), or None if no match.
    """
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    # from <pkg> import ... or from <pkg>.something import ...
    match = re.match(r"from (\w[\w.]*) import ", stripped)
    if match:
        module_path = match.group(1)
        if module_path.split(".")[0] in top_level:
            rest = stripped[match.end() :]
            return f"{indent}from {package_path}.{module_path} import {rest}"

    # import <pkg> or import <pkg>.something
    match = re.match(r"import (\w[\w.]*)", stripped)
    if not match:
        return None
    module_path = match.group(1)
    root = module_path.split(".")[0]
    if root not in top_level:
        return None
    rest = stripped[match.end() :]
    if "." not in module_path:
        # import <pkg> -> from <package_path> import <pkg>
        return f"{indent}from {package_path} import {module_path}{rest}"
    # import <pkg>.something -> from <full_parent> import <leaf>
    parent, leaf = module_path.rsplit(".", 1)
    return f"{indent}from {package_path}.{parent} import {leaf}{rest}"


def _rewrite_absolute_imports(actual_path: Path, canonical_path: Path) -> None:
    """
    Many imports in the repo will be done relative to the external repo root.

    These would work when cloning the repo standalone and running commands
    from the repo root, but since we're not doing that, we need to change these
    import paths to be done relative to the qai_hub_models path.

    For example, an external repo might have a module src/models/yolo.py that it
    tries to import else where using ``from src.models.yolo import ...``

    This would change to something like
    ``from qai_hub_models.models.yolov8_det.external_repos.yolo.src.models.yolo import``

    To fix this, we first identify all top-level directories and python modules in the
    external repo, then grep for all files trying to import from any of these and replace
    with the longer import statement.

    Parameters
    ----------
    actual_path
        Path where the cloned files actually live.
    canonical_path
        The final package path used for deriving the dotted import prefix.
    """
    package_path = _derive_package_path(canonical_path)
    top_level = _get_top_level_packages(actual_path)
    if not top_level:
        return

    for py_file in actual_path.rglob("*.py"):
        if py_file.is_symlink() and not py_file.exists():
            continue
        with open(py_file, errors="surrogateescape") as f:
            lines = f.readlines()
        changed = False
        for i, line in enumerate(lines):
            rewritten = _rewrite_import_line(line, top_level, package_path)
            if rewritten is not None:
                lines[i] = rewritten
                changed = True
        if changed:
            with open(py_file, "w", errors="surrogateescape") as f:
                f.writelines(lines)


def _make_package(path: Path) -> None:
    """Recursively prepare a directory tree for Python import.

    For each directory (skipping hidden ones like ``.git``):
    - Prepends ``# mypy: ignore-errors`` to all ``.py`` files that lack it.
    - Creates an ``__init__.py`` if one doesn't exist.

    Parameters
    ----------
    path
        Root directory to process.
    """
    for entry in path.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            _make_package(entry)
        elif entry.suffix == ".py":
            if entry.is_symlink() and not entry.exists():
                continue
            content = entry.read_text(errors="surrogateescape")
            if not content.startswith(MYPY_IGNORE):
                entry.write_text(MYPY_IGNORE + content, errors="surrogateescape")
    if not (path / "__init__.py").exists():
        (path / "__init__.py").write_text(MYPY_IGNORE)


def postprocess_repo(actual_path: Path, canonical_path: Path) -> None:
    """Walk a cloned repo and make it importable as a Python package.

    1. Creates missing ``__init__.py`` files in every directory.
    2. Prepends ``# mypy: ignore-errors`` to all ``.py`` files.
    3. Rewrites absolute imports of repo-internal modules to use the full
       ``qai_hub_models.models.<model>.external_repos.<repo>.*`` package path.

    Parameters
    ----------
    actual_path
        Path where the cloned files actually live (may be a temp dir during clone).
    canonical_path
        The final package path used for import rewriting, e.g.
        ``<external_repos_dir>/<repo_name>``.
    """
    _make_package(actual_path)
    _rewrite_absolute_imports(actual_path, canonical_path)
    # Handle src layout: if src/ exists with packages inside, also rewrite
    # imports within src/ using src/<pkg> as the canonical sub-path.
    src_dir = actual_path / "src"
    if src_dir.is_dir():
        _rewrite_absolute_imports(src_dir, canonical_path / "src")


def clone_and_patch(
    url: str,
    commit_sha: str,
    target: Path,
    external_repos_path: Path,
    repo_name: str,
    patch_path: Path | None = None,
) -> None:
    """Clone a repo at a pinned commit, apply patches, and postprocess.

    Order of operations:
    1. Shallow clone at pinned SHA
    2. Apply .diff patch (model-specific fixes applied to original source)
    3. Postprocess: add __init__.py files + rewrite imports to fully-qualified paths

    Patches are applied BEFORE import rewriting, so patch diffs reference the
    original repo's import style (e.g. ``from models.X import Y``). The
    postprocess step then rewrites all imports (including patched ones) to use
    the full ``qai_hub_models.models.<id>.external_repos.<repo>.`` prefix.

    Parameters
    ----------
    url
        Git repository URL.
    commit_sha
        Pinned commit SHA.
    target
        Local directory to clone into.
    external_repos_path
        Path to the external_repos directory (used to derive package path).
    repo_name
        Name of the repo subdirectory.
    patch_path
        Path to the ``.diff`` patch file, or None.
    """
    _shallow_clone(url, commit_sha, target)
    if patch_path and patch_path.exists():
        _apply_patch(target, patch_path)
    postprocess_repo(target, external_repos_path / repo_name)


def clone_and_link_repo(
    repo_name: str,
    repo_url: str,
    commit_sha: str,
    content_hash: str,
    external_repos_path: Path,
    patch_path: Path | None = None,
    ask_to_clone: bool = not EXECUTING_IN_CI_ENVIRONMENT,
) -> None:
    """Clone a single repo and make it available locally.

    Operates in one of three modes:

    - **Pip** (any OS): clones into cache; caller uses ``__path__`` redirect.
    - **Dev + symlinks** (unix): clones into cache; symlinks into source tree.
    - **Dev + no symlinks** (Windows): clones directly into source tree.

    Crash safety: clones into a temp directory, then renames atomically.

    Parameters
    ----------
    repo_name
        Name of the repo (used as subdirectory name and display name).
    repo_url
        Git repository URL.
    commit_sha
        Pinned commit SHA.
    content_hash
        Content hash of the repo configuration (used for cache path).
    external_repos_path
        Path to the external_repos directory.
    patch_path
        Path to the ``.diff`` patch file, or None.
    ask_to_clone
        Whether to prompt the user before cloning. Defaults to True
        outside CI.
    """
    # Only use cache for in-tree recipes (under QAIHM_MODELS_ROOT).
    # Standalone recipes clone directly into their external_repos/ folder.
    try:
        external_repos_path.resolve().relative_to(QAIHM_MODELS_ROOT.resolve())
        is_in_tree = True
    except ValueError:
        is_in_tree = False

    use_cache = is_in_tree and (IS_PIP_PACKAGE or USE_SYMLINK_CACHE)
    cache_dir = get_cache_dir(external_repos_path, repo_name, content_hash)
    repo_in_cache = cache_dir / repo_name
    local_dir = external_repos_path / repo_name

    if use_cache:
        if repo_in_cache.is_dir():
            if not IS_PIP_PACKAGE and not local_dir.exists():
                local_dir.symlink_to(repo_in_cache)
            return
    elif local_dir.is_dir():
        return

    if ask_to_clone:
        should_clone = query_yes_no(
            f"{repo_name} requires external repo {repo_url}. Ok to clone?",
            timeout=60,
        )
        if not should_clone:
            raise RuntimeError(
                f"Unable to load without required repository {repo_url}."
            )

    print(f"Cloning {repo_url} ({commit_sha[:8]}) for {repo_name}...")

    if use_cache:
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True, dir=cache_dir.parent
        ) as tmp:
            clone_and_patch(
                repo_url,
                commit_sha,
                Path(tmp) / repo_name,
                external_repos_path,
                repo_name,
                patch_path,
            )
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            Path(tmp).rename(cache_dir)

        if not IS_PIP_PACKAGE and not local_dir.exists():
            local_dir.symlink_to(repo_in_cache)
    else:
        with tempfile.TemporaryDirectory(
            ignore_cleanup_errors=True, dir=external_repos_path
        ) as tmp:
            clone_and_patch(
                repo_url,
                commit_sha,
                Path(tmp) / repo_name,
                external_repos_path,
                repo_name,
                patch_path,
            )
            if local_dir.exists():
                shutil.rmtree(local_dir)
            (Path(tmp) / repo_name).rename(local_dir)

    print("Done")


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def setup_external_repos_impl(
    repo_configs: dict[str, RepoConfig],
    external_repos_path: Path,
) -> dict[str, Path]:
    """Clone all external repos if needed.

    In pip mode, cache directory existence is sufficient validation
    (the content hash is embedded in the path). In dev mode, each repo
    has a ``<repo_name>_hash.txt`` file; hashes are computed live and
    compared. On mismatch or missing, the repo is re-cloned.

    Parameters
    ----------
    repo_configs
        Mapping of repo name to configuration.
    external_repos_path
        Path to the external_repos directory.

    Returns
    -------
    dict[str, Path]
        Mapping of repo name to its resolved directory path on disk.
    """
    per_repo_hashes = compute_repo_hashes(repo_configs, external_repos_path)
    repo_paths: dict[str, Path] = {}
    for repo_name, config in repo_configs.items():
        content_hash = per_repo_hashes[repo_name]

        if not IS_PIP_PACKAGE:
            hash_file = external_repos_path / f"{repo_name}_hash.txt"
            local_dir = external_repos_path / repo_name

            stored_hash = hash_file.read_text().strip() if hash_file.exists() else None
            if stored_hash == content_hash and local_dir.exists():
                repo_paths[repo_name] = local_dir.resolve()
                continue
            if hash_file.exists():
                hash_file.unlink()
            if local_dir.is_symlink():
                local_dir.unlink()
            elif local_dir.is_dir():
                shutil.rmtree(local_dir)

        patch_path = None
        if config.patches_filename:
            patch_path = (external_repos_path / config.patches_filename).resolve()
        clone_and_link_repo(
            repo_name=repo_name,
            repo_url=config.repo_url,
            commit_sha=config.commit_sha,
            content_hash=content_hash,
            external_repos_path=external_repos_path,
            patch_path=patch_path,
        )

        if not IS_PIP_PACKAGE:
            hash_file.write_text(content_hash)
            repo_paths[repo_name] = (external_repos_path / repo_name).resolve()
        else:
            cache_dir = get_cache_dir(external_repos_path, repo_name, content_hash)
            repo_paths[repo_name] = cache_dir / repo_name

    return repo_paths


def get_repo_cache_paths(recipe_dir: Path) -> list[Path]:
    """Return cache directory paths for all external repos.

    Used to set ``__path__`` for pip installs so sub-imports resolve
    from the cache.

    Parameters
    ----------
    recipe_dir
        On-disk path to the recipe folder (contains ``manifest.yaml`` and
        an ``external_repos/`` subfolder).

    Returns
    -------
    list[Path]
        List of cache directory paths.
    """
    repo_configs, external_repos_path = _resolve_external_repos(recipe_dir)
    return [
        get_cache_dir(external_repos_path, repo_name, content_hash)
        for repo_name, content_hash in compute_repo_hashes(
            repo_configs, external_repos_path
        ).items()
    ]


def setup_external_repos(recipe_dir: Path) -> dict[str, Path]:
    """Setup external repos by reading manifest.yaml and cloning if needed.

    Parameters
    ----------
    recipe_dir
        On-disk path to the recipe folder (contains ``manifest.yaml``).

    Returns
    -------
    dict[str, Path]
        Mapping of repo name to its resolved directory path on disk.
    """
    repo_configs, external_repos_path = _resolve_external_repos(recipe_dir)
    if repo_configs:
        return setup_external_repos_impl(repo_configs, external_repos_path)
    return {}


def rewrite_hydra_targets(
    cfg: Any,
    repo_package_path: str,
    top_level_packages: set[str] | None = None,
) -> None:
    """Recursively rewrite Hydra ``_target_`` strings to fully-qualified paths.

    Hydra configs use ``_target_`` keys to specify Python class paths for
    dynamic instantiation (e.g. ``_target_: "models.backbone.ResNet"``).
    After migrating a repo into external_repos, these bare paths no longer
    resolve — they need the full ``qai_hub_models.models.<id>.external_repos.<repo>.``
    prefix prepended.

    Parameters
    ----------
    cfg
        A nested dict/DictConfig (or list) loaded from a Hydra YAML config.
    repo_package_path
        The full dotted package path of the external repo root,
        e.g. ``"qai_hub_models.models.cvt.external_repos.cross_view_transformers"``.
    top_level_packages
        If provided, only rewrite targets whose root module is in this set.
        If None, rewrites any target that doesn't already start with
        ``"qai_hub_models"``.
    """
    prefix = repo_package_path.rstrip(".") + "."
    if hasattr(cfg, "items"):
        for key, val in cfg.items():
            if key == "_target_" and isinstance(val, str):
                if val.startswith("qai_hub_models"):
                    continue
                root = val.split(".")[0]
                if top_level_packages is None or root in top_level_packages:
                    cfg[key] = prefix + val
            else:
                rewrite_hydra_targets(val, repo_package_path, top_level_packages)
    elif isinstance(cfg, (list, tuple)):
        for item in cfg:
            rewrite_hydra_targets(item, repo_package_path, top_level_packages)
