# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Heavy-side ``qai-hub-models install <model_id>`` implementation.

Walks a model's dependency graph (datasets -> templates -> other models)
via DFS, installing each node's ``requirements.txt`` and any manifest
``pre_pip_install_commands`` / ``post_pip_install_commands`` exactly
once. Traversal is post-order: a node's dependencies install before the
node itself, so pip resolves in the order the manifest declares.

Dependencies are declared in each folder's ``manifest.yaml``:

  - Model manifests (``models/<id>/manifest.yaml``) may declare
    ``datasets``, ``templates``, ``models`` (cross-model), plus the pre/post
    pip command lists.
  - Template manifests (``models/templates/<name>/manifest.yaml``)
    may declare ``datasets`` and ``templates`` transitively.
  - Dataset manifests (``datasets/<name>/manifest.yaml``) may declare
    ``datasets``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path

from qai_hub_models_cli.errors import CLIUserError

from qai_hub_models.configs.manifest_yaml import PipCommand, QAIHMModelManifest
from qai_hub_models.utils.export.context import (
    looks_like_path,
    unknown_model_error,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_PACKAGE_ROOT

DATASETS_ROOT = QAIHM_PACKAGE_ROOT / "datasets"
TEMPLATES_ROOT = QAIHM_PACKAGE_ROOT / "models" / "templates"
MODELS_ROOT = QAIHM_PACKAGE_ROOT / "models"

_NVIDIA_SMI_TIMEOUT_SECONDS = 5


class NodeKind(Enum):
    DATASET = "dataset"
    TEMPLATE = "template"
    MODEL = "model"


@dataclass(frozen=True)
class Node:
    """A single installable unit in the dependency graph.

    Datasets and templates always resolve inside the installed
    ``qai_hub_models`` package (they're shared library code). Models
    normally resolve to ``MODELS_ROOT / name`` — but the DFS root can be
    an arbitrary folder outside the package, passed via
    ``folder_override``. That lets ``qai-hub-models install my_model``
    walk a standalone recipe's dependency graph.
    """

    kind: NodeKind
    name: str
    folder_override: Path | None = None

    @property
    def folder(self) -> Path:
        if self.folder_override is not None:
            return self.folder_override
        if self.kind is NodeKind.DATASET:
            return DATASETS_ROOT / self.name
        if self.kind is NodeKind.TEMPLATE:
            return TEMPLATES_ROOT / self.name
        return MODELS_ROOT / self.name

    @property
    def manifest_path(self) -> Path:
        return self.folder / "manifest.yaml"

    @property
    def requirements_path(self) -> Path:
        return self.folder / "requirements.txt"


def _has_cuda_gpu() -> bool:
    """Return True if ``nvidia-smi`` runs successfully.

    Proxy for CUDA GPU availability, matching the check used by the
    internal ``scripts/tasks/venv.py`` install path. A short timeout guards
    against a wedged GPU driver hanging the CLI indefinitely.
    """
    try:
        return (
            subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@cache
def uv_installed() -> bool:
    try:
        return (
            subprocess.run(["which", "uv"], check=False, capture_output=True).returncode
            == 0
        )
    except Exception:
        return False


@cache
def get_pip() -> str:
    return "uv pip" if uv_installed() else "pip"


def _rewrite_pip_tokens(tokens: list[str]) -> list[str]:
    assert tokens and tokens[0] == "pip", tokens
    if uv_installed():
        tokens = [t for t in tokens if t != "--use-pep517"]
        if len(tokens) >= 2 and tokens[1] == "uninstall":
            tokens = [t for t in tokens if t not in ("-y", "--yes")]
        return tokens
    return [t for t in tokens if t != "--no-build-isolation"]


def normalize_pip_command(raw: str) -> str:
    """Rewrite a manifest ``pip …`` string for the resolved pip binary."""
    tokens = _rewrite_pip_tokens(raw.split())
    return " ".join([get_pip(), *tokens[1:]])


def normalize_pip_argv(argv: list[str]) -> list[str]:
    """Rewrite an argv list starting with ``pip`` for the resolved pip binary."""
    tokens = _rewrite_pip_tokens(list(argv))
    return [*get_pip().split(), *tokens[1:]]


def _validate_dep_name(name: str, kind: NodeKind, referrer: Node) -> None:
    r"""Reject dep names that could escape their intended root.

    ``Node.folder`` builds ``<ROOT> / self.name`` directly from strings read
    out of a manifest, so a name containing ``/``, ``\``, or ``..`` would
    resolve outside its expected root. Since ``manifest.yaml`` and
    ``requirements.txt`` files at that escaped location would then be loaded
    and run, every dep name has to be a plain folder name.
    """
    if not name or "/" in name or "\\" in name or name in {"..", "."}:
        raise CLIUserError(
            f"{referrer.kind.value} {referrer.name!r} declares an invalid "
            f"{kind.value} dependency name {name!r}. Dep names must be plain "
            "folder names (no path separators, no '..')."
        )


def _load_deps(node: Node) -> tuple[list[str], list[str], list[str]]:
    """Return ``(datasets, templates, models)`` from ``node``'s manifest.

    Missing manifest yields empty lists (used by nodes that have a
    ``requirements.txt`` but no ``manifest.yaml``, e.g. datasets without
    transitive deps). Missing keys yield empty lists. Only model manifests
    may declare cross-model deps; other node kinds always return
    ``models=[]`` regardless of what their manifest says. Every declared
    name is validated so it cannot escape its intended root.
    """
    manifest = QAIHMModelManifest.from_yaml(
        node.manifest_path, create_empty_if_no_file=True
    )
    datasets = list(manifest.datasets)
    templates = list(manifest.templates)
    models = list(manifest.models) if node.kind is NodeKind.MODEL else []
    for name in datasets:
        _validate_dep_name(name, NodeKind.DATASET, node)
    for name in templates:
        _validate_dep_name(name, NodeKind.TEMPLATE, node)
    for name in models:
        _validate_dep_name(name, NodeKind.MODEL, node)
    return datasets, templates, models


def _load_pip_commands(node: Node, field: str) -> list[PipCommand]:
    """Return ``PipCommand`` entries from ``field`` on ``node``'s manifest.

    Only model manifests may declare pip commands; other node kinds return
    an empty list (their manifests don't carry these fields).
    """
    if node.kind is not NodeKind.MODEL:
        return []
    manifest = QAIHMModelManifest.from_yaml(
        node.manifest_path, create_empty_if_no_file=True
    )
    commands: list[PipCommand] = getattr(manifest, field)
    return commands


def build_install_order(root: Node) -> list[Node]:
    """DFS the dependency graph and return nodes in install order.

    Post-order traversal: each node's dependencies appear before the node
    itself. Nodes already seen are skipped so a shared dep is emitted
    exactly once. Cycles are broken by the visited set (a node in progress
    is treated as already emitted from the perspective of a back-edge).
    """
    order: list[Node] = []
    visited: set[Node] = set()
    in_progress: set[Node] = set()

    def _visit(node: Node) -> None:
        if node in visited or node in in_progress:
            return
        if node is not root and not node.folder.exists():
            raise CLIUserError(
                f"{node.kind.value} {node.name!r} is declared as a dependency "
                f"but its folder does not exist at {node.folder}. Check for a "
                "typo in the referencing manifest.yaml."
            )
        in_progress.add(node)
        datasets, templates, models = _load_deps(node)
        for name in datasets:
            _visit(Node(NodeKind.DATASET, name))
        for name in templates:
            _visit(Node(NodeKind.TEMPLATE, name))
        for name in models:
            _visit(Node(NodeKind.MODEL, name))
        in_progress.discard(node)
        visited.add(node)
        order.append(node)

    _visit(root)
    return order


def _filter_pip_commands(
    commands: Iterable[PipCommand], on_gpu: bool
) -> list[PipCommand]:
    """Drop commands whose ``machine`` doesn't apply to the current host."""
    kept: list[PipCommand] = []
    for cmd in commands:
        if cmd.machine == "gpu" and not on_gpu:
            continue
        if cmd.machine == "cpu" and on_gpu:
            continue
        kept.append(cmd)
    return kept


def _node_install_commands(node: Node, on_gpu: bool) -> list[list[str]]:
    """Return the argv list for each command needed to install ``node``.

    Order per node:
      1. every ``pre_pip_install_commands`` entry that applies to this host
      2. ``pip install -r requirements.txt`` (only if the file exists)
      3. every ``post_pip_install_commands`` entry that applies to this host
    """
    pre = _filter_pip_commands(
        _load_pip_commands(node, "pre_pip_install_commands"), on_gpu
    )
    post = _filter_pip_commands(
        _load_pip_commands(node, "post_pip_install_commands"), on_gpu
    )

    commands: list[list[str]] = [
        normalize_pip_argv(shlex.split(c.command)) for c in pre
    ]
    if node.requirements_path.exists():
        commands.append(
            [*get_pip().split(), "install", "-r", str(node.requirements_path)]
        )
    commands.extend(normalize_pip_argv(shlex.split(c.command)) for c in post)
    return commands


def _folder_root(folder: Path) -> Node:
    """Build the root Node for a recipe folder, validating it holds a manifest."""
    if not (folder / "manifest.yaml").exists():
        raise CLIUserError(f"{folder} has no manifest.yaml — nothing to install.")
    return Node(NodeKind.MODEL, folder.name, folder_override=folder)


def _resolve_root(target: str) -> Node:
    """Build the root Node for ``qai-hub-models install <target>``.

    Accepts a folder path (external recipe not necessarily under the
    installed package), a bare model id (resolved to ``MODELS_ROOT /
    <id>``), or a bare folder name matching a directory in the current
    working directory. Installed model ids win over cwd folders of the
    same name, matching :func:`resolve_recipe_dir`. Display names are not
    accepted.
    """
    if looks_like_path(target):
        folder = Path(target).resolve()
        if not folder.is_dir():
            raise CLIUserError(
                f"{target!r} is not a directory. Point at a recipe folder "
                "that contains manifest.yaml."
            )
        return _folder_root(folder)

    if target not in MODEL_IDS:
        if (cwd_folder := Path(target)).is_dir():
            return _folder_root(cwd_folder.resolve())
        raise CLIUserError(unknown_model_error(target))
    return Node(NodeKind.MODEL, target)


def plan_install(target: str) -> list[tuple[Node, list[list[str]]]]:
    """Return the full install plan for ``target``.

    Each entry pairs a graph node with the argv list of every command
    needed to install it, in the order they should run.

    Parameters
    ----------
    target
        A recipe folder path (``my_model``), a bare model id
        (``yolov8_det``), or a bare folder name in the current directory.
        Templates and datasets always resolve inside the installed package.

    Returns
    -------
    list[tuple[Node, list[list[str]]]]
        Install plan as (node, argv-lists) pairs in the order they should run.
    """
    root = _resolve_root(target)
    on_gpu = _has_cuda_gpu()
    return [
        (node, _node_install_commands(node, on_gpu))
        for node in build_install_order(root)
    ]


def _run(argv: list[str], dry_run: bool) -> None:
    """Print ``argv`` and (unless ``dry_run``) invoke it via subprocess."""
    print("+", shlex.join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


class InstallAborted(RuntimeError):
    """Raised when the user declines the install prompt."""


def _print_plan(plan: list[tuple[Node, list[list[str]]]]) -> None:
    """Print the full install plan (every pip command, grouped per node)."""
    for node, commands in plan:
        if not commands:
            print(f"# {node.kind.value} {node.name}: nothing to install", flush=True)
            continue
        print(f"# {node.kind.value} {node.name}", flush=True)
        for argv in commands:
            print("  " + shlex.join(argv), flush=True)


def install_model(target: str, dry_run: bool = False, assume_yes: bool = False) -> None:
    """Install ``target`` and every declared dependency in DFS order.

    Prints the full plan first, then prompts ``[y/N]`` before running any
    commands. Pass ``assume_yes=True`` (or ``--yes`` on the CLI) to skip
    the prompt for non-interactive use.

    Parameters
    ----------
    target
        A recipe folder path (``my_model``), a bare model id
        (``yolov8_det``), or a bare folder name in the current directory.
        Templates and datasets always resolve inside the installed package.
    dry_run
        If True, print each command that would run instead of executing it.
        Useful for previewing the plan.
    assume_yes
        If True, skip the interactive ``[y/N]`` confirmation and proceed.
        ``qai-hub-models validate`` intentionally leaves this ``False`` so
        Install remains the first interactive gate in the report card
        whenever there is something to install.
    """
    plan = plan_install(target)
    print(f"Install plan for {target}:", flush=True)
    _print_plan(plan)

    if not any(commands for _, commands in plan):
        print("Nothing to install.", flush=True)
        return

    if dry_run:
        return

    if not assume_yes:
        try:
            reply = input("Proceed with install? [y/N]: ").strip().lower()
        except EOFError:
            reply = ""
        if reply not in ("y", "yes"):
            raise InstallAborted("install declined by user at the [y/N] prompt")

    for _, commands in plan:
        for argv in commands:
            _run(argv, dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qai-hub-models install",
        description="Install the dependencies for a model, "
        "including its declared datasets, templates, and cross-model deps.",
    )
    parser.add_argument(
        "target",
        help=(
            "Recipe folder path (e.g. my_model), an installed model id "
            "(e.g. yolov8_det), or a folder name in the current directory. "
            "The folder must contain a manifest.yaml."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install plan without running any pip commands.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the [y/N] confirmation prompt and run the plan immediately.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point called from the lean-CLI dispatcher.

    Kept intentionally tiny — the lean CLI resolves the model id and
    forwards the remaining args here. Options mirror ``pip install`` where
    they overlap.
    """
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    try:
        install_model(args.target, dry_run=args.dry_run, assume_yes=args.yes)
    except InstallAborted as exc:
        print(f"Aborted: {exc}", flush=True)


if __name__ == "__main__":
    main()
