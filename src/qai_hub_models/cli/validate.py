# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""``qai-hub-models validate <target>`` — recipe report card.

Runs every check that can be performed locally against a recipe folder:
folder shape, manifest schema, requirements.txt vs. the base package's
pin set, model imports and torch forward pass, App instantiation, and
URL reachability against every URL declared in the manifest.

No AI Hub workbench calls, no device, no dataset. Exit 0 iff every check
passes; WARNs do not fail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any

import torch
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet

from qai_hub_models import SampleInputsType
from qai_hub_models.cli.install import InstallAborted, install_model
from qai_hub_models.configs._info_yaml_enums import MODEL_STATUS
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, QAIHM_WEB_ASSET
from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.device import CANARY_DEVICES
from qai_hub_models.utils.export.context import (
    import_recipe_module,
    resolve_manifest,
    resolve_model_app_cls,
    resolve_model_cls,
    resolve_recipe_dir,
)
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.metrics import VALID_METRIC_PAIRS
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT, QAIHM_PACKAGE_ROOT
from qai_hub_models.utils.url_check import head_check_urls
from qai_hub_models.utils.validation import perform_runtime_model_validation


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class Result:
    name: str
    category: str
    status: Status
    detail: str = ""


@dataclass
class Report:
    rows: list[Result] = field(default_factory=list)

    def add(self, row: Result) -> None:
        self.rows.append(row)

    @property
    def failed(self) -> bool:
        return any(r.status is Status.FAIL for r in self.rows)

    def counts(self) -> dict[str, int]:
        totals = {s.value: 0 for s in Status}
        for r in self.rows:
            totals[r.status.value] += 1
        return totals


CATEGORIES = (
    "Install",
    "Folder shape",
    "Manifest",
    "Requirements",
    "Model code",
    "App",
    "Tests",
    "Datasets / evaluator",
    "URL reachability",
    "Internal",
)


REQUIRED_FILES = ("__init__.py", "model.py", "manifest.yaml")
OPTIONAL_FILES = {
    "demo.py": "recipe has no demo.py; consumers cannot run qai-hub-models demo <folder>.",
    "test.py": "recipe has no test.py; consumers cannot run pytest <folder>/test.py.",
}
REQUIRED_INIT_EXPORTS = ("MODEL_ID", "Model")
OPTIONAL_INIT_EXPORTS = ("App",)


def _is_in_tree(source_dir: Path) -> bool:
    try:
        source_dir.resolve().relative_to(QAIHM_MODELS_ROOT.resolve())
    except ValueError:
        return False
    return True


# =========================================================================
# Install
# =========================================================================


def _check_install(source_dir: Path, report: Report) -> bool:
    """Run ``qai-hub-models install`` against the recipe. Returns True on success.

    Prints the full plan and prompts ``[y/N]`` before running. Every
    downstream check that imports recipe code (model, App, evaluator) is
    gated on this — if install fails or the user declines, we don't try
    to import.
    """
    try:
        install_model(str(source_dir), assume_yes=False)
    except InstallAborted:
        report.add(
            Result(
                "qai-hub-models install",
                "Install",
                Status.FAIL,
                (
                    "install declined at the [y/N] prompt. Rerun with "
                    "--no-install if the deps are already installed."
                ),
            )
        )
        return False
    except Exception as exc:
        report.add(
            Result(
                "qai-hub-models install",
                "Install",
                Status.FAIL,
                (
                    f"{exc.__class__.__name__}: {exc}. Downstream import-based "
                    "checks (model code, App, evaluator) will be skipped."
                ),
            )
        )
        return False
    report.add(
        Result(
            "qai-hub-models install",
            "Install",
            Status.PASS,
            "install plan accepted and executed",
        )
    )
    return True


def _skip_downstream_on_install_failure(report: Report) -> None:
    """Emit SKIP rows for the categories that depend on a working install."""
    report.add(
        Result(
            "__init__.py imports",
            "Folder shape",
            Status.SKIP,
            "install failed",
        )
    )
    report.add(
        Result(
            "Model class imports",
            "Model code",
            Status.SKIP,
            "install failed",
        )
    )
    report.add(
        Result("App instantiates from Model", "App", Status.SKIP, "install failed")
    )
    report.add(
        Result(
            "Evaluator declared",
            "Datasets / evaluator",
            Status.SKIP,
            "install failed",
        )
    )


# =========================================================================
# Folder shape
# =========================================================================


def _check_files_present(source_dir: Path, report: Report) -> None:
    missing = [f for f in REQUIRED_FILES if not (source_dir / f).exists()]
    if missing:
        report.add(
            Result(
                "Required files present",
                "Folder shape",
                Status.FAIL,
                f"missing: {', '.join(missing)}",
            )
        )
    else:
        report.add(
            Result(
                "Required files present",
                "Folder shape",
                Status.PASS,
                ", ".join(REQUIRED_FILES),
            )
        )

    for filename, warn_message in OPTIONAL_FILES.items():
        if not (source_dir / filename).exists():
            report.add(
                Result(f"{filename} present", "Folder shape", Status.WARN, warn_message)
            )
        else:
            report.add(Result(f"{filename} present", "Folder shape", Status.PASS))


def _check_init_exports(source_dir: Path, report: Report) -> None:
    try:
        module = import_recipe_module(source_dir)
    except Exception as exc:
        report.add(
            Result(
                "__init__.py imports",
                "Folder shape",
                Status.FAIL,
                (
                    f"import failed: {exc.__class__.__name__}: {exc}. Run "
                    "`qai-hub-models install <target>` to install missing deps."
                ),
            )
        )
        return

    missing = [n for n in REQUIRED_INIT_EXPORTS if not hasattr(module, n)]
    if missing:
        report.add(
            Result(
                "__init__.py exports MODEL_ID, Model",
                "Folder shape",
                Status.FAIL,
                f"missing: {', '.join(missing)}",
            )
        )
    else:
        report.add(
            Result("__init__.py exports MODEL_ID, Model", "Folder shape", Status.PASS)
        )

    if not any(hasattr(module, n) for n in OPTIONAL_INIT_EXPORTS):
        report.add(
            Result(
                "__init__.py exports App",
                "Folder shape",
                Status.WARN,
                "no App class declared — demo / inference APIs that expect one won't work.",
            )
        )
    else:
        report.add(Result("__init__.py exports App", "Folder shape", Status.PASS))


def _check_external_repos_init(
    source_dir: Path, manifest: QAIHMModelManifest, report: Report
) -> None:
    if not manifest.external_repos:
        return
    if (source_dir / "external_repos" / "__init__.py").exists():
        report.add(
            Result("external_repos/__init__.py present", "Folder shape", Status.PASS)
        )
        return
    report.add(
        Result(
            "external_repos/__init__.py present",
            "Folder shape",
            Status.FAIL,
            (
                "manifest declares external_repos: but external_repos/__init__.py "
                "is missing. Run `qai-hub-models generate-files <target>`."
            ),
        )
    )


_SELF_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+qai_hub_models\.models\.(\w+)", re.MULTILINE
)


def _check_no_self_referential_imports(source_dir: Path, report: Report) -> None:
    # In-tree recipes legitimately import from `qai_hub_models.models.<id>`
    # because they ARE that package. Only flag for standalone folders.
    if _is_in_tree(source_dir):
        return
    folder_name = source_dir.name
    hits: list[str] = []
    for filename in ("model.py", "app.py", "demo.py", "test.py"):
        path = source_dir / filename
        if not path.exists():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in _SELF_IMPORT_RE.finditer(text):
            if match.group(1) == folder_name:
                lineno = text[: match.start()].count("\n") + 1
                hits.append(f"{filename}:{lineno}")
    if hits:
        report.add(
            Result(
                "No self-referential imports",
                "Folder shape",
                Status.FAIL,
                (
                    f"found `from qai_hub_models.models.{folder_name}` in: "
                    f"{', '.join(hits)}. Use `from .<submodule>` instead — "
                    "the qai_hub_models.models.<id> path only exists in the "
                    "installed package."
                ),
            )
        )
    else:
        report.add(Result("No self-referential imports", "Folder shape", Status.PASS))


# =========================================================================
# Manifest
# =========================================================================


def _check_manifest(source_dir: Path, report: Report) -> QAIHMModelManifest | None:
    try:
        raw_manifest = resolve_manifest(source_dir)
    except Exception as exc:
        report.add(
            Result(
                "manifest.yaml loads",
                "Manifest",
                Status.FAIL,
                f"{exc.__class__.__name__}: {exc}",
            )
        )
        return None

    try:
        # mode="json" so nested mappings keyed by Precision/TargetRuntime dump to
        # strings; ModelDisableReasonsMapping.__init__ takes **kwargs and pydantic
        # splats these keys, which TypeErrors on non-str keys.
        QAIHMModelManifest.model_validate(raw_manifest.model_dump(mode="json"))
        report.add(Result("manifest.yaml validates", "Manifest", Status.PASS))
    except Exception as exc:
        report.add(Result("manifest.yaml validates", "Manifest", Status.FAIL, str(exc)))

    _check_id_matches_folder(source_dir, raw_manifest, report)
    _check_status_not_unset(source_dir, raw_manifest, report)
    _check_external_repo_shas(raw_manifest, report)
    return raw_manifest


def _check_id_matches_folder(
    source_dir: Path, manifest: QAIHMModelManifest, report: Report
) -> None:
    if manifest.id is None:
        return
    if manifest.id != source_dir.name:
        report.add(
            Result(
                "id matches folder name",
                "Manifest",
                Status.FAIL,
                (
                    f"manifest.id={manifest.id!r} but folder is "
                    f"{source_dir.name!r}. The folder name is the import path."
                ),
            )
        )
    else:
        report.add(Result("id matches folder name", "Manifest", Status.PASS))


def _check_status_not_unset(
    source_dir: Path, manifest: QAIHMModelManifest, report: Report
) -> None:
    if not _is_in_tree(source_dir):
        return
    if manifest.status is MODEL_STATUS.UNSET:
        report.add(
            Result(
                "status set (in-tree recipe)",
                "Manifest",
                Status.FAIL,
                (
                    "in-tree recipes must set `status:` to one of published, "
                    "pending, or unpublished."
                ),
            )
        )
    else:
        report.add(
            Result(
                "status set (in-tree recipe)",
                "Manifest",
                Status.PASS,
                manifest.status.value,
            )
        )


def _check_external_repo_shas(manifest: QAIHMModelManifest, report: Report) -> None:
    if not manifest.external_repos:
        return
    bad: list[str] = []
    for name, cfg in manifest.external_repos.items():
        sha = cfg.commit_sha
        if len(sha) != 40 or not all(c in "0123456789abcdef" for c in sha.lower()):
            bad.append(f"{name}={sha!r}")
    if bad:
        report.add(
            Result(
                "external_repos commit_sha is 40-char hex",
                "Manifest",
                Status.FAIL,
                f"not a full SHA: {', '.join(bad)}",
            )
        )
    else:
        report.add(
            Result("external_repos commit_sha is 40-char hex", "Manifest", Status.PASS)
        )


# =========================================================================
# Requirements
# =========================================================================


_GLOBAL_REQUIREMENTS_PATH = QAIHM_PACKAGE_ROOT / "global_requirements.txt"


@cache
def _load_base_package_pins() -> dict[str, SpecifierSet]:
    """Parse the base package's global_requirements.txt into ``{name: SpecifierSet}``."""
    pins: dict[str, SpecifierSet] = {}
    if not _GLOBAL_REQUIREMENTS_PATH.exists():
        return pins
    for line in _GLOBAL_REQUIREMENTS_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            req = Requirement(stripped)
        except InvalidRequirement:
            continue
        pins[req.name.lower()] = req.specifier
    return pins


def _iter_requirements(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, spec_string)`` for every dependency line."""
    out: list[tuple[int, str]] = []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        out.append((i, line))
    return out


def _check_requirements_txt(source_dir: Path, report: Report) -> None:
    req_path = source_dir / "requirements.txt"
    if not req_path.exists():
        return
    base = _load_base_package_pins()
    unpinned: list[str] = []
    conflicts: list[str] = []
    parse_errors: list[str] = []
    for lineno, spec in _iter_requirements(req_path.read_text()):
        try:
            req = Requirement(spec)
        except InvalidRequirement as exc:
            parse_errors.append(f"line {lineno}: {exc}")
            continue
        if not str(req.specifier):
            unpinned.append(f"line {lineno}: `{req.name}` has no version specifier")
            continue
        base_spec = base.get(req.name.lower())
        if base_spec is None:
            continue
        merged = SpecifierSet(f"{base_spec},{req.specifier}")
        if not _specifier_is_satisfiable(merged):
            conflicts.append(
                f"{req.name}: recipe wants {req.specifier}, "
                f"base package wants {base_spec}"
            )

    if parse_errors:
        report.add(
            Result(
                "requirements.txt parses",
                "Requirements",
                Status.FAIL,
                "; ".join(parse_errors),
            )
        )
    else:
        report.add(Result("requirements.txt parses", "Requirements", Status.PASS))

    if unpinned:
        report.add(
            Result(
                "requirements.txt entries pinned",
                "Requirements",
                Status.FAIL,
                "; ".join(unpinned) + ". Add at least a lower bound (e.g. `foo>=1.0`).",
            )
        )
    else:
        report.add(
            Result("requirements.txt entries pinned", "Requirements", Status.PASS)
        )

    if conflicts:
        report.add(
            Result(
                "requirements.txt vs. base package",
                "Requirements",
                Status.FAIL,
                "; ".join(conflicts),
            )
        )
    else:
        report.add(
            Result("requirements.txt vs. base package", "Requirements", Status.PASS)
        )


def _specifier_is_satisfiable(spec: SpecifierSet) -> bool:
    """Best-effort satisfiability check by intersecting bounds.

    Not exhaustive — but catches the common cases: `torch==2.99` intersected
    with `torch>=2.4,<=2.11.0` is unsatisfiable because 2.99 fails the upper
    bound. We probe each equality specifier from *spec* against the rest.
    """
    equalities = [s for s in spec if s.operator == "=="]
    if equalities:
        for eq in equalities:
            for other in spec:
                if other is eq:
                    continue
                if not other.contains(eq.version, prereleases=True):
                    return False
        return True
    return True


def _extract_pip_command_reqs(cmd: str) -> list[Requirement]:
    """Return ``Requirement`` objects installed by a ``pip install ...`` command.

    Skips ``-e`` / ``-r`` / ``--find-links`` / VCS URLs — those can't be
    matched against base pins by name alone.
    """
    tokens = cmd.split()
    if len(tokens) < 3 or tokens[0] != "pip" or tokens[1] != "install":
        return []
    reqs: list[Requirement] = []
    i = 2
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-e", "-r", "--find-links", "--index-url", "--extra-index-url"):
            i += 2
            continue
        if tok.startswith(("-", "git+", "http://", "https://", "file://")):
            i += 1
            continue
        try:
            req = Requirement(tok)
        except InvalidRequirement:
            i += 1
            continue
        reqs.append(req)
        i += 1
    return reqs


def _extract_pip_command_pkgs(cmd: str) -> list[str]:
    """Return package names installed by a ``pip install ...`` command.

    Skips ``-e`` / ``-r`` / ``--find-links`` / VCS URLs — those can't be
    matched against base pins by name alone.
    """
    return [req.name for req in _extract_pip_command_reqs(cmd)]


def _check_pip_commands(manifest: QAIHMModelManifest, report: Report) -> None:
    commands = list(manifest.pre_pip_install_commands) + list(
        manifest.post_pip_install_commands
    )
    if not commands:
        return
    base = _load_base_package_pins()
    conflicts: list[str] = []
    for cmd in commands:
        for req in _extract_pip_command_reqs(cmd.command):
            name = req.name
            base_spec = base.get(name.lower())
            if base_spec is None:
                continue
            try:
                if req.specifier and not _specifier_is_satisfiable(
                    SpecifierSet(f"{base_spec},{req.specifier}")
                ):
                    conflicts.append(
                        f"{name}: manifest command wants {req.specifier}, "
                        f"base package wants {base_spec}"
                    )
            except (InvalidRequirement, Exception):
                continue
    if conflicts:
        report.add(
            Result(
                "manifest pip commands vs. base package",
                "Requirements",
                Status.FAIL,
                "; ".join(conflicts),
            )
        )
    else:
        report.add(
            Result(
                "manifest pip commands vs. base package", "Requirements", Status.PASS
            )
        )


# =========================================================================
# Model code
# =========================================================================


_CUDA_PATTERNS = (
    re.compile(r"\.cuda\s*\("),
    re.compile(r"torch\.cuda\."),
    re.compile(r"\.to\s*\(\s*['\"]cuda"),
)
_SYS_MODULES_PATTERN = re.compile(r"sys\.modules\s*\[")


def _grep_source(path: Path, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    """Return ``"lineno: match"`` for every line matching any pattern."""
    if not path.exists():
        return []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        for pat in patterns:
            if pat.search(line):
                hits.append(f"line {i}: {line.strip()}")
                break
    return hits


def _check_cuda_free(source_dir: Path, report: Report) -> None:
    hits: list[str] = []
    for filename in ("model.py", "app.py"):
        hits.extend(
            f"{filename} {hit}"
            for hit in _grep_source(source_dir / filename, _CUDA_PATTERNS)
        )
    if hits:
        report.add(
            Result(
                "CPU-only model.py / app.py",
                "Model code",
                Status.WARN,
                "; ".join(hits[:3])
                + (f"... ({len(hits)} total)" if len(hits) > 3 else "")
                + ". Verify each occurrence is guarded by "
                "`if torch.cuda.is_available()` or removed.",
            )
        )
    else:
        report.add(Result("CPU-only model.py / app.py", "Model code", Status.PASS))


def _check_no_sys_modules_hacks(source_dir: Path, report: Report) -> None:
    hits = _grep_source(source_dir / "model.py", (_SYS_MODULES_PATTERN,))
    if hits:
        report.add(
            Result(
                "No sys.modules monkey-patches",
                "Model code",
                Status.FAIL,
                "; ".join(hits[:3])
                + ". Use `external_repos:` in manifest.yaml to expose upstream "
                "code — sys.modules shims aren't supported.",
            )
        )
    else:
        report.add(Result("No sys.modules monkey-patches", "Model code", Status.PASS))


def _check_model_code(
    source_dir: Path, report: Report, manifest: QAIHMModelManifest
) -> Any:
    try:
        model_cls = resolve_model_cls(source_dir)
        app_cls = resolve_model_app_cls(source_dir)
    except Exception as exc:
        report.add(
            Result(
                "Model class imports",
                "Model code",
                Status.FAIL,
                f"{exc.__class__.__name__}: {exc}",
            )
        )
        return None

    try:
        model_id = manifest.id if manifest.id else source_dir.name
        perform_runtime_model_validation(model_cls, model_id, app_cls, manifest)
        report.add(
            Result(
                "perform_runtime_model_validation",
                "Model code",
                Status.PASS,
                "I/O names, eval-dataset wiring, mixed-precision litemp",
            )
        )
    except AssertionError as exc:
        report.add(
            Result(
                "perform_runtime_model_validation", "Model code", Status.FAIL, str(exc)
            )
        )
        return None
    except Exception as exc:
        report.add(
            Result(
                "perform_runtime_model_validation",
                "Model code",
                Status.FAIL,
                f"{exc.__class__.__name__}: {exc}",
            )
        )
        return None

    _check_cuda_free(source_dir, report)
    _check_no_sys_modules_hacks(source_dir, report)
    return _check_forward_pass(model_cls, report)


def _flatten_tensors(outputs: Any) -> tuple[Any, ...]:
    """Flatten nested tuples/lists of tensors, as KV-cache outputs are grouped."""
    if isinstance(outputs, torch.Tensor):
        return (outputs,)
    if isinstance(outputs, (tuple, list)):
        return tuple(t for o in outputs for t in _flatten_tensors(o))
    return (outputs,)


def _spec_mismatch(
    component_name: str, spec: InputSpec, sample: SampleInputsType
) -> list[str]:
    """Report any way a component's sample inputs disagree with its input spec."""
    errors = []
    missing = sorted(set(spec) - set(sample))
    extra = sorted(set(sample) - set(spec))
    if missing or extra:
        return [
            f"component '{component_name}' sample inputs do not match spec "
            f"(missing {missing}, unexpected {extra})"
        ]
    for input_name, tensor_spec in spec.items():
        array = sample[input_name][0]
        if tuple(array.shape) != tuple(tensor_spec[0]):
            errors.append(
                f"component '{component_name}' input '{input_name}' sample has shape "
                f"{tuple(array.shape)}, spec says {tuple(tensor_spec[0])}"
            )
        if str(array.dtype) != tensor_spec[1]:
            errors.append(
                f"component '{component_name}' input '{input_name}' sample has dtype "
                f"{array.dtype}, spec says {tensor_spec[1]}"
            )
    return errors


def _check_collection_forward_pass(model: CollectionModel, report: Report) -> None:
    """Run a forward pass through every component of a collection model."""
    name = "from_pretrained + per-component forward pass"
    errors: list[str] = []
    components = getattr(model, "components", None)
    try:
        for component_name in model.component_names:
            spec = model.get_component_input_spec(component_name)
            if components is None or component_name not in components:
                errors.append(f"component '{component_name}' is not reachable")
                continue
            component = components[component_name]
            # The model's own sample inputs, not random tensors from the spec: real
            # values in valid ranges, and the same data the scorecard compares against.
            sample = model.get_component_sample_inputs(
                component_name, use_channel_last_format=False
            )
            mismatch = _spec_mismatch(component_name, spec, sample)
            if mismatch:
                errors.extend(mismatch)
                continue
            inputs = [torch.from_numpy(sample[key][0]) for key in spec]
            with torch.no_grad():
                outputs = component(*inputs)
            out_tuple = _flatten_tensors(outputs)
            output_names = list(model.get_component_output_spec(component_name))
            if len(out_tuple) != len(output_names):
                errors.append(
                    f"component '{component_name}' forward returned "
                    f"{len(out_tuple)} tensors, output spec has {len(output_names)}"
                )
                continue
            for i, out in enumerate(out_tuple):
                if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
                    errors.append(
                        f"component '{component_name}' output {i} "
                        "contains NaN or Inf values"
                    )
    except Exception as exc:
        report.add(
            Result(name, "Model code", Status.FAIL, f"{exc.__class__.__name__}: {exc}")
        )
        return

    if errors:
        report.add(Result(name, "Model code", Status.FAIL, "; ".join(errors)))
        return

    report.add(
        Result(
            name,
            "Model code",
            Status.PASS,
            f"{len(model.component_names)} component(s)",
        )
    )


def _check_multi_graph_sample_inputs(
    model: MultiGraphWorkbenchModel, report: Report
) -> None:
    """Check every graph's sample inputs against its declared input spec."""
    name = "per-graph sample inputs match input spec"
    errors: list[str] = []
    try:
        graph_names = model.graph_names
        for graph_name in graph_names:
            spec = model.get_graph_input_spec(graph_name)
            # Compare in the spec's own layout: the spec is always channel-first,
            # so the default channel-last transpose would fail every such input.
            # Passing spec through also avoids recomputing it inside the getter.
            sample = model.get_graph_sample_inputs(
                graph_name, input_spec=spec, use_channel_last_format=False
            )
            missing = sorted(set(spec) - set(sample))
            extra = sorted(set(sample) - set(spec))
            if missing or extra:
                errors.append(
                    f"graph '{graph_name}' sample inputs do not match spec "
                    f"(missing {missing}, unexpected {extra})"
                )
                continue
            for input_name, tensor_spec in spec.items():
                shape = tuple(sample[input_name][0].shape)
                if shape != tuple(tensor_spec[0]):
                    errors.append(
                        f"graph '{graph_name}' input '{input_name}' has shape "
                        f"{shape}, spec says {tuple(tensor_spec[0])}"
                    )
                dtype = str(sample[input_name][0].dtype)
                if dtype != tensor_spec[1]:
                    errors.append(
                        f"graph '{graph_name}' input '{input_name}' has dtype "
                        f"{dtype}, spec says {tensor_spec[1]}"
                    )
    except Exception as exc:
        report.add(
            Result(name, "Model code", Status.FAIL, f"{exc.__class__.__name__}: {exc}")
        )
        return

    if errors:
        report.add(Result(name, "Model code", Status.FAIL, "; ".join(errors)))
        return

    report.add(
        Result(name, "Model code", Status.PASS, f"{len(graph_names)} graph(s) checked")
    )


def _check_forward_pass(model_cls: Any, report: Report) -> Any:
    try:
        model = model_cls.from_pretrained()
    except Exception as exc:
        report.add(
            Result(
                "from_pretrained + forward pass",
                "Model code",
                Status.FAIL,
                f"from_pretrained: {exc.__class__.__name__}: {exc}",
            )
        )
        return None

    # A CollectionModel's get_input_spec returns a component-name-keyed group, not
    # an InputSpec, so it has to be walked per component rather than called once.
    if isinstance(model, CollectionModel):
        _check_collection_forward_pass(model, report)
        return model

    if isinstance(model, MultiGraphWorkbenchModel):
        _check_multi_graph_sample_inputs(model, report)
        return model

    if not hasattr(model, "get_input_spec"):
        report.add(
            Result(
                "from_pretrained + forward pass",
                "Model code",
                Status.SKIP,
                "model has no get_input_spec",
            )
        )
        return model

    try:
        input_spec = model.get_input_spec()
        inputs = make_torch_inputs(input_spec, seed=42)
        with torch.no_grad():
            outputs = model(*inputs)
    except Exception as exc:
        report.add(
            Result(
                "from_pretrained + forward pass",
                "Model code",
                Status.FAIL,
                f"forward: {exc.__class__.__name__}: {exc}",
            )
        )
        return model

    output_names = list(model.get_output_spec())
    out_tuple = (outputs,) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    if len(out_tuple) != len(output_names):
        report.add(
            Result(
                "from_pretrained + forward pass",
                "Model code",
                Status.FAIL,
                (
                    f"forward returned {len(out_tuple)} tensors, "
                    f"get_output_spec has {len(output_names)}"
                ),
            )
        )
        return model

    report.add(
        Result(
            "from_pretrained + forward pass",
            "Model code",
            Status.PASS,
            f"{len(out_tuple)} output(s)",
        )
    )
    _check_finite_outputs(out_tuple, report)
    _check_output_shapes_match_spec(model, out_tuple, report)
    _check_deterministic_forward(model, inputs, out_tuple, report)
    return model


def _check_finite_outputs(outputs: tuple[Any, ...], report: Report) -> None:
    for i, out in enumerate(outputs):
        if not isinstance(out, torch.Tensor):
            continue
        if not torch.isfinite(out).all():
            report.add(
                Result(
                    "Outputs are finite",
                    "Model code",
                    Status.FAIL,
                    f"output {i} contains NaN or Inf values",
                )
            )
            return
    report.add(Result("Outputs are finite", "Model code", Status.PASS))


def _check_output_shapes_match_spec(
    model: Any, outputs: tuple[Any, ...], report: Report
) -> None:
    spec = model.get_output_spec()
    mismatches: list[str] = []
    for (name, tensor_spec), out in zip(spec.items(), outputs, strict=False):
        if not isinstance(out, torch.Tensor):
            continue
        declared_shape = _extract_shape(tensor_spec)
        if declared_shape is None:
            continue
        if tuple(out.shape) != tuple(declared_shape):
            mismatches.append(
                f"{name}: forward returned {tuple(out.shape)}, spec says {tuple(declared_shape)}"
            )
    if mismatches:
        report.add(
            Result(
                "Output shapes match get_output_spec",
                "Model code",
                Status.FAIL,
                "; ".join(mismatches),
            )
        )
    else:
        report.add(
            Result("Output shapes match get_output_spec", "Model code", Status.PASS)
        )


def _extract_shape(tensor_spec: Any) -> tuple[int, ...] | None:
    """Try to pull the shape tuple out of an OutputSpec entry.

    OutputSpec values may be either a bare ``(shape, dtype)`` tuple (the
    common ``InputSpec``-style form used by many recipes) or a
    ``TensorSpec`` dataclass with a ``.shape`` attribute. Handle both
    without importing the exact class (keeps this loose enough for
    external recipes that stub their own TensorSpec).
    """
    if isinstance(tensor_spec, tuple) and len(tensor_spec) == 2:
        candidate = tensor_spec[0]
        if isinstance(candidate, (list, tuple)):
            declared = tuple(candidate)
            # Empty tuple means "shape not declared" — skip.
            return declared if declared else None
    shape = getattr(tensor_spec, "shape", None)
    if isinstance(shape, (list, tuple)):
        declared = tuple(shape)
        return declared if declared else None
    return None


def _check_deterministic_forward(
    model: Any, inputs: list[torch.Tensor], first: tuple[Any, ...], report: Report
) -> None:
    try:
        with torch.no_grad():
            second_raw = model(*inputs)
    except Exception:
        report.add(
            Result(
                "Deterministic forward pass",
                "Model code",
                Status.WARN,
                "second forward raised; skipping determinism check",
            )
        )
        return
    second = (
        (second_raw,) if isinstance(second_raw, torch.Tensor) else tuple(second_raw)
    )
    for a, b in zip(first, second, strict=False):
        if (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and not torch.equal(a, b)
        ):
            report.add(
                Result(
                    "Deterministic forward pass",
                    "Model code",
                    Status.WARN,
                    "forward is not deterministic under fixed seed — set "
                    "torch.use_deterministic_algorithms or accept the "
                    "on-device numerics may drift.",
                )
            )
            return
    report.add(Result("Deterministic forward pass", "Model code", Status.PASS))


# =========================================================================
# App
# =========================================================================


def _check_app(source_dir: Path, model: Any, report: Report) -> None:
    app_cls = resolve_model_app_cls(source_dir)
    if app_cls is None:
        return
    try:
        try:
            instance = app_cls(model)
        except TypeError:
            components = getattr(model, "components", None)
            if components is None:
                raise
            instance = app_cls(**components)
    except Exception as exc:
        report.add(
            Result(
                "App instantiates from Model",
                "App",
                Status.FAIL,
                f"{exc.__class__.__name__}: {exc}",
            )
        )
        return

    report.add(Result("App instantiates from Model", "App", Status.PASS))

    if any(callable(getattr(instance, name, None)) for name in ("predict", "__call__")):
        report.add(Result("App has predict() or __call__()", "App", Status.PASS))
    else:
        report.add(
            Result(
                "App has predict() or __call__()",
                "App",
                Status.FAIL,
                "App class declares neither predict nor __call__.",
            )
        )


def _check_pytest(source_dir: Path, report: Report) -> None:
    """Run pytest on test.py and report results."""
    test_file = source_dir / "test.py"
    if not test_file.exists():
        return

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            check=False,
            cwd=source_dir.parent,
            env={**os.environ, "PYTHONPATH": str(source_dir.parent)},
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            report.add(Result("pytest", "Tests", Status.PASS, "all tests pass"))
        else:
            # Extract failure summary from pytest output
            output = result.stdout + result.stderr
            lines = output.splitlines()
            # Look for "FAILED" lines or summary line
            failed_tests = [
                line for line in lines if "FAILED" in line or "ERROR" in line
            ]
            if failed_tests:
                detail = "; ".join(failed_tests[:3])  # First 3 failures
            else:
                detail = f"exit code {result.returncode}"
            report.add(Result("pytest", "Tests", Status.FAIL, detail))
    except subprocess.TimeoutExpired:
        report.add(Result("pytest", "Tests", Status.FAIL, "timeout after 300s"))
    except Exception as exc:
        report.add(
            Result("pytest", "Tests", Status.FAIL, f"{exc.__class__.__name__}: {exc}")
        )


# =========================================================================
# Datasets / evaluator
# =========================================================================


def _check_evaluator(model: Any, report: Report) -> None:
    get_dataset_classes = getattr(model, "get_eval_dataset_classes", None)
    if not callable(get_dataset_classes):
        return
    try:
        eval_datasets = get_dataset_classes()
    except Exception as exc:
        report.add(
            Result(
                "Evaluator declared",
                "Datasets / evaluator",
                Status.WARN,
                f"get_eval_dataset_classes raised: {exc}",
            )
        )
        return
    if not eval_datasets:
        report.add(
            Result(
                "Evaluator declared",
                "Datasets / evaluator",
                Status.WARN,
                "no evaluator declared — `qai-hub-models evaluate` will not be "
                "usable. Recipe is still valid for export.",
            )
        )
        return
    report.add(
        Result(
            "Evaluator declared",
            "Datasets / evaluator",
            Status.PASS,
            f"{len(eval_datasets)} eval dataset class(es)",
        )
    )


# =========================================================================
# URL reachability
# =========================================================================


def _collect_manifest_urls(
    manifest: QAIHMModelManifest,
) -> list[tuple[str, str, str]]:
    """Return ``(row_name, url, error_label)`` triples to HEAD-check."""
    urls: list[tuple[str, str, str]] = []
    if manifest.license:
        urls.append(("license", manifest.license, "License URL unreachable"))
    if manifest.source_repo:
        urls.append(("source_repo", manifest.source_repo, "Source repo unreachable"))
    if manifest.research_paper:
        urls.append(
            (
                "research_paper",
                manifest.research_paper,
                "Research paper URL unreachable",
            )
        )
    if manifest.external_repos:
        for name, cfg in manifest.external_repos.items():
            urls.append(
                (
                    f"external_repos.{name}.repo_url",
                    cfg.repo_url,
                    f"external_repos.{name}.repo_url unreachable",
                )
            )
    return urls


def _check_url_reachability(manifest: QAIHMModelManifest, report: Report) -> None:
    triples = _collect_manifest_urls(manifest)
    if not triples:
        report.add(
            Result(
                "URLs reachable",
                "URL reachability",
                Status.SKIP,
                "no URLs declared in manifest.",
            )
        )
        return
    pairs = [(url, label) for _, url, label in triples]
    errors = head_check_urls(pairs)
    error_by_url = {}
    for err in errors:
        # error format: "<label> at <url> (...)"
        for _, url, _label in triples:
            if url in err:
                error_by_url[url] = err
                break
    for name, url, _ in triples:
        if url in error_by_url:
            report.add(Result(name, "URL reachability", Status.FAIL, error_by_url[url]))
        else:
            report.add(Result(name, "URL reachability", Status.PASS, url))


# =========================================================================
# Internal (in-tree, Qualcomm-catalog) checks
# =========================================================================


def _add(report: Report, name: str, status: Status, detail: str = "") -> None:
    report.add(Result(name, "Internal", status, detail))


def _check_in_tree_status(source_dir: Path, report: Report) -> None:
    if _is_in_tree(source_dir):
        _add(report, "recipe lives under qai_hub_models/models/", Status.PASS)
        return
    _add(
        report,
        "recipe lives under qai_hub_models/models/",
        Status.FAIL,
        f"{source_dir} is not under {QAIHM_MODELS_ROOT}. --internal checks "
        "apply only to in-tree recipes.",
    )


def _check_website_fields_set(manifest: QAIHMModelManifest, report: Report) -> None:
    missing = [n for n in ("id", "name", "headline") if getattr(manifest, n) is None]
    if missing:
        _add(
            report,
            "website-facing fields set",
            Status.FAIL,
            f"missing: {', '.join(missing)} (required for published in-tree recipes).",
        )
        return
    _add(report, "website-facing fields set", Status.PASS, "id, name, headline")


def _check_name_style(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.name is None:
        return
    if " " in manifest.name:
        _add(report, "name style", Status.FAIL, "name must not contain spaces.")
        return
    if "_" in manifest.name:
        _add(
            report,
            "name style",
            Status.FAIL,
            "name should use dashes (-), not underscores.",
        )
        return
    _add(report, "name style", Status.PASS, manifest.name)


def _check_headline_period(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.headline is None:
        return
    if not manifest.headline.endswith("."):
        _add(
            report,
            "headline ends with period",
            Status.FAIL,
            f"headline={manifest.headline!r} must end with a period.",
        )
        return
    _add(report, "headline ends with period", Status.PASS)


def _check_related_not_self(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.id is None:
        return
    for r_model in manifest.related_models:
        if r_model == manifest.id:
            _add(
                report,
                "related_models excludes self",
                Status.FAIL,
                f"{manifest.id!r} cannot appear in its own related_models.",
            )
            return
    _add(report, "related_models excludes self", Status.PASS)


def _check_arxiv_abs(manifest: QAIHMModelManifest, report: Report) -> None:
    paper = manifest.research_paper
    if paper is None or not paper.startswith("https://arxiv.org/"):
        return
    if "/abs/" not in paper:
        _add(
            report,
            "arxiv link is /abs/",
            Status.FAIL,
            "arxiv URLs should be /abs/ links, not direct PDFs.",
        )
        return
    _add(report, "arxiv link is /abs/", Status.PASS)


def _check_status_reason(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.status is MODEL_STATUS.UNPUBLISHED and not manifest.status_reason:
        _add(
            report,
            "status_reason coupled with status",
            Status.FAIL,
            "unpublished models must set status_reason with an issue link.",
        )
        return
    if manifest.status is MODEL_STATUS.PUBLISHED and manifest.status_reason:
        _add(
            report,
            "status_reason coupled with status",
            Status.FAIL,
            "published models must not set status_reason.",
        )
        return
    _add(report, "status_reason coupled with status", Status.PASS)


def _check_can_publish(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.status is not MODEL_STATUS.PUBLISHED:
        return
    can_publish, reason = manifest.can_promote_to_published()
    if not can_publish:
        _add(
            report,
            "can be promoted to published",
            Status.FAIL,
            reason or "can_promote_to_published() returned False.",
        )
        return
    _add(report, "can be promoted to published", Status.PASS)


def _check_published_artifacts(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.status is not MODEL_STATUS.PUBLISHED or manifest.id is None:
        return
    pkg_path = manifest.get_package_path()
    problems: list[str] = []
    if not (pkg_path / "manifest.yaml").exists():
        problems.append("manifest.yaml missing")
    if not (pkg_path / "perf.yaml").exists():
        problems.append("perf.yaml missing (published models need one)")
    if not manifest.supports_at_least_1_runtime:
        problems.append("no runtime supported")
    if not manifest.has_static_banner:
        problems.append("has_static_banner is false")
    if problems:
        _add(
            report,
            "published artifacts present",
            Status.FAIL,
            "; ".join(problems),
        )
        return
    _add(report, "published artifacts present", Status.PASS)


def _check_numerics_benchmark(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.numerics_benchmark is None:
        return
    pair = (
        manifest.numerics_benchmark.metric_name,
        manifest.numerics_benchmark.unit,
    )
    if pair not in VALID_METRIC_PAIRS:
        _add(
            report,
            "numerics_benchmark metric pair",
            Status.FAIL,
            f"({pair[0]!r}, {pair[1]!r}) is not a known (metric_name, unit).",
        )
        return
    _add(report, "numerics_benchmark metric pair", Status.PASS)


def _check_default_device_canary(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.default_device not in CANARY_DEVICES:
        _add(
            report,
            "default_device in CANARY_DEVICES",
            Status.FAIL,
            f"default_device={manifest.default_device!r} is not one of CANARY_DEVICES.",
        )
        return
    _add(report, "default_device in CANARY_DEVICES", Status.PASS)


def _check_llm_call_to_action(manifest: QAIHMModelManifest, report: Report) -> None:
    if (
        manifest.id is None
        or not manifest.model_type_llm
        or manifest.llm_details is None
    ):
        return
    cta = manifest.llm_details.call_to_action
    download_ctas = {
        LLM_CALL_TO_ACTION.DOWNLOAD,
        LLM_CALL_TO_ACTION.DOWNLOAD_AND_VIEW_README,
    }
    if cta in download_ctas and manifest.restrict_model_sharing:
        _add(
            report,
            "LLM call_to_action vs restrict_model_sharing",
            Status.FAIL,
            "call_to_action=download is incompatible with restrict_model_sharing=true.",
        )
        return
    release_assets = QAIHM_MODELS_ROOT / manifest.id / "release-assets.yaml"
    if (
        cta not in download_ctas
        and not manifest.restrict_model_sharing
        and release_assets.exists()
    ):
        _add(
            report,
            "LLM call_to_action vs restrict_model_sharing",
            Status.FAIL,
            "LLM has release-assets.yaml but call_to_action is not 'download'.",
        )
        return
    _add(report, "LLM call_to_action vs restrict_model_sharing", Status.PASS)


def _check_qaihm_repo_path(manifest: QAIHMModelManifest, report: Report) -> None:
    if manifest.id is None:
        return
    expected = Path("src") / "qai_hub_models" / "models" / manifest.id
    if expected != ASSET_CONFIG.get_qaihm_repo(manifest.id):
        _add(
            report,
            "QAIHM repo path",
            Status.FAIL,
            f"expected {expected}, got {ASSET_CONFIG.get_qaihm_repo(manifest.id)}.",
        )
        return
    _add(report, "QAIHM repo path", Status.PASS)


def _collect_web_asset_urls(
    manifest: QAIHMModelManifest,
) -> list[tuple[str, str, str]]:
    if manifest.id is None:
        return []
    urls: list[tuple[str, str, str]] = []
    if manifest.has_static_banner:
        urls.append(
            (
                "static banner",
                ASSET_CONFIG.get_web_asset_url(manifest.id, QAIHM_WEB_ASSET.STATIC_IMG),
                "static banner unreachable",
            )
        )
    if manifest.has_animated_banner:
        urls.append(
            (
                "animated banner",
                ASSET_CONFIG.get_web_asset_url(
                    manifest.id, QAIHM_WEB_ASSET.ANIMATED_MOV
                ),
                "animated banner unreachable",
            )
        )
    return urls


def _check_web_asset_urls(manifest: QAIHMModelManifest, report: Report) -> None:
    triples = _collect_web_asset_urls(manifest)
    if not triples:
        return
    pairs = [(url, label) for _, url, label in triples]
    errors = head_check_urls(pairs)
    error_by_url: dict[str, str] = {}
    for err in errors:
        for _name, url, _label in triples:
            if url in err:
                error_by_url[url] = err
                break
    for name, url, _ in triples:
        if url in error_by_url:
            _add(report, name, Status.FAIL, error_by_url[url])
        else:
            _add(report, name, Status.PASS, url)


def _run_internal_checks(
    source_dir: Path, manifest: QAIHMModelManifest | None, report: Report
) -> None:
    _check_in_tree_status(source_dir, report)
    if manifest is None:
        _add(
            report,
            "website-facing fields set",
            Status.SKIP,
            "skipped: manifest failed to load.",
        )
        return
    _check_website_fields_set(manifest, report)
    _check_name_style(manifest, report)
    _check_headline_period(manifest, report)
    _check_related_not_self(manifest, report)
    _check_arxiv_abs(manifest, report)
    _check_status_reason(manifest, report)
    _check_can_publish(manifest, report)
    _check_published_artifacts(manifest, report)
    _check_numerics_benchmark(manifest, report)
    _check_default_device_canary(manifest, report)
    _check_llm_call_to_action(manifest, report)
    _check_qaihm_repo_path(manifest, report)
    _check_web_asset_urls(manifest, report)


# =========================================================================
# Orchestration
# =========================================================================


def _run_all_checks(
    source_dir: Path,
    report: Report,
    skip_install: bool = False,
) -> QAIHMModelManifest | None:
    if skip_install:
        report.add(
            Result(
                "qai-hub-models install",
                "Install",
                Status.SKIP,
                "skipped via --no-install",
            )
        )
        install_ok = True
    else:
        install_ok = _check_install(source_dir, report)

    _check_files_present(source_dir, report)
    _check_no_self_referential_imports(source_dir, report)
    manifest = _check_manifest(source_dir, report)
    if manifest is not None:
        _check_external_repos_init(source_dir, manifest, report)
        _check_requirements_txt(source_dir, report)
        _check_pip_commands(manifest, report)

    if not install_ok:
        _skip_downstream_on_install_failure(report)
        if manifest is not None:
            _check_url_reachability(manifest, report)
        return manifest

    _check_init_exports(source_dir, report)
    model = None
    if manifest is not None:
        model = _check_model_code(source_dir, report, manifest)
    if model is not None:
        _check_app(source_dir, model, report)
        _check_evaluator(model, report)
    _check_pytest(source_dir, report)
    if manifest is not None:
        _check_url_reachability(manifest, report)
    return manifest


def _render_text(report: Report) -> str:
    lines: list[str] = []
    by_category: dict[str, list[Result]] = {}
    for row in report.rows:
        by_category.setdefault(row.category, []).append(row)
    for category in CATEGORIES:
        rows = by_category.get(category)
        if not rows:
            continue
        lines.append(category)
        for row in rows:
            head = f"  {row.status.value:<4}  {row.name}"
            if row.detail:
                head += f" - {row.detail}"
            lines.append(head)
        lines.append("")
    counts = report.counts()
    lines.append(
        f"{counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['WARN']} warnings, {counts['SKIP']} skipped."
    )
    return "\n".join(lines)


def _render_json(report: Report) -> str:
    payload = {
        "rows": [
            {
                "name": r.name,
                "category": r.category,
                "status": r.status.value,
                "detail": r.detail,
            }
            for r in report.rows
        ],
        "counts": report.counts(),
        "failed": report.failed,
    }
    return json.dumps(payload, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qai-hub-models validate",
        description=(
            "Run the local recipe report card: folder shape, manifest, "
            "requirements.txt vs. base package, model imports, torch forward "
            "pass, App instantiation, URL reachability. No workbench calls. "
            "Exit 0 iff every check passes (WARNs do not fail)."
        ),
    )
    parser.add_argument(
        "target",
        help=(
            "Recipe folder path (my_model) or an installed model id "
            "(e.g. mobilenet_v2)."
        ),
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help=(
            "Skip the install step. Use when the recipe's dependencies are "
            "already installed in the current environment."
        ),
    )
    parser.add_argument(
        "--internal",
        action="store_true",
        help=(
            "Also run the Qualcomm-catalog (in-tree) checks: website-facing "
            "manifest fields (name/headline style, CANARY_DEVICES, arxiv "
            "/abs/, status_reason coupling, PUBLISHED artifact requirements, "
            "numerics_benchmark metric pair, LLM call-to-action rules, "
            "QAIHM repo path) plus static/animated banner URL HEAD checks. "
            "Only meaningful for recipes under src/qai_hub_models/models/."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit the report as JSON on stdout instead of the human report card.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        source_dir = resolve_recipe_dir(args.target)
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    report = Report()
    manifest = _run_all_checks(source_dir, report, skip_install=args.no_install)
    if args.internal:
        _run_internal_checks(source_dir, manifest, report)

    output = _render_json(report) if args.as_json else _render_text(report)
    print(output)
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
