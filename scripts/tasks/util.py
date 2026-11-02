# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import contextlib
import functools
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .constants import (
    PY_PACKAGE_MODELS_ROOT,
    SCORECARD_PACKAGE_MODELS_ROOT,
    bash_argv,
    process_output,
    run_and_get_output,
    venv_activate_command,
)


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    YELLOW = "\033[0;33m"
    OFF = "\033[0m"


@contextlib.contextmanager
def new_cd(x: str | os.PathLike[str]) -> contextlib.AbstractContextManager[None]:
    d = os.getcwd()

    # This could raise an exception, but it's probably
    # best to let it propagate and let the caller
    # deal with it, since they requested x
    os.chdir(x)

    try:
        yield

    finally:
        # This could also raise an exception, but you *really*
        # aren't equipped to figure out what went wrong if the
        # old working directory can't be restored.
        os.chdir(d)


@functools.cache
def check_manifest_field(model_name: str, field_name: str) -> bool:
    """
    This process does not have the yaml package, so use this primitive way
    to check if a manifest field is true and apply branching logic within CI/scorecard.
    """
    yaml_path = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "manifest.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            if f"{field_name}: true" in f.read():
                return True
    return False


@functools.cache
def check_scorecard_config_field(model_name: str, field_name: str) -> bool:
    """
    This process does not have the yaml package, so use this primitive way
    to check if an internal-config field is true and apply branching logic within CI/scorecard.
    """
    yaml_path = (
        Path(SCORECARD_PACKAGE_MODELS_ROOT) / model_name / "internal-config.yaml"
    )
    # Static bench models have no config. For a recipe, returning False would
    # silently *enable* scorecard when its author set skip_scorecard, so refuse.
    if not yaml_path.exists():
        if not (Path(PY_PACKAGE_MODELS_ROOT) / model_name / "manifest.yaml").exists():
            return False
        raise FileNotFoundError(
            f"{model_name} has no {yaml_path}; cannot tell whether {field_name} is set."
        )
    with open(yaml_path) as f:
        return f"{field_name}: true" in f.read()


@functools.cache
def get_manifest_str_field(model_name: str, field_name: str) -> str | None:
    """This process does not have the yaml package, so use this primitive way to get manifest field value."""
    yaml_path = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "manifest.yaml"
    if not yaml_path.exists():
        return None
    prefix = f"{field_name}:"
    with open(yaml_path) as f:
        lines = f.readlines()
    parts: list[str] = []
    for i, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        parts.append(line[len(prefix) :].rstrip("\n").strip())
        for cont in lines[i + 1 :]:
            if cont[:1] in (" ", "\t"):
                parts.append(cont.strip())
            else:
                break
        break
    if not parts:
        return None
    field = " ".join(p for p in parts if p)
    if not field:
        return None
    if (field[0] == '"' and field[-1] == '"') or (field[0] == "'" and field[-1] == "'"):
        field = field[1:-1]
    return field


@functools.cache
def get_pip_install_commands(
    model_name: str, field_name: str
) -> tuple[tuple[str, str], ...]:
    """Read a top-level list of PipCommand entries out of manifest.yaml.

    Returns a tuple of ``(command, machine)`` pairs. ``machine`` defaults to
    ``"any"`` when unset. Returns an empty tuple if the field is absent.

    Written primitively (no yaml dep) to match the other manifest helpers in
    this module. Understands both shapes that manifests emit:

        <field_name>:
        - pip install foo             # bare string, machine="any"
        - command: pip install bar    # mapping form, needed to set machine
          machine: gpu

    Values may be plain, single-quoted, or double-quoted. Nothing more.
    """
    if field_name not in {"pre_pip_install_commands", "post_pip_install_commands"}:
        raise ValueError(f"Unsupported pip-commands field: {field_name}")

    yaml_path = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "manifest.yaml"
    if not yaml_path.exists():
        return ()

    with open(yaml_path) as f:
        lines = f.readlines()

    entries: list[tuple[str, str]] = []
    in_section = False
    current_command: str | None = None
    current_machine: str = "any"

    def _flush() -> None:
        nonlocal current_command, current_machine
        if current_command is not None:
            entries.append((current_command, current_machine))
        current_command = None
        current_machine = "any"

    def _unquote(v: str) -> str:
        v = v.strip()
        if len(v) >= 2 and (
            (v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")
        ):
            return v[1:-1]
        return v

    prefix = f"{field_name}:"
    for line in lines:
        stripped_full = line.rstrip("\n")
        if not in_section:
            if stripped_full == prefix or stripped_full.startswith(prefix + " "):
                in_section = True
            continue

        stripped = stripped_full.strip()
        if not stripped:
            continue

        # A new list item is either "- command: ..." (mapping form),
        # "- pip install ..." (bare string), or a continuation "machine: ..."
        # (indented). Anything else at column 0 ends the section.
        if stripped_full.startswith("- "):
            _flush()
            rest = stripped[2:].strip()
            if rest.startswith("command:"):
                current_command = _unquote(rest[len("command:") :].strip())
            else:
                current_command = _unquote(rest)
        elif line[0].isspace() and stripped.startswith("machine:"):
            current_machine = _unquote(stripped[len("machine:") :].strip())
        elif line[0].isspace() and stripped.startswith("command:"):
            # Unusual layout: "-" on its own line, "command:" on the next.
            _flush()
            current_command = _unquote(stripped[len("command:") :].strip())
        else:
            # A non-list, column-0 line means we've walked out of the section.
            _flush()
            break

    _flush()
    return tuple(entries)


def is_quantized_llm_model(model_name: str) -> bool:
    quantize_script = Path(PY_PACKAGE_MODELS_ROOT) / model_name / "quantize.py"
    return (
        check_manifest_field(model_name, "model_type_llm") and quantize_script.exists()
    )


def can_support_aimet(platform: str = sys.platform) -> bool:
    # aimet-onnx >= 2.28 ships cp310-abi3 wheels, so any python >= 3.10 is installable.
    return platform in {"linux", "linux2"} and sys.version_info >= (3, 10)


def get_is_hub_quantized(model_name: str) -> bool:
    return not check_manifest_field(
        model_name, "is_precompiled"
    ) and not check_manifest_field(model_name, "is_aimet")


def get_requires_aot_prepare(model_name: str) -> bool:
    return check_manifest_field(model_name, "requires_aot_prepare")


def model_needs_aimet(model_name: str) -> bool:
    return check_manifest_field(model_name, "is_aimet")


def get_model_python_version_requirements(
    model_name: str,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    # Returns minimum required version,
    # and "less than" required version (eg. python version must be less than the provided version)
    # None == No version set (any version OK)
    manifest = os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "manifest.yaml")
    req_less_than, min_version = None, None
    if os.path.exists(manifest):
        with open(manifest) as f:
            manifest_data = f.read()
        req_less_than = re.search(
            r'python_version_less_than:\s*["\']([\d.]+)["\']', manifest_data
        )
        if req_less_than:
            spl = req_less_than.group(1).split(".")
            req_less_than = int(spl[0]), int(spl[1])

        min_version = re.search(
            r'python_version_greater_than_or_equal_to:\s*["\']([\d.]+)["\']',
            manifest_data,
        )
        if min_version:
            spl = min_version.group(1).split(".")
            min_version = int(spl[0]), int(spl[1])

    return (min_version, req_less_than)


def default_parallelism() -> int:
    """A conservative number of processes across which to spread pytests desiring parallelism."""
    from .github import on_github  # avoid circular import

    cpu_count = os.cpu_count()
    if not cpu_count:
        return 1

    # In CI, saturate the machine
    if on_github():
        return cpu_count

    # When running locally, leave a little CPU for other uses
    return max(1, int(cpu_count - 2))


# Convenience function for printing to stdout without buffering.
def echo(value: object, **args: Any) -> None:
    print(value, flush=True, **args)


def have_root() -> bool:
    return os.geteuid() == 0


def on_linux() -> bool:
    return platform.uname().system == "Linux"


def on_mac() -> bool:
    return platform.uname().system == "Darwin"


def run(command: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(bash_argv(command), check=True)


def run_with_venv(
    venv: str | None, command: str, env: dict[str, str] | None = None
) -> None:
    if venv is not None:
        subprocess.run(
            bash_argv(f"{venv_activate_command(venv)} && {command}"),
            check=True,
            env=env,
        )
    else:
        run(command)


def run_with_venv_and_get_output(venv: str | None, command: str) -> str:
    if venv is not None:
        return process_output(
            subprocess.run(
                bash_argv(f"{venv_activate_command(venv)} && {command}"),
                stdout=subprocess.PIPE,
                check=True,
            )
        )
    return run_and_get_output(command)


def str_to_bool(word: str) -> bool:
    return word.lower() in ["1", "true", "yes"]


def get_env_bool(key: str, default: bool | None = None) -> bool | None:
    val = os.environ.get(key, None)
    if val is None:
        return None
    return str_to_bool(val)


def on_ci() -> bool:
    return get_env_bool("QAIHM_CI") or False


def debug_mode() -> bool:
    return get_env_bool("DEBUG_MODE") or False


@functools.cache
def uv_installed() -> bool:
    try:
        result = subprocess.run(bash_argv("which uv"), check=False, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False


@functools.cache
def get_pip() -> str:
    if uv_installed():
        return "uv pip"
    return "pip"


@functools.cache
def has_cuda_gpu() -> bool:
    """Return True if nvidia-smi exits with code 0, used as a proxy for CUDA GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False
