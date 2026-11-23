# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Per-model Python environments for run_scorecard.py.

Models compatible with the global requirements run in the current
environment. Every other model gets a venv under a shared directory, created
once with `uv venv`, given qai_hub_models and its CLI from wheels (built once
and cached, so uv reuses the build across envs), and then
`qai-hub-models install <model>`.
"""

from __future__ import annotations

import fcntl
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import qai_hub_models
from qai_hub_models.cli.install import _has_cuda_gpu
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig

_READY_MARKER = ".scorecard-env-ready"
_TORCH_PACKAGES = ("torch", "torchvision", "torchcodec")
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def default_envs_dir() -> Path:
    root = os.environ.get("QAIHM_STORE_ROOT") or Path.home()
    return Path(root) / ".qaihm" / "scorecard-envs"


def needs_own_env(model_id: str) -> bool:
    return QAIHMModelInternalConfig.from_model(
        model_id
    ).global_requirements_incompatible


def _repo_root() -> Path:
    """The checkout holding src/ and cli/; the package may be installed elsewhere."""
    candidates = [Path(qai_hub_models.__file__).resolve().parents[2], Path.cwd()]
    for candidate in candidates:
        if (candidate / "src" / "pyproject.toml").exists() and (
            candidate / "cli" / "pyproject.toml"
        ).exists():
            return candidate
    raise RuntimeError(
        "Could not find the repo checkout (src/ and cli/) to build wheels from; "
        "run run_scorecard.py from the repo root."
    )


def _run(argv: list[str | Path], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(str(a) for a in argv), flush=True)
    start = time.perf_counter()
    try:
        subprocess.run([str(a) for a in argv], check=True, env=env)
    finally:
        print(f"  ({time.perf_counter() - start:.1f}s)", flush=True)


def _cpu_torch_specs() -> list[str]:
    """The global requirements' torch pins, e.g. ``torch>=2.9,<=2.14.1``."""
    global_reqs = Path(qai_hub_models.__file__).parent / "global_requirements.txt"
    specs = []
    for raw in global_reqs.read_text().splitlines():
        line = raw.split("#")[0].strip()
        if re.match(rf"^({'|'.join(_TORCH_PACKAGES)})[<>=!~]", line):
            specs.append(line)
    return specs


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialize env / wheel creation across concurrent run_scorecard processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _build_wheels(envs_dir: Path) -> Path:
    """Build the qai_hub_models and CLI wheels once; later envs install from them."""
    wheel_dir = envs_dir / "wheels"
    with _locked(envs_dir / "wheels.lock"):
        if not (
            list(wheel_dir.glob("qai_hub_models-*.whl"))
            and list(wheel_dir.glob("qai_hub_models_cli-*.whl"))
        ):
            repo = _repo_root()
            _run(["uv", "build", "--wheel", "--out-dir", wheel_dir, repo / "src"])
            _run(["uv", "build", "--wheel", "--out-dir", wheel_dir, repo / "cli"])
    return wheel_dir


def ensure_model_env(model_id: str, envs_dir: Path) -> Path:
    """Create (if missing) and return the python executable of this model's venv."""
    venv = envs_dir / model_id
    python = venv / "bin" / "python"
    start = time.perf_counter()
    with _locked(envs_dir / f"{model_id}.lock"):
        if (venv / _READY_MARKER).exists():
            return python

        wheel_dir = _build_wheels(envs_dir)
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        _run(["uv", "venv", "--clear", "--python", py_version, venv])
        env = {**os.environ, "VIRTUAL_ENV": str(venv)}
        env["PATH"] = f"{venv / 'bin'}{os.pathsep}{env['PATH']}"
        # Without this, uv resolves a multi-GB CUDA torch from the default index.
        if not _has_cuda_gpu() and (specs := _cpu_torch_specs()):
            _run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    python,
                    *specs,
                    "--index-url",
                    _TORCH_CPU_INDEX,
                ],
                env=env,
            )
        # The scorecard code imports pytest, which comes from the dev extra.
        wheels = [
            f"{next(wheel_dir.glob('qai_hub_models-*.whl'))}[dev]",
            f"{next(wheel_dir.glob('qai_hub_models_cli-*.whl'))}[internal]",
        ]
        _run(["uv", "pip", "install", "--python", python, *wheels], env=env)
        _run([venv / "bin" / "qai-hub-models", "install", model_id, "--yes"], env=env)
        (venv / _READY_MARKER).touch()
        print(
            f"{model_id} | env created in {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    return python
