# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run one Python command on a Windows-on-Snapdragon device through a device farm.

A "device run" is a directory of files plus a Python command. The helpers here
turn it into a farm bundle with the generic Windows device script and submit it:
x64 Python is installed on the device, a requirements file is pip-installed,
then the command runs.

Model code is not involved: callers supply the directory, the command and the
requirements file.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qai_hub_models.utils.devicefarm.devicefarm import (
    DeviceFarm,
    HubDevicePlatform,
    walk_dir_entries,
)

DEVICE_SCRIPTS_DIR = Path(__file__).parent / "device_scripts"
WINDOWS_SCRIPT = "run_windows.ps1"

# Emulated x64 on purpose; see run_windows.ps1. The device checks each download
# against its SHA256 before using it.
DEFAULT_PYTHON_EMBED_URL = (
    "https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip"
)
DEFAULT_PYTHON_EMBED_SHA256 = (
    "0d57bb6cb078b74d23dbfe91f77d6780d45bed328911609f1f7ee2ba1606bf44"
)
# A pinned pypa/get-pip commit; bootstrap.pypa.io changes with every pip release.
GET_PIP_URL = (
    "https://raw.githubusercontent.com/pypa/get-pip/"
    "37ae3c4daaf63a0c203176251fb99c0e238562b5/public/get-pip.py"
)
GET_PIP_SHA256 = "f4448a01e3ab2914c65f4ca59d4cacba6b2dbde19a483fbac46587f8bbaff424"

# File in the bundle holding the python arguments, one per line.
COMMAND_FILE = "device_command.txt"
# The device script wraps each argument and requirement in double quotes for
# cmd.exe, where only a quote or a %variable% expansion can break out.
_UNSAFE_CHARS = re.compile(r'["%\r\n]')
# Same rule as run_windows.ps1: a "#" after whitespace starts a comment.
_INLINE_COMMENT = re.compile(r"\s#.*$")

WINDOWS_DEVICE_ROOT = "C:\\Temp\\TestContent"
# QDC only harvests files written under this directory name.
WINDOWS_LOG_DIR = "C:\\Temp\\QDC_Logs"


@dataclass(frozen=True)
class DeviceRun:
    """What to run on a device.

    Attributes
    ----------
    bundle_dir
        Directory shipped to the device; its contents appear under
        ``C:/Temp/TestContent``.
    command
        Arguments to ``python`` (the script path first). Build paths to files in
        the bundle with :func:`bundle_path`.
    requirements_file
        File in ``bundle_dir`` with one pip requirement per line (blank lines and
        ``#`` comments are ignored; pip option lines such as ``--extra-index-url``
        are rejected). The file is installed with one
        ``pip install --only-binary=:all: -r`` call, so the requirements must
        resolve together and use packages that ship wheels: an sdist-only package
        fails with an opaque pip error. If that call fails, each line is
        installed on its own (one log per line) to show which ones fail, and the
        run fails either way.
    python_embed_url
        Embeddable Python zip to install.
    python_embed_sha256
        SHA256 of ``python_embed_url``; required when the URL is overridden.
    """

    bundle_dir: Path
    command: Sequence[str]
    requirements_file: str | None = None
    python_embed_url: str = DEFAULT_PYTHON_EMBED_URL
    python_embed_sha256: str = DEFAULT_PYTHON_EMBED_SHA256


def bundle_path(relative: str) -> str:
    """Path of a bundle file as seen by the device command; "" is the bundle root.

    No trailing separator: a Windows argument ending in a backslash before its
    closing quote is parsed as an escaped quote.
    """
    path = f"{WINDOWS_DEVICE_ROOT}\\{relative.replace('/', chr(92))}"
    return path.rstrip("\\")


def log_path() -> str:
    """Directory the device command should write result files to."""
    return WINDOWS_LOG_DIR


def _fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def render_windows_script(run: DeviceRun) -> str:
    """Fill ``run_windows.ps1`` for ``run``."""
    if (
        run.python_embed_url != DEFAULT_PYTHON_EMBED_URL
        and run.python_embed_sha256 == DEFAULT_PYTHON_EMBED_SHA256
    ):
        raise ValueError(
            "python_embed_url is overridden: pass its python_embed_sha256."
        )
    template = (DEVICE_SCRIPTS_DIR / WINDOWS_SCRIPT).read_text(encoding="utf-8")
    return _fill(
        template,
        {
            "PYTHON_EMBED_URL": run.python_embed_url,
            "PYTHON_EMBED_SHA256": run.python_embed_sha256,
            "GET_PIP_URL": GET_PIP_URL,
            "GET_PIP_SHA256": GET_PIP_SHA256,
            "REQUIREMENTS_FILE": run.requirements_file or "",
            "COMMAND_FILE": COMMAND_FILE,
        },
    )


def _check_safe(value: str, what: str) -> None:
    if _UNSAFE_CHARS.search(value):
        raise ValueError(
            f'{what} {value!r} has a double quote, a "%" or a line break; '
            "cmd.exe on the device cannot be given those safely."
        )


def render_command_file(run: DeviceRun) -> str:
    r"""The python arguments, one per line, as ``run_windows.ps1`` reads them.

    Example: ``["C:\x\run.py", "it's"]`` becomes two lines, which the script
    runs as ``python.exe "C:\x\run.py" "it's"``.
    """
    for arg in run.command:
        if not arg:
            raise ValueError(
                "Empty command argument; a blank line in the command file would "
                "be dropped on the device and shift the arguments after it."
            )
        _check_safe(arg, "Command argument")
        if arg.endswith("\\"):
            raise ValueError(
                f"Command argument {arg!r} ends in a backslash, which would escape "
                "its closing quote; drop the trailing backslash."
            )
    return "".join(f"{arg}\n" for arg in run.command)


def _requirement_lines(path: Path) -> list[str]:
    """Requirement lines as the device script installs them, one pip call each."""
    lines = (
        _INLINE_COMMENT.sub("", raw).strip()
        for raw in path.read_text(encoding="utf-8").splitlines()
    )
    return [line for line in lines if line and not line.startswith("#")]


def _check_requirements(path: Path) -> None:
    for line in _requirement_lines(path):
        if line.startswith("-"):
            raise ValueError(
                f"{path.name}: {line!r} is a pip option; each line is installed "
                "as one requirement, so options cannot be passed."
            )
        _check_safe(line, f"Requirement in {path.name}:")


def build_bundle(
    run: DeviceRun, scratch_dir: Path
) -> tuple[list[tuple[str, str]], str]:
    """Return ``(entries, entry_script)`` for ``DeviceFarm.submit_bundle``.

    The rendered device script and the command file are written into
    ``scratch_dir``, which must outlive the submission.
    """
    if not run.bundle_dir.is_dir():
        raise FileNotFoundError(f"bundle_dir {run.bundle_dir} is not a directory.")
    for reserved in (WINDOWS_SCRIPT, COMMAND_FILE):
        if (run.bundle_dir / reserved).exists():
            raise ValueError(
                f"bundle_dir already has {reserved}, which the helper generates; "
                "rename it."
            )
    if run.requirements_file:
        _check_safe(run.requirements_file, "requirements_file")
        requirements = run.bundle_dir / run.requirements_file
        if not requirements.is_file():
            raise FileNotFoundError(
                f"requirements_file {run.requirements_file!r} is not in {run.bundle_dir}."
            )
        _check_requirements(requirements)

    script_path = scratch_dir / WINDOWS_SCRIPT
    script_path.write_text(render_windows_script(run), encoding="utf-8", newline="\n")
    command_path = scratch_dir / COMMAND_FILE
    command_path.write_text(render_command_file(run), encoding="utf-8", newline="\n")

    entries = walk_dir_entries(os.fspath(run.bundle_dir))
    entries.append((os.fspath(script_path), WINDOWS_SCRIPT))
    entries.append((os.fspath(command_path), COMMAND_FILE))
    return entries, f"{WINDOWS_DEVICE_ROOT}\\{WINDOWS_SCRIPT}"


def submit_device_run(
    backend: DeviceFarm,
    hub_device_name: str,
    run: DeviceRun,
    job_name: str,
    job_timeout: int,
) -> str:
    r"""Submit ``run`` to the device named ``hub_device_name`` and return the job id.

    Does not wait for the job; poll the backend (``backend.is_done`` /
    ``backend.get_job_log_files``) as usual.

    Parameters
    ----------
    backend
        A QDC device farm. The device paths (``C:\Temp\TestContent``,
        ``QDC_Logs``) and the entry-script handling are QDC-specific; AWS Device
        Farm treats ``entry_script`` differently. Typed as ``DeviceFarm`` because
        the QDC class imports the QDC SDK, which this module avoids at import time.
    hub_device_name
        Hub name of a Windows device, e.g. ``"Snapdragon X2 Elite CRD"``.
    run
        What to run.
    job_name
        Name of the farm job.
    job_timeout
        Job timeout in seconds.

    Returns
    -------
    job_id : str
        The farm's id for the submitted job.
    """
    if not HubDevicePlatform(hub_device_name).is_windows_platform:
        raise NotImplementedError(
            f"This device run supports Windows devices; {hub_device_name!r} is not one."
        )
    with tempfile.TemporaryDirectory() as scratch:
        entries, entry_script = build_bundle(run, Path(scratch))
        return backend.submit_bundle(
            hub_device_name=hub_device_name,
            entries=entries,
            entry_script=entry_script,
            job_name=job_name,
            job_timeout=job_timeout,
        )
