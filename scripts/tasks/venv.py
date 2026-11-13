# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import functools
import os
import re
import shlex
import subprocess
from collections.abc import Iterable
from pathlib import Path

from .constants import (
    DEFAULT_PYTHON,
    DEV_REQUIREMENTS_PATH,
    GLOBAL_REQUIREMENTS_PATH,
    PY_CLI_INSTALL_ROOT,
    PY_PACKAGE_INSTALL_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    REPO_ROOT,
    REQUIREMENTS_PATH,
    bash_argv,
    venv_activate_command,
)
from .task import CompositeTask, RunCommandsTask, RunCommandsWithVenvTask
from .util import get_pip, get_pip_install_commands, has_cuda_gpu, uv_installed


@functools.cache
def get_package_version(
    package_name: str, requirements_path: str = REQUIREMENTS_PATH
) -> str | None:
    """Extract version constraint for a package from a requirements file."""
    if not os.path.exists(requirements_path):
        return None

    with open(requirements_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Remove inline comments
            line = line.split("#")[0].strip()
            # Match package name followed by version specifier
            match = re.match(rf"^{re.escape(package_name)}([<>=!~\[].*)$", line)
            if match:
                version = match.group(1)
                # Remove extras (e.g., diffusers[torch] -> diffusers)
                version = re.sub(r"\[.*?\]", "", version)
                # Remove environment markers (e.g., "; python_version>='3.8'")
                return version.split(";")[0].strip()
    return None


def get_torch_cpu_install_command(
    extras: Iterable[str] = (),
    search_global_reqs: bool = False,
    packages: list[str] | None = None,
) -> str | None:
    """
    Build a pip install command for torch packages from the CPU index.
    Returns None if no CUDA GPU is available or no packages need to be installed.

    Parameters
    ----------
    extras
        QAIHM extras being installed; "dev" adds the dev requirements' torch pins.
    search_global_reqs
        Whether to also check global requirements
    packages
        List of torch-related packages to check for version requirements.

    Returns
    -------
    str | None
        A pip install command for the torch CPU packages with appropriate versions, or None if no installation is needed.
        If a package is not found in any requirements file, it will be skipped and not included in the install command.
    """
    if has_cuda_gpu():
        return None

    # Check for torch package versions in the provided requirements files.
    # Requirements files that are later / last in the list take highest priority.
    reqfiles_to_check = [REQUIREMENTS_PATH]
    if search_global_reqs:
        reqfiles_to_check.append(GLOBAL_REQUIREMENTS_PATH)
    if "dev" in extras:
        reqfiles_to_check.append(DEV_REQUIREMENTS_PATH)

    # Get versions for each package.
    packages_to_versions: dict[str, str | None] = dict.fromkeys(
        packages or ["torch", "torchvision", "torchaudio", "torchcodec"]
    )
    for reqfile in reqfiles_to_check[::-1]:
        packages_to_versions = {
            package: version if version else get_package_version(package, reqfile)
            for package, version in packages_to_versions.items()
        }

    # Convert to version strings.
    pgk_install_strings = []
    for package, version in packages_to_versions.items():
        if version is not None:
            pgk_install_strings.append(f"'{package}{version}'")

    return (
        f"{get_pip()} install {' '.join(pgk_install_strings)} --index-url https://download.pytorch.org/whl/cpu"
        if pgk_install_strings
        else None
    )


class CreateVenvTask(RunCommandsTask):
    def __init__(
        self, venv_path: str | None, python_executable: str | None = None
    ) -> None:
        super().__init__(
            f"Creating virtual environment at {venv_path}",
            f'source "{Path(REPO_ROOT, "scripts", "util", "env_create.sh").as_posix()}" '
            f'"--python={python_executable or DEFAULT_PYTHON}" '
            + (f'"--venv={Path(venv_path).as_posix()}" ' if venv_path else "")
            + "--no-sync",
        )


def is_package_installed(package_name: str, venv_path: str | None = None) -> bool:
    if venv_path is not None:
        if not os.path.exists(venv_path):
            return False
        command = (
            f'{venv_activate_command(venv_path)} && python -c "import {package_name}"'
        )
    else:
        command = f'python -c "import {package_name}"'

    try:
        subprocess.check_call(bash_argv(command))
        return True
    except subprocess.CalledProcessError:
        return False


class GenerateGlobalRequirementsTask(RunCommandsWithVenvTask):
    # Global requirements change based on the python version,
    # and should therefore be regenerated before running any model tests.
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Generate Global Requirements",
            venv,
            ["python -m qai_hub_models.scripts.generate_global_requirements"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class AggregateScorecardResultsTask(RunCommandsWithVenvTask):
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Aggregate Scorecard Results",
            venv,
            ["python -m qai_hub_models.scripts.aggregate_scorecard_results"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class DownloadQDCWheelTask(RunCommandsWithVenvTask):
    # Needed to run tests relying on QDC (e.g. Genie exports)
    def __init__(
        self,
        venv: str | None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        super().__init__(
            "Download QDC Wheel",
            venv,
            [
                f"bash {REPO_ROOT}/scripts/ci/download-qdc-wheel.sh '{REPO_ROOT}'",
            ],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class DownloadQAIRTAutoSDKTask(RunCommandsWithVenvTask):
    # Downloads the QAIRT SDK for automotive devices from S3 (requires AWS credentials)
    def __init__(
        self,
        venv: str | None,
        output_path: str | None = None,
        env: dict[str, str] | None = None,
        raise_on_failure: bool = True,
        ignore_return_codes: list[int] | None = None,
    ) -> None:
        cmd = "python -m qai_hub_models.scripts.download_auto_qairt"
        if output_path:
            cmd += f" --output {shlex.quote(output_path)}"
        super().__init__(
            "Download QAIRT Auto SDK",
            venv,
            [cmd],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class InstallGlobalRequirementsTask(RunCommandsWithVenvTask):
    def __init__(self, venv_path: str | None) -> None:
        commands: list[str] = []

        if torch_cpu_cmd := get_torch_cpu_install_command(search_global_reqs=True):
            commands.append(torch_cpu_cmd)

        commands.append(f'{get_pip()} install -r "{GLOBAL_REQUIREMENTS_PATH}" ')

        super().__init__(
            group_name="Install Global Requirements",
            venv=venv_path,
            commands=commands,
            retries=2,
        )


# Path to the LLM grader requirements file, relative to REPO_ROOT.
GRADER_REQUIREMENTS_PATH = os.path.join(
    REPO_ROOT,
    "src",
    "qai_hub_models",
    "models",
    "templates",
    "llm",
    "grader",
    "requirements.txt",
)


class InstallLLMGraderRequirementsTask(RunCommandsWithVenvTask):
    """Install LLM grader deps + qai_hub_models (editable, no deps).

    The grader requires ``transformers>=5.2``, which conflicts with the
    older transformers pins used by LLM source repos. Pass ``--venv`` to
    install into a dedicated grader venv rather than the default one.

    The editable ``qai_hub_models`` install (``--no-deps``) lets the grader
    CLI import its support modules without pulling in the full repo
    dependency tree.
    """

    def __init__(self, venv_path: str | None) -> None:
        commands = []
        # Preinstall the cu130 torch so the requirements install sees it satisfied.
        if has_cuda_gpu():
            torch_version = get_package_version("torch", GRADER_REQUIREMENTS_PATH) or ""
            commands.append(
                f"{get_pip()} install 'torch{torch_version}' --index-url "
                "https://download.pytorch.org/whl/cu130"
            )
        commands += [
            f'{get_pip()} install -r "{GRADER_REQUIREMENTS_PATH}"',
            f'{get_pip()} install --no-deps -e "{PY_PACKAGE_INSTALL_ROOT}"',
        ]
        super().__init__(
            group_name="Install LLM Grader Requirements",
            venv=venv_path,
            commands=commands,
            retries=2,
        )


class InstallCLITask(RunCommandsWithVenvTask):
    """Install the qai_hub_models_cli package from a wheel or as editable."""

    def __init__(
        self,
        venv_path: str | None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "install_cli",
        junit_classname: str = "qai_hub_models_cli",
    ) -> None:
        if cli_wheel_dir is not None:
            commands = [
                f'WHL=$(ls {cli_wheel_dir}/qai_hub_models_cli-*.whl) && {get_pip()} install "${{WHL}}[internal]"',
            ]
            install_method = "wheel"
        else:
            commands = [
                f'{get_pip()} install -e "{PY_CLI_INSTALL_ROOT}[internal]"',
            ]
            install_method = "editable"
        super().__init__(
            group_name=f"Install CLI ({install_method})",
            venv=venv_path,
            commands=commands,
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
            retries=2,
        )


def _normalize_pip_command(raw: str) -> str:
    """Rewrite a manifest ``pip install ...`` command for the current tool.

    - Replaces the leading ``pip`` with the resolved pip binary (``uv pip``
      when uv is installed, otherwise ``pip``).
    - Strips ``--use-pep517`` when running under uv (uv treats PEP 517 as the
      default and rejects the flag).
    - Strips ``--no-build-isolation`` when running under plain pip (that
      flag is only meaningful under uv here — pip uses build isolation by
      default and the surrounding logic installs the required build deps).
    - Strips ``-y`` / ``--yes`` from ``uninstall`` under uv (uv never prompts
      and rejects the flag; plain pip needs it to skip the confirmation).
    """
    tokens = raw.split()
    assert tokens and tokens[0] == "pip", raw
    if uv_installed():
        tokens = [t for t in tokens if t != "--use-pep517"]
        if len(tokens) >= 2 and tokens[1] == "uninstall":
            tokens = [t for t in tokens if t not in ("-y", "--yes")]
    else:
        tokens = [t for t in tokens if t != "--no-build-isolation"]
    return " ".join([get_pip(), *tokens[1:]])


def _filter_pip_commands(
    commands: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Drop entries that don't apply to the current machine and normalize them."""
    on_gpu = has_cuda_gpu()
    kept: list[tuple[str, str]] = []
    for cmd, machine in commands:
        if machine == "gpu" and not on_gpu:
            continue
        if machine == "cpu" and on_gpu:
            continue
        kept.append((cmd, machine))
    return kept


def _needs_no_build_isolation_workaround(
    pre_commands: Iterable[tuple[str, str]],
    post_commands: Iterable[tuple[str, str]],
) -> bool:
    """Match the old venv.py behavior for ``--no-build-isolation`` /
    ``--use-pep517``.

    If any pre/post command sets one of those flags, the main QAIHM install
    won't get its build deps installed automatically. Pre-install
    setuptools/setuptools-scm to compensate.
    """
    for cmd, _ in list(pre_commands) + list(post_commands):
        if "--use-pep517" in cmd or "--no-build-isolation" in cmd:
            return True
    return False


class SyncLocalQAIHMVenvTask(CompositeTask):
    """Sync the provided environment with local QAIHM and the provided extras.

    Install order:
        1. torch (CPU-only build, when no CUDA GPU is present)
        2. build-dep bootstrap (setuptools-scm/setuptools) if any pre/post
           command requires ``--no-build-isolation`` / ``--use-pep517``
        3. each entry in ``pre_pip_install_commands`` (filtered by machine)
        4. ``pip install ...[extras]`` (the main QAIHM install)
        5. each entry in ``post_pip_install_commands`` (filtered by machine)
    """

    def __init__(
        self,
        venv_path: str | None,
        extras: Iterable[str] = [],
        pre_pip_install_commands: Iterable[tuple[str, str]] = (),
        post_pip_install_commands: Iterable[tuple[str, str]] = (),
        qaihm_wheel_dir: str | os.PathLike | None = None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
        junit_testsuite: str = "",
        junit_name: str = "",
        junit_classname: str = "",
    ) -> None:
        # Make sure extras name use dashes
        extras_str = (
            f"[{','.join(e.replace('_', '-') for e in extras)}]" if extras else ""
        )

        pre = _filter_pip_commands(pre_pip_install_commands)
        post = _filter_pip_commands(post_pip_install_commands)

        needs_no_build_isolation = _needs_no_build_isolation_workaround(pre, post)

        commands: list[str] = []

        if torch_cpu_cmd := get_torch_cpu_install_command(extras):
            commands.append(torch_cpu_cmd)

        if needs_no_build_isolation and (
            qaihm_wheel_dir is None or cli_wheel_dir is None
        ):
            # A pre/post command uses --no-build-isolation / --use-pep517,
            # which means pip/uv won't install the minimum build deps to
            # build the AI Hub Models wheel. Install them manually first.
            commands.append(
                f"{get_pip()} install 'setuptools-scm>=9,<10' 'setuptools>=80'"
            )

        for cmd, _ in pre:
            commands.append(_normalize_pip_command(cmd))

        if qaihm_wheel_dir is not None:
            # Find wheel file and install it (use relative path to work in both local and CI)
            commands.append(
                f"{get_pip()} install $(ls {qaihm_wheel_dir}/qai_hub_models-*.whl){extras_str}"
            )
            install_method = "wheel"
        else:
            # Local development: Use editable install
            commands.append(
                f'{get_pip()} install -e "{PY_PACKAGE_INSTALL_ROOT}{extras_str}"'
            )
            install_method = "editable"

        for cmd, _ in post:
            commands.append(_normalize_pip_command(cmd))

        super().__init__(
            group_name=f"Install QAIHM{extras_str} ({install_method})",
            tasks=[
                InstallCLITask(venv_path, cli_wheel_dir),
                RunCommandsWithVenvTask(
                    group_name=f"Install QAIHM{extras_str} ({install_method})",
                    venv=venv_path,
                    commands=commands,
                    retries=2,
                ),
            ],
            junit_xml_path=junit_xml_path,
            junit_testsuite=junit_testsuite,
            junit_name=junit_name,
            junit_classname=junit_classname,
        )


class SyncModelVenvTask(SyncLocalQAIHMVenvTask):
    """Sync the venv with QAIHM base. Model deps are installed later via `qai-hub-models install`."""

    def __init__(
        self,
        model_name: str,
        venv_path: str | None,
        include_dev_deps: bool = False,
        qaihm_wheel_dir: str | os.PathLike | None = None,
        cli_wheel_dir: str | os.PathLike | None = None,
        junit_xml_path: str | None = None,
    ) -> None:
        extras = ["dev"] if include_dev_deps else []
        super().__init__(
            venv_path,
            extras,
            qaihm_wheel_dir=qaihm_wheel_dir,
            cli_wheel_dir=cli_wheel_dir,
            junit_xml_path=junit_xml_path,
            junit_testsuite="pytest",
            junit_name="environment_setup",
            junit_classname=f"qai_hub_models.models.{model_name}",
        )


class SyncModelRequirementsVenvTask(RunCommandsWithVenvTask):
    """Sync the provided environment with requirements from model_name's requirements.txt.
    Will not re-install QAI Hub Models. Intended for speeding up CI compared to building an entirely new env for each model.
    """

    def __init__(
        self, model_name: str, venv_path: str | None, pip_force_install: bool = True
    ) -> None:
        requirements_txt = os.path.join(
            PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt"
        )
        pre = _filter_pip_commands(
            get_pip_install_commands(model_name, "pre_pip_install_commands")
        )
        post = _filter_pip_commands(
            get_pip_install_commands(model_name, "post_pip_install_commands")
        )

        commands: list[str] = [_normalize_pip_command(cmd) for cmd, _ in pre]
        if os.path.exists(requirements_txt):
            commands.append(
                f"{get_pip()} install {'--force-reinstall ' if pip_force_install else ''}"
                f'-r "{requirements_txt}"'
            )
        commands.extend(_normalize_pip_command(cmd) for cmd, _ in post)

        super().__init__(
            group_name=f"Install Model Requirements for {model_name}",
            venv=venv_path,
            commands=commands,
        )
