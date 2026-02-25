# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterable

from .constants import (
    DEFAULT_PYTHON,
    GLOBAL_REQUIREMENTS_PATH,
    PY_PACKAGE_INSTALL_ROOT,
    PY_PACKAGE_MODELS_ROOT,
    REPO_ROOT,
)
from .task import RunCommandsTask, RunCommandsWithVenvTask
from .util import get_code_gen_str_field, get_pip, uv_installed


class CreateVenvTask(RunCommandsTask):
    def __init__(self, venv_path: str, python_executable: str | None = None) -> None:
        super().__init__(
            f"Creating virtual environment at {venv_path}",
            f"source {REPO_ROOT}/scripts/util/env_create.sh --python={python_executable or DEFAULT_PYTHON} --venv={venv_path} --no-sync",
        )


def is_package_installed(package_name: str, venv_path: str | None = None) -> bool:
    if venv_path is not None:
        if not os.path.exists(venv_path):
            return False
        command = f'. {venv_path}/bin/activate && python -c "import {package_name}"'
    else:
        command = f'python -c "import {package_name}"'

    try:
        subprocess.check_call(command, shell=True)
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
            ["python -m qai_hub_models.scripts.download_qdc_tools"],
            env,
            raise_on_failure,
            ignore_return_codes or [],
        )


class InstallGlobalRequirementsTask(RunCommandsWithVenvTask):
    def __init__(self, venv_path: str | None) -> None:
        super().__init__(
            group_name="Install Global Requirements",
            venv=venv_path,
            commands=[
                f'{get_pip()} install -r "{GLOBAL_REQUIREMENTS_PATH}" ',
            ],
        )


class SyncLocalQAIHMVenvTask(RunCommandsWithVenvTask):
    """Sync the provided environment with local QAIHM and the provided extras."""

    def __init__(
        self,
        venv_path: str | None,
        extras: Iterable[str] = [],
        flags: str | None = None,
        pre_install: str | None = None,
        qaihm_wheel_dir: str | os.PathLike | None = None,
    ) -> None:
        extras_str = f"[{','.join(extras)}]" if extras else ""

        if flags is not None and uv_installed():
            # use pep 517 is default behavior for UV, and therefore is not a valid arg.
            flags = flags.replace("--use-pep517", "")
        if flags is not None and not uv_installed():
            # This flag disables the `--use-pep517` behavior for uv. This is the default for pip, and is not a valid pip arg.
            flags = flags.replace("--no-build-isolation", "")

        if qaihm_wheel_dir is not None:
            # Find wheel file and install it (use relative path to work in both local and CI)
            commands = [
                f"{get_pip()} install $(ls {qaihm_wheel_dir}/qai_hub_models-*.whl){extras_str} {flags or ''}"
            ]
            install_method = "wheel"
        else:
            # Local development: Use editable install
            commands = [
                f'{get_pip()} install -e "{PY_PACKAGE_INSTALL_ROOT}{extras_str}" {flags or ""}'
            ]
            install_method = "editable"
        if pre_install:
            commands.insert(0, f"{get_pip()} install {pre_install}")

        super().__init__(
            group_name=f"Install QAIHM{extras_str} ({install_method})",
            venv=venv_path,
            commands=commands,
        )


class SyncModelVenvTask(SyncLocalQAIHMVenvTask):
    """Sync the provided environment with local QAIHM and the provided extras needed for the model_name."""

    def __init__(
        self,
        model_name: str,
        venv_path: str | None,
        include_dev_deps: bool = False,
        qaihm_wheel_dir: str | os.PathLike | None = None,
    ) -> None:
        extras = []
        if include_dev_deps:
            extras.append("dev")
        if os.path.exists(
            os.path.join(PY_PACKAGE_MODELS_ROOT, model_name, "requirements.txt")
        ):
            extras.append(model_name)

        super().__init__(
            venv_path,
            extras,
            get_code_gen_str_field(model_name, "pip_install_flags"),
            get_code_gen_str_field(model_name, "pip_pre_build_reqs"),
            qaihm_wheel_dir,
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
        extra_flags = get_code_gen_str_field(model_name, "pip_install_flags")
        pre_install = get_code_gen_str_field(model_name, "pip_pre_build_reqs")
        if os.path.exists(requirements_txt):
            commands = [
                f'{get_pip()} install {"--force-reinstall" if pip_force_install else None} -r "{requirements_txt}" {extra_flags or ""}'
            ]
            if pre_install:
                commands.insert(0, f"{get_pip()} install {pre_install}")
        else:
            commands = []

        super().__init__(
            group_name=f"Install Model Requirements for {model_name}",
            venv=venv_path,
            commands=commands,
        )
