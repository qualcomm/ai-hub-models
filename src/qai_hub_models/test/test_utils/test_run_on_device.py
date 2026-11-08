# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from qai_hub_models.utils.devicefarm.backends.qdc import run_on_device
from qai_hub_models.utils.devicefarm.backends.qdc.run_on_device import (
    build_run,
    expand_command,
)


def _args(**kwargs: object) -> argparse.Namespace:
    defaults = {
        "bundle_dir": "b",
        "command": "{bundle}/run.py --out {logs}",
        "requirements": None,
        "image_tar": None,
        "image_tag": None,
    }
    return argparse.Namespace(**{**defaults, **kwargs})


def test_expand_command_fills_paths_per_os() -> None:
    assert expand_command("{bundle}/run.py --out {logs}", True, "QDC_logs") == [
        "C:\\Temp\\TestContent/run.py",
        "--out",
        "C:\\Temp\\QDC_Logs",
    ]
    assert expand_command("{bundle}/run.py --out {logs}", False, "QDC_logs") == [
        "/work/run.py",
        "--out",
        "/out",
    ]


def test_expand_command_keeps_quoted_arguments_together() -> None:
    assert (
        expand_command("{bundle}/run.py --name 'a b'", False, "QDC_logs")[-1] == "a b"
    )


def test_windows_run_takes_requirements() -> None:
    run = build_run(_args(requirements="reqs.txt"), is_windows=True)
    assert run.requirements_file == "reqs.txt"
    assert run.bundle_dir == Path("b")


def test_windows_run_rejects_image_arguments() -> None:
    with pytest.raises(SystemExit, match="Linux"):
        build_run(_args(image_tar="i.tar", image_tag="i:1"), is_windows=True)


def test_linux_run_needs_image() -> None:
    with pytest.raises(SystemExit, match="image-tar"):
        build_run(_args(), is_windows=False)


def test_linux_run_rejects_requirements() -> None:
    with pytest.raises(SystemExit, match="Windows"):
        build_run(
            _args(requirements="r.txt", image_tar="i.tar", image_tag="i:1"),
            is_windows=False,
        )


def test_main_submits_the_run() -> None:
    device = SimpleNamespace(reference_device=SimpleNamespace(name="Some Device"))
    with (
        mock.patch.object(run_on_device, "_resolve_device", return_value=device),
        mock.patch.object(
            run_on_device,
            "HubDevicePlatform",
            return_value=SimpleNamespace(is_windows_platform=True),
        ),
        mock.patch.object(run_on_device, "get_device_farm") as get_farm,
        mock.patch.object(
            run_on_device, "submit_device_run", return_value="123"
        ) as submit,
    ):
        run_on_device.main(
            [
                "--device",
                "cs_x2_elite",
                "--bundle-dir",
                "b",
                "--command",
                "{bundle}/run.py",
                "--job-timeout",
                "900",
            ]
        )
    get_farm.assert_called_once_with(device)
    assert submit.call_args.kwargs["job_timeout"] == 900
    assert submit.call_args.args[1] == "Some Device"


def test_expand_command_only_replaces_placeholders_on_windows() -> None:
    assert expand_command("{bundle}/sub/run.py", True, "QDC_logs") == [
        "C:\\Temp\\TestContent/sub/run.py"
    ]
    assert expand_command("--cfg=http://h/x,{bundle}/a", True, "QDC_logs") == [
        "--cfg=http://h/x,C:\\Temp\\TestContent/a"
    ]
