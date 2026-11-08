# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import re
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from qai_hub_models.utils.devicefarm.backends.qdc import device_run
from qai_hub_models.utils.devicefarm.backends.qdc.device_run import (
    COMMAND_FILE,
    DEFAULT_PYTHON_EMBED_SHA256,
    GET_PIP_SHA256,
    DeviceRun,
    build_bundle,
    bundle_path,
    log_path,
    render_command_file,
    render_linux_script,
    render_windows_script,
    submit_device_run,
)

PLACEHOLDER = re.compile(r"\{[A-Z_]+\}")


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "models" / "a").mkdir(parents=True)
    (bundle / "models" / "a" / "w.onnx").write_text("x")
    (bundle / "run.py").write_text("print(1)")
    (bundle / "reqs.txt").write_text("numpy\n")
    return bundle


def _windows_run(bundle: Path) -> DeviceRun:
    return DeviceRun(
        bundle_dir=bundle,
        command=[bundle_path(True, "run.py"), "--out", log_path(True, "QDC_logs")],
        requirements_file="reqs.txt",
    )


def _linux_run(bundle: Path) -> DeviceRun:
    return DeviceRun(
        bundle_dir=bundle,
        command=[bundle_path(False, "run.py"), "--out", log_path(False, "QDC_logs")],
        image_tar="img.tar",
        image_tag="img:latest",
    )


def test_bundle_path_and_log_path_per_os() -> None:
    assert (
        bundle_path(True, "models/a/w.onnx")
        == "C:\\Temp\\TestContent\\models\\a\\w.onnx"
    )
    assert bundle_path(False, "models/a/w.onnx") == "/work/models/a/w.onnx"
    assert bundle_path(True, "") == "C:\\Temp\\TestContent"
    assert bundle_path(False, "") == "/work"
    assert log_path(True, "QDC_logs") == "C:\\Temp\\QDC_Logs"
    assert log_path(False, "QDC_logs") == "/out"


def test_windows_script_has_no_unfilled_placeholders(tmp_path: Path) -> None:
    script = render_windows_script(_windows_run(_bundle(tmp_path)))
    assert not PLACEHOLDER.findall(script)
    assert "reqs.txt" in script
    assert "python-3.12.7-embed-amd64.zip" in script
    assert COMMAND_FILE in script


def test_windows_script_installs_requirements_in_one_call_then_diagnoses(
    tmp_path: Path,
) -> None:
    script = render_windows_script(_windows_run(_bundle(tmp_path)))
    one_call = script.index('"pip_all"')
    per_line = script.index('("pip_{0:D3}" -f $index)')
    # The per-line installs are only the failure diagnosis inside the else path.
    assert one_call < script.index("if ($rc -ne 0) {", one_call) < per_line
    assert '-r `"$Root\\$ReqFile`"' in script
    assert script.count("throw") >= 3


def test_requirements_file_name_must_be_safe_for_cmd(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / 'r"eqs.txt').write_text("numpy\n")
    run = DeviceRun(
        bundle_dir=bundle, command=["run.py"], requirements_file='r"eqs.txt'
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="requirements_file"):
        build_bundle(run, True, "QDC_logs", scratch)


def test_windows_script_verifies_every_download(tmp_path: Path) -> None:
    script = render_windows_script(_windows_run(_bundle(tmp_path)))
    assert DEFAULT_PYTHON_EMBED_SHA256 in script
    assert GET_PIP_SHA256 in script
    assert "bootstrap.pypa.io" not in script


def test_overriding_the_python_url_needs_its_own_hash(tmp_path: Path) -> None:
    run = DeviceRun(
        bundle_dir=_bundle(tmp_path),
        command=["run.py"],
        python_embed_url="https://example.com/python.zip",
    )
    with pytest.raises(ValueError, match="python_embed_sha256"):
        render_windows_script(run)
    pinned = DeviceRun(
        bundle_dir=run.bundle_dir,
        command=["run.py"],
        python_embed_url="https://example.com/python.zip",
        python_embed_sha256="ab" * 32,
    )
    assert "ab" * 32 in render_windows_script(pinned)


def test_command_file_has_one_argument_per_line(tmp_path: Path) -> None:
    run = DeviceRun(bundle_dir=_bundle(tmp_path), command=["run.py", "it's", "a b&c"])
    assert render_command_file(run) == "run.py\nit's\na b&c\n"


@pytest.mark.parametrize("bad", ['say "hi"', "50%", "a\nb", "C:\\dir\\"])
def test_command_file_rejects_arguments_cmd_cannot_take_safely(
    tmp_path: Path, bad: str
) -> None:
    run = DeviceRun(bundle_dir=_bundle(tmp_path), command=["run.py", bad])
    with pytest.raises(ValueError, match="Command argument"):
        render_command_file(run)


def test_build_bundle_rejects_an_unsafe_requirement(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "reqs.txt").write_text('numpy\nfoo ; python_version == "3.12"\n')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match=r"Requirement in reqs\.txt"):
        build_bundle(_windows_run(bundle), True, "QDC_logs", scratch)


def test_linux_script_has_no_unfilled_placeholders(tmp_path: Path) -> None:
    script = render_linux_script(_linux_run(_bundle(tmp_path)), "QDC_logs")
    assert not PLACEHOLDER.findall(script)
    assert "/data/local/tmp/QDC_logs" in script
    assert '"img:latest" python /work/run.py --out /out' in script
    assert "libdmabufheap" in script


def test_linux_script_quotes_arguments(tmp_path: Path) -> None:
    run = DeviceRun(
        bundle_dir=_bundle(tmp_path),
        command=["run.py", "a b"],
        image_tar="img.tar",
        image_tag="img:latest",
    )
    assert "run.py 'a b'" in render_linux_script(run, "QDC_logs")


def test_linux_run_requires_image(tmp_path: Path) -> None:
    run = DeviceRun(bundle_dir=_bundle(tmp_path), command=["run.py"])
    with pytest.raises(ValueError, match="image_tar"):
        render_linux_script(run, "QDC_logs")


@pytest.mark.parametrize("is_windows", [True, False])
def test_build_bundle_contains_files_and_script(
    tmp_path: Path, is_windows: bool
) -> None:
    bundle = _bundle(tmp_path)
    run = _windows_run(bundle) if is_windows else _linux_run(bundle)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    entries, entry_script = build_bundle(run, is_windows, "QDC_logs", scratch)

    arcnames = {arc.replace("\\", "/") for _, arc in entries}
    assert {"run.py", "models/a/w.onnx"} <= arcnames
    script_name = "run_windows.ps1" if is_windows else "run_linux_container.sh"
    assert script_name in arcnames
    assert (COMMAND_FILE in arcnames) == is_windows
    assert entry_script.endswith(script_name)
    assert entry_script.startswith("C:\\" if is_windows else "/bin/bash ")


def test_command_file_rejects_an_empty_argument(tmp_path: Path) -> None:
    run = DeviceRun(bundle_dir=_bundle(tmp_path), command=["run.py", "", "--x"])
    with pytest.raises(ValueError, match="Empty command argument"):
        render_command_file(run)


def test_build_bundle_without_requirements(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    run = DeviceRun(bundle_dir=_bundle(tmp_path), command=["run.py"])
    entries, _ = build_bundle(run, True, "QDC_logs", scratch)
    assert "reqs.txt" in {Path(arc).name for _, arc in entries}


def test_build_bundle_rejects_a_missing_requirements_file(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    run = DeviceRun(
        bundle_dir=_bundle(tmp_path),
        command=["run.py"],
        requirements_file="nope.txt",
    )
    with pytest.raises(FileNotFoundError, match=r"nope\.txt"):
        build_bundle(run, True, "QDC_logs", scratch)


@pytest.mark.parametrize("reserved", ["run_windows.ps1", "device_command.txt"])
def test_build_bundle_rejects_reserved_file_names(
    tmp_path: Path, reserved: str
) -> None:
    bundle = _bundle(tmp_path)
    (bundle / reserved).write_text("mine")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match=re.escape(reserved)):
        build_bundle(_windows_run(bundle), True, "QDC_logs", scratch)


def test_build_bundle_allows_reserved_names_in_subfolders(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "models" / "run_windows.ps1").write_text("nested")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    entries, _ = build_bundle(_windows_run(bundle), True, "QDC_logs", scratch)
    assert any(arc.replace("\\", "/") == "models/run_windows.ps1" for _, arc in entries)


@pytest.mark.parametrize("line", ["--extra-index-url https://x/simple", "-r other.txt"])
def test_requirements_reject_pip_option_lines(tmp_path: Path, line: str) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "reqs.txt").write_text(f"numpy\n{line}\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match="pip option"):
        build_bundle(_windows_run(bundle), True, "QDC_logs", scratch)


def test_requirements_ignore_blank_comment_and_trailing_comment_lines(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "reqs.txt").write_text(
        "# header\n\n   \nnumpy==1.26  # pinned for x\n  pandas>2  \n"
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    build_bundle(_windows_run(bundle), True, "QDC_logs", scratch)


def test_requirements_still_reject_unsafe_characters(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "reqs.txt").write_text('foo ; python_version == "3.12"  # quoted\n')
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(ValueError, match=r"Requirement in reqs\.txt"):
        build_bundle(_windows_run(bundle), True, "QDC_logs", scratch)


def test_build_bundle_rejects_missing_dir(tmp_path: Path) -> None:
    run = DeviceRun(bundle_dir=tmp_path / "missing", command=["run.py"])
    with pytest.raises(FileNotFoundError):
        build_bundle(run, True, "QDC_logs", tmp_path)


@pytest.mark.parametrize(
    ("attrs", "expect_ps1"),
    [
        (SimpleNamespace(is_windows_platform=True, is_iot_platform=False), True),
        (SimpleNamespace(is_windows_platform=False, is_iot_platform=True), False),
    ],
)
def test_submit_device_run_picks_script_by_os(
    tmp_path: Path, attrs: SimpleNamespace, expect_ps1: bool
) -> None:
    bundle = _bundle(tmp_path)
    run = _windows_run(bundle) if expect_ps1 else _linux_run(bundle)
    backend = mock.MagicMock()
    backend.submit_bundle.return_value = "123"
    with (
        mock.patch.object(device_run, "HubDevicePlatform", return_value=attrs),
        mock.patch.object(device_run, "device_logs_dir_name", return_value="QDC_logs"),
    ):
        job_id = submit_device_run(backend, "Some Device", run, "job", 900)

    assert job_id == "123"
    kwargs = backend.submit_bundle.call_args.kwargs
    assert kwargs["hub_device_name"] == "Some Device"
    assert kwargs["job_timeout"] == 900
    assert ("run_windows.ps1" in kwargs["entry_script"]) == expect_ps1


def test_submit_device_run_rejects_other_platforms(tmp_path: Path) -> None:
    attrs = SimpleNamespace(is_windows_platform=False, is_iot_platform=False)
    with (
        mock.patch.object(device_run, "HubDevicePlatform", return_value=attrs),
        pytest.raises(NotImplementedError),
    ):
        submit_device_run(
            mock.MagicMock(), "Phone", _linux_run(_bundle(tmp_path)), "job", 900
        )
