# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Unit tests for qai_hub_models.scripts.notify_nightly_failures."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from qai_hub_models.scripts.notify_nightly_failures import (
    categorize_failures,
    load_failed_jobs_json,
    main,
    render_issue_body,
)

_VERIFY = "Run Tests (py 3.10) / Verify Model Tests"
_WEBSITE = "Sync Performance Data with Public Website / Test Public Website Import"


def _run_main(tmp_path: Path, failed_jobs_json: Path) -> Path:
    out = tmp_path / "issues"
    argv = ["notify_nightly_failures"]
    for name in (_VERIFY, _WEBSITE):
        argv += ["--workflow-failure", name, "--workflow-failure-url", "https://x/job"]
    argv += [
        "--output-dir",
        str(out),
        "--repository",
        "org/repo",
        "--run-url",
        "https://x/run",
        "--ref-name",
        "main",
        "--failed-jobs-json",
        str(failed_jobs_json),
    ]
    with patch.object(sys, "argv", argv):
        main()
    return out


def test_categorize_failures() -> None:
    """Splits workbench vs general failures by job name."""
    names = [
        "Run Tests (py 3.10) / Verify Model Tests",
        "Run Tests (py 3.10) / Run QAIHM Tests",
        "Run Tests (py 3.10) / Pre-commit",
    ]
    urls = [
        "https://github.com/org/repo/actions/runs/123/job/456",
        "https://github.com/org/repo/actions/runs/123/job/789",
        "https://github.com/org/repo/actions/runs/123/job/111",
    ]
    workbench, general = categorize_failures(names, urls)
    assert len(workbench) == 1
    assert "Verify Model Tests" in workbench[0]["name"]
    assert len(general) == 2


def test_load_failed_jobs_json(tmp_path: Path) -> None:
    """Loads structured JSON; returns empty dict for missing file."""
    assert load_failed_jobs_json(None) == {}
    assert load_failed_jobs_json(str(tmp_path / "nonexistent.json")) == {}

    data = {"compile/resnet50_TFLITE": "https://dev.aihub.qualcomm.com/jobs/j123"}
    json_path = tmp_path / "failed-workbench-jobs.json"
    json_path.write_text(json.dumps(data))
    assert load_failed_jobs_json(str(json_path)) == data


def test_render_workbench_issue() -> None:
    """Workbench template includes failure links and AI Hub job URLs."""
    body = render_issue_body(
        "workbench_issue.j2",
        today="2026-04-24",
        failures=[{"name": "Verify Model Tests", "url": "https://example.com/job/1"}],
        failed_aihub_jobs={
            "compile/resnet50": "https://dev.aihub.qualcomm.com/jobs/j1"
        },
        run_url="https://github.com/org/repo/actions/runs/123",
        repository="org/repo",
        ref_name="main",
    )
    assert "Workbench Job Failures" in body
    assert "Verify Model Tests" in body
    assert "https://dev.aihub.qualcomm.com/jobs/j1" in body
    assert "View Workflow Run" in body


def test_render_general_issue() -> None:
    """General template includes failure links and triage steps."""
    body = render_issue_body(
        "general_issue.j2",
        today="2026-04-24",
        failures=[{"name": "Run QAIHM Tests", "url": "https://example.com/job/2"}],
        run_url="https://github.com/org/repo/actions/runs/123",
        repository="org/repo",
        ref_name="main",
    )
    assert "Test Failures" in body
    assert "Run QAIHM Tests" in body
    assert "Nightly Failure Log" in body


def test_issue_job_lists_scope_breeze_comments(tmp_path: Path) -> None:
    """Each issue's job list names only its own failures.

    The Breeze analyst scopes each issue's comment to this list; if the lists
    overlapped, both issues would again carry the same analysis.
    """
    aihub = tmp_path / "failed.json"
    aihub.write_text(json.dumps({"quantize/owl_vit_w8a16": "https://x/jobs/j1"}))
    out = _run_main(tmp_path, aihub)
    assert (out / "workbench_jobs.txt").read_text() == _VERIFY
    assert (out / "general_jobs.txt").read_text() == _WEBSITE

    # No AI Hub job failed, so the verify failure is infra and moves to the
    # general issue — its job list must follow it there.
    out = _run_main(tmp_path / "no_aihub", tmp_path / "missing.json")
    assert not (out / "workbench_jobs.txt").exists()
    assert (out / "general_jobs.txt").read_text() == f"{_WEBSITE}; {_VERIFY}"


def test_issue_bodies_fit_github_cap_on_mass_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub rejects bodies over 65,536 chars, which silences the Slack alert.

    On 2026-11-10, 980 failed LLM compile jobs pushed the workbench body to ~198 KB.
    """
    summary_dir = tmp_path / "build" / "nightly-test-results"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.md").write_text("## Nightly Test Summary\n" + "x" * 400_000)
    monkeypatch.chdir(tmp_path)

    # Names slightly longer than that night's longest (~104 chars).
    aihub = tmp_path / "failed.json"
    aihub.write_text(
        json.dumps(
            {
                f"compile/{'qwen3_4b_instruct_w4a16_PRECOMPILED_QNN_ONNX' * 2}_cs_8_elite_{i}": f"https://workbench.aihub.qualcomm.com/jobs/j{i:08d}/"
                for i in range(980)
            }
        )
    )
    out = _run_main(tmp_path, aihub)
    workbench = (out / "workbench_issue.md").read_text()
    assert len(workbench) <= 65_536
    assert "## Failed AI Hub Jobs (first 100 of 980)" in workbench
    assert "and 880 more" in workbench
    assert len((out / "general_issue.md").read_text()) <= 65_536


def test_issue_summaries_show_only_own_failure_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stack trace on the wrong issue sends triagers to the wrong place.

    On 2026-10-07 the centernet quantize-timeout trace appeared on the general
    issue too, and the triage discussion landed there instead of the workbench one.
    """
    summary = (
        "## Nightly Test Summary\n\n"
        "### Workflow Failures\n\nWORKFLOW_TABLE\n\n"
        "### Test Results\n\nRESULTS_TABLE\n\n"
        "### py3.10 Verify Workbench Jobs Failures\n\nWORKBENCH_TRACE\n"
        "#### Stack Traces\nWORKBENCH_STACK\n"
        "### py3.10 Model Tests Failures\n\nMODEL_TRACE\n"
    )
    summary_dir = tmp_path / "build" / "nightly-test-results"
    summary_dir.mkdir(parents=True)
    (summary_dir / "summary.md").write_text(summary)
    monkeypatch.chdir(tmp_path)

    aihub = tmp_path / "failed.json"
    aihub.write_text(json.dumps({"quantize/centernet_3d": "https://x/jobs/j1"}))
    out = _run_main(tmp_path, aihub)
    workbench = (out / "workbench_issue.md").read_text()
    general = (out / "general_issue.md").read_text()
    for body in (workbench, general):
        assert "WORKFLOW_TABLE" in body
        assert "RESULTS_TABLE" in body
    assert "WORKBENCH_STACK" in workbench
    assert "MODEL_TRACE" not in workbench
    assert "MODEL_TRACE" in general
    assert "WORKBENCH_TRACE" not in general

    # With no workbench issue filed, the general issue is the only place the
    # verify failure is reported, so it must keep that trace.
    out = _run_main(tmp_path / "no_aihub", tmp_path / "missing.json")
    general = (out / "general_issue.md").read_text()
    assert "WORKBENCH_STACK" in general
    assert "MODEL_TRACE" in general
