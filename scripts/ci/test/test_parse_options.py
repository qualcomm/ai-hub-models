# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for parse_options.parse and its CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_CI_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CI_DIR))
from parse_options import parse, validate_single_model  # noqa: E402

_SCRIPT = _CI_DIR / "parse_options.py"

_CHECKPOINT_KEY = "ephemeral_quantize_assets/gh_actions/123/qwen3_0_6b-w4a16.zip"


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("", {}),
        ("qairt_version=2.45", {"qairt_version": "2.45"}),
        (
            "qairt_version=2.45,geniex_version=0.3.7",
            {"qairt_version": "2.45", "geniex_version": "0.3.7"},
        ),
        ("geniex_version=0.3.7", {"geniex_version": "0.3.7"}),
        # Order is irrelevant; whitespace and stray commas are tolerated.
        (
            "  geniex_version = 0.3.7 , , qairt_version = 2.45 ",
            {"qairt_version": "2.45", "geniex_version": "0.3.7"},
        ),
        # An explicitly empty qairt_version falls back to the default sentinel.
        ("qairt_version=", {}),
        ("genie_scope=perf", {"genie_scope": "perf"}),
        (f"checkpoint={_CHECKPOINT_KEY}", {"checkpoint": _CHECKPOINT_KEY}),
        (
            f"qairt_version=2.45,checkpoint={_CHECKPOINT_KEY}",
            {"qairt_version": "2.45", "checkpoint": _CHECKPOINT_KEY},
        ),
        ("eval_dataset=grace2", {"eval_dataset": "grace2"}),
    ],
)
def test_parse_ok(raw: str, want: dict[str, str]) -> None:
    expected = {
        "qairt_version": "qaihm_default",
        "geniex_version": "",
        "genie_scope": "",
        "checkpoint": "",
        "eval_dataset": "",
    }
    expected.update(want)
    assert parse(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected_msg"),
    [
        # Bare (positional) values are not part of the grammar.
        ("2.45", "bare value"),
        ("qaihm_default", "bare value"),
        ("2.45,geniex_version=0.3.7", "bare value"),
        ("unknown_key=x", "Unknown option key"),
        ("qairt_version=2.45,qairt_version=2.50", "Duplicate option key"),
        ("geniex_version=0.3.7,geniex_version=0.3.8", "Duplicate option key"),
        # Bare SemVer required; leading v is rejected.
        ("geniex_version=v0.3.7", "must not start with 'v'"),
        ("genie_scope=compile_only", "genie_scope must be one of"),
        # checkpoint takes a bare key in the private bucket, not a URL.
        (
            f"checkpoint=s3://qai-hub-models-private-assets/{_CHECKPOINT_KEY}",
            "not a URL",
        ),
        ("checkpoint=https://example.com/x.zip", "not a URL"),
        (f"checkpoint=/{_CHECKPOINT_KEY}", "without '\\.\\.'"),
        ("checkpoint=a/../../etc/passwd.zip", "without '\\.\\.'"),
        ("checkpoint=a\\b.zip", "without '\\.\\.'"),
        # The override points at a quantize_and_upload zip, not a directory.
        ("checkpoint=ephemeral_quantize_assets/gh_actions/123/w4a16", "must name a"),
        ("checkpoint=some/key/model.encodings", "must name a"),
    ],
)
def test_parse_errors(raw: str, expected_msg: str) -> None:
    with pytest.raises(ValueError, match=expected_msg):
        parse(raw)


def test_validate_single_model_skips_when_models_unavailable() -> None:
    # scorecard.yml's LLM run parses options before it knows its model list, so
    # it passes None. Enforcing there would reject every legitimate run.
    validate_single_model(_CHECKPOINT_KEY, None)


@pytest.mark.parametrize(
    "models",
    [
        # A checkpoint is only correct for the one model it was quantized from.
        "",
        "all",
        "PYTORCH_NO_LLM",
        "qwen3_0_6b,qwen3_1_7b",
    ],
)
def test_validate_single_model_rejects(models: str) -> None:
    with pytest.raises(ValueError, match="exactly one model id"):
        validate_single_model(_CHECKPOINT_KEY, models)


def test_validate_single_model_accepts_one_model() -> None:
    validate_single_model(_CHECKPOINT_KEY, "qwen3_0_6b")


def test_cli_writes_github_output(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45,geniex_version=0.3.7",
            "--github-output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    text = out.read_text()
    assert "qairt_version=2.45\n" in text
    assert "geniex_version=0.3.7\n" in text


def test_cli_writes_defaults_for_empty_input(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    res = subprocess.run(
        [sys.executable, str(_SCRIPT), "--raw", "", "--github-output", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    text = out.read_text()
    assert "qairt_version=qaihm_default\n" in text
    assert "geniex_version=\n" in text


def test_cli_allows_checkpoint_when_no_models_env(tmp_path: Path) -> None:
    # The regression from run 37517995061: MODELS unset must not be read as an
    # empty models input, or the LLM on-device run fails on a valid dispatch.
    out = tmp_path / "gh_output"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            f"checkpoint={_CHECKPOINT_KEY}",
            "--github-output",
            str(out),
        ],
        capture_output=True,
        text=True,
        env={"PATH": ""},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"checkpoint={_CHECKPOINT_KEY}\n" in out.read_text()


def test_cli_emits_env_when_requested(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    env_file = tmp_path / "gh_env"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            f"qairt_version=2.45,geniex_version=0.3.7,checkpoint={_CHECKPOINT_KEY}",
            "--models",
            "qwen3_0_6b",
            "--github-output",
            str(out),
            "--github-env",
            str(env_file),
            "--emit-env",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    env_text = env_file.read_text()
    assert "QAIHM_TEST_QAIRT_VERSION=2.45\n" in env_text
    assert "QAIHM_TEST_GENIEX_VERSION=0.3.7\n" in env_text
    assert f"QAIHM_TEST_CHECKPOINT={_CHECKPOINT_KEY}\n" in env_text


def test_cli_skips_env_by_default(tmp_path: Path) -> None:
    env_file = tmp_path / "gh_env"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45",
            "--github-env",
            str(env_file),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert not env_file.exists()


def test_cli_reports_error_and_exits_nonzero() -> None:
    res = subprocess.run(
        [sys.executable, str(_SCRIPT), "--raw", "unknown_key=x"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "Unknown option key" in res.stderr


def test_cli_rejects_newline_injection(tmp_path: Path) -> None:
    out = tmp_path / "gh_output"
    res = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--raw",
            "qairt_version=2.45\nqairt_version=eviltag",
            "--github-output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 1
    assert "newlines" in res.stderr
    assert not out.exists() or "eviltag" not in out.read_text()
