# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from qai_hub_models_cli.proto_helpers._common import safe_load_yaml

from qai_hub_models.scorecard.artifacts import (
    RUNTIME_STAGE_ACCURACY,
    RUNTIME_STAGE_EXPORT_TEST,
    RUNTIME_STAGE_JOB_SUBMISSION,
)
from qai_hub_models.scripts.extract_model_runtimes import (
    _download_scorecard_artifact,
    build_runtime_estimates,
    parse_junit_per_model_seconds,
    write_runtime_estimates_yaml,
)


def _write_xml(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).strip())


def test_parse_junit_attributes_time_by_classname(tmp_path: Path) -> None:
    xml = tmp_path / "submission.xml"
    _write_xml(
        xml,
        """
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites>
          <testsuite name="pytest" tests="3">
            <testcase classname="qai_hub_models.models.yolov7.test_export" name="test_compile" time="100.5"/>
            <testcase classname="qai_hub_models.models.yolov7.test_export" name="test_profile" time="50.0"/>
            <testcase classname="qai_hub_models.models.whisper_base.test_export" name="test_compile" time="200.0"/>
          </testsuite>
        </testsuites>
        """,
    )
    totals = parse_junit_per_model_seconds(xml)
    assert totals == {"yolov7": 150.5, "whisper_base": 200.0}


def test_parse_junit_skips_non_model_testcases(tmp_path: Path) -> None:
    xml = tmp_path / "x.xml"
    _write_xml(
        xml,
        """
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites>
          <testsuite name="pytest" tests="2">
            <testcase classname="qai_hub_models.scorecard.test_things" name="test_misc" time="42.0"/>
            <testcase classname="qai_hub_models.models.yolov7.test_export" name="test_compile" time="10.0"/>
          </testsuite>
        </testsuites>
        """,
    )
    totals = parse_junit_per_model_seconds(xml)
    assert totals == {"yolov7": 10.0}


def test_parse_junit_handles_dotted_name_attr(tmp_path: Path) -> None:
    xml = tmp_path / "x.xml"
    _write_xml(
        xml,
        """
        <?xml version="1.0" encoding="utf-8"?>
        <testsuites>
          <testsuite>
            <testcase classname="" name="qai_hub_models.models.yolov7.test_export.test_compile" time="7.5"/>
          </testsuite>
        </testsuites>
        """,
    )
    assert parse_junit_per_model_seconds(xml) == {"yolov7": 7.5}


def test_parse_junit_missing_file_returns_empty(tmp_path: Path) -> None:
    assert parse_junit_per_model_seconds(tmp_path / "nope.xml") == {}


def test_parse_junit_malformed_xml_returns_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Truncated/corrupt XML should not crash the runtime-refresh step."""
    xml = tmp_path / "broken.xml"
    xml.write_text("<testsuites><testsuite>truncated")
    assert parse_junit_per_model_seconds(xml) == {}
    err = capsys.readouterr().err
    assert "could not parse" in err


@pytest.mark.parametrize(
    "bad_action_id",
    ["--name", "-1", "12;rm -rf /", "abc", "12 34", ""],
)
def test_download_scorecard_artifact_rejects_non_numeric_action_id(
    tmp_path: Path, bad_action_id: str
) -> None:
    """Anything that isn't a pure run-id digit string is rejected before exec."""
    with pytest.raises(ValueError, match="action_id must be numeric"):
        _download_scorecard_artifact(bad_action_id, repo=None, dest=tmp_path)


def test_download_scorecard_artifact_rejects_bad_repo(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo must be"):
        _download_scorecard_artifact("12345", repo="evil; rm -rf /", dest=tmp_path)


def test_build_runtime_estimates_merges_three_stages(tmp_path: Path) -> None:
    sub = tmp_path / "sub.xml"
    exp = tmp_path / "exp.xml"
    acc = tmp_path / "acc.xml"
    _write_xml(
        sub,
        """
        <?xml version="1.0"?>
        <testsuites><testsuite>
          <testcase classname="qai_hub_models.models.yolov7.test_export" name="t" time="100"/>
          <testcase classname="qai_hub_models.models.whisper.test_export" name="t" time="300"/>
        </testsuite></testsuites>
        """,
    )
    _write_xml(
        exp,
        """
        <?xml version="1.0"?>
        <testsuites><testsuite>
          <testcase classname="qai_hub_models.models.yolov7.test_export_e2e" name="t" time="20"/>
        </testsuite></testsuites>
        """,
    )
    _write_xml(
        acc,
        """
        <?xml version="1.0"?>
        <testsuites><testsuite>
          <testcase classname="qai_hub_models.models.whisper.test_accuracy" name="t" time="40"/>
        </testsuite></testsuites>
        """,
    )

    est = build_runtime_estimates(sub, exp, acc)
    # Alphabetical key order is required.
    assert list(est.keys()) == ["whisper", "yolov7"]
    assert est["yolov7"] == {
        RUNTIME_STAGE_JOB_SUBMISSION: 100.0,
        RUNTIME_STAGE_EXPORT_TEST: 20.0,
    }
    assert est["whisper"] == {
        RUNTIME_STAGE_JOB_SUBMISSION: 300.0,
        RUNTIME_STAGE_ACCURACY: 40.0,
    }


def test_write_runtime_estimates_yaml_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "estimates.yaml"
    estimates = {
        "model_a": {RUNTIME_STAGE_JOB_SUBMISSION: 100.0, RUNTIME_STAGE_ACCURACY: 50.0},
        "model_b": {RUNTIME_STAGE_EXPORT_TEST: 25.0},
    }
    write_runtime_estimates_yaml(out, estimates, source_action_id="42")
    with open(out) as f:
        loaded = safe_load_yaml(f)
    assert loaded["source_action_id"] == "42"
    assert loaded["models"] == estimates
