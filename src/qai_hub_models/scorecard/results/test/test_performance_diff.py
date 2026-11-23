# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

from qai_hub_models_cli.proto import model_perf_pb2

from qai_hub_models import Precision
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard.device import ScorecardDevice, cs_8_gen_3
from qai_hub_models.scorecard.params import ScJobParams
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.perf_yaml import (
    QAIHMModelPerf,
)
from qai_hub_models.scorecard.results.performance_diff import PerformanceDiff
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    ToolVersionChange,
    ToolVersionsByPathYaml,
)

MODEL_ID = "dummy"
PREV_JOB_ID = "jgzr270o5"
JOB_ID = "jp4kr0kvg"
PREV_COMPILE_JOB_ID = "jcprev9aa"
COMPILE_JOB_ID = "jcnewa11b"
COMPONENT_ID = "dummy_component"


def _details(
    job_id: str, inference_time_milliseconds: float | None
) -> model_perf_pb2.ModelPerf.PerformanceDetails:
    details = model_perf_pb2.ModelPerf.PerformanceDetails(
        profile_job=model_perf_pb2.ModelPerf.ProfileJob(id=job_id)
    )
    if inference_time_milliseconds is not None:
        details.metrics.inference_time_milliseconds = inference_time_milliseconds
    return details


def _report(
    device: ScorecardDevice,
    times: dict[ScorecardInferencePath, float | None],
    job_id: str = JOB_ID,
) -> QAIHMModelPerf:
    perf = QAIHMModelPerf(MODEL_ID)
    for path, time_ms in times.items():
        perf.update_entry(
            QAIHMModelPerf.Key(Precision.float, COMPONENT_ID, device, path).stamp(
                _details(job_id, time_ms)
            )
        )
    return perf


def get_basic_speedup_report(
    job_id: str = JOB_ID,
    onnx_tf_inference_time: float | None = None,
    onnx_ort_qnn_inference_time: float | None = 100.0,
) -> QAIHMModelPerf:
    return _report(
        cs_8_gen_3,
        {
            ScorecardInferencePath.TFLITE: onnx_tf_inference_time,
            ScorecardInferencePath.ONNX: 5.0,
            ScorecardInferencePath.QNN_DLC: onnx_ort_qnn_inference_time,
        },
        job_id,
    )


def _compile_mapping_for(
    compile_job_id: str,
    path: ScorecardInferencePath,
    device: ScorecardDevice = cs_8_gen_3,
) -> CompileScorecardJobYaml:
    """Build a CompileScorecardJobYaml that matches what
    PerformanceDiff._compile_job_id_for() looks up.
    """
    params = ScJobParams(
        model_id=MODEL_ID,
        path=path,
        precision=Precision.float,
        device=device,
        component=COMPONENT_ID,
    )
    return CompileScorecardJobYaml({params.compile_job_id: compile_job_id})


def validate_perf_diff_is_empty(perf_diff: PerformanceDiff) -> None:
    # No difference captured
    for val in perf_diff.progressions.values():
        assert len(val) == 0
    for val in perf_diff.regressions.values():
        assert len(val) == 0
    # No new reports captured
    assert len(perf_diff.new_models) == 0
    # No missing devices found in updated report
    assert len(perf_diff.missing_devices) == 0


def test_model_inference_run_toggle() -> None:
    # Test model inference fail/pass toggle is captured
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=None,
        onnx_ort_qnn_inference_time=10.0,
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=10.0, onnx_ort_qnn_inference_time=None
    )

    perf_diff = PerformanceDiff(
        current_compile_jobs=_compile_mapping_for(
            COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
        previous_compile_jobs=_compile_mapping_for(
            PREV_COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
    )
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.progressions[float("inf")] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardInferencePath.TFLITE,
            float("-inf"),
            10.0,
            float("inf"),
            JOB_ID,
            PREV_JOB_ID,
            COMPILE_JOB_ID,
            PREV_COMPILE_JOB_ID,
        )
    ]


def test_perf_progression_basic() -> None:
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=10.0,
        onnx_ort_qnn_inference_time=5.123,
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.5, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff(
        current_compile_jobs=_compile_mapping_for(
            COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
        previous_compile_jobs=_compile_mapping_for(
            PREV_COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
    )
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.progressions[10] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardInferencePath.TFLITE,
            10.0,
            0.5,
            20.0,
            JOB_ID,
            PREV_JOB_ID,
            COMPILE_JOB_ID,
            PREV_COMPILE_JOB_ID,
        )
    ]


def test_perf_regression_basic() -> None:
    # Test regression in perf numbers
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=10.0,
        onnx_ort_qnn_inference_time=5.123,
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff(
        current_compile_jobs=_compile_mapping_for(
            COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
        previous_compile_jobs=_compile_mapping_for(
            PREV_COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
    )
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert perf_diff.regressions[2] == [
        (
            MODEL_ID,
            Precision.float,
            COMPONENT_ID,
            cs_8_gen_3,
            ScorecardInferencePath.TFLITE,
            10.0,
            20.0,
            2.0,
            JOB_ID,
            PREV_JOB_ID,
            COMPILE_JOB_ID,
            PREV_COMPILE_JOB_ID,
        ),
    ]


def test_missing_devices() -> None:
    prev_perf_metrics = get_basic_speedup_report(
        PREV_JOB_ID, onnx_tf_inference_time=1.123, onnx_ort_qnn_inference_time=5.123
    )
    new_perf_metrics = get_basic_speedup_report(
        onnx_tf_inference_time=0.372, onnx_ort_qnn_inference_time=5.123
    )

    # Add a second device to both reports, then remove cs_8_gen_3 from the new one.
    # The component still exists in both, so only the device is reported missing.
    other_device = ScorecardDevice.get("Samsung Galaxy S25")
    for report in (prev_perf_metrics, new_perf_metrics):
        report.update_entry(
            QAIHMModelPerf.Key(
                Precision.float,
                COMPONENT_ID,
                other_device,
                ScorecardInferencePath.TFLITE,
            ).stamp(_details(JOB_ID, 5.0))
        )
    new_perf_metrics.drop_entries(lambda key: key.device == cs_8_gen_3)
    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)

    assert len(perf_diff.missing_devices) == 1
    assert perf_diff.missing_devices[0] == (
        MODEL_ID,
        Precision.float,
        COMPONENT_ID,
        cs_8_gen_3,
    )


def test_empty_report() -> None:
    prev_perf_metrics = get_basic_speedup_report()
    new_perf_metrics = get_basic_speedup_report()
    new_perf_metrics.drop_entries(lambda params: True)

    perf_diff = PerformanceDiff()
    validate_perf_diff_is_empty(perf_diff)

    # Update perf summary
    perf_diff.update_summary(MODEL_ID, prev_perf_metrics, new_perf_metrics)
    assert perf_diff.empty_models == [MODEL_ID]


def test_auto_device_excluded() -> None:
    """Automotive device regressions are excluded from the diff."""
    # Use an auto device reference name from devices_and_chipsets.yaml
    auto_device = ScorecardDevice(
        name="test_auto",
        reference_device_name="SA8295P ADP",
        register=False,
    )

    # Build reports with a clear 2x regression on the auto device
    auto_report = _report(auto_device, {ScorecardInferencePath.TFLITE: 10.0})
    auto_report_regressed = _report(auto_device, {ScorecardInferencePath.TFLITE: 20.0})

    perf_diff = PerformanceDiff()
    perf_diff.update_summary(MODEL_ID, auto_report, auto_report_regressed)

    # 2x regression should be excluded because device is automotive
    for val in perf_diff.regressions.values():
        assert len(val) == 0


def test_small_regression_excluded() -> None:
    """Regressions with absolute diff <=1ms are excluded (within noise margin)."""
    # 5.0 -> 5.9ms = 0.9ms diff, 1.18x ratio (would normally be in 1.1 bucket)
    prev = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=5.0,
        onnx_ort_qnn_inference_time=5.0,
    )
    new = get_basic_speedup_report(
        onnx_tf_inference_time=5.9,
        onnx_ort_qnn_inference_time=5.0,
    )

    perf_diff = PerformanceDiff()
    perf_diff.update_summary(MODEL_ID, prev, new)

    # 0.9ms regression should be excluded even though ratio is 1.18x
    for val in perf_diff.regressions.values():
        assert len(val) == 0


def test_get_severe_regressions() -> None:
    """get_severe_regressions returns structured dicts for 2x+ regressions."""
    prev = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=10.0,
        onnx_ort_qnn_inference_time=5.123,
    )
    new = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff(
        current_compile_jobs=_compile_mapping_for(
            COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
        previous_compile_jobs=_compile_mapping_for(
            PREV_COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
    )
    perf_diff.update_summary(MODEL_ID, prev, new)

    results = perf_diff.get_severe_regressions(min_factor=2.0)
    assert len(results) == 1
    assert results[0].model_id == MODEL_ID
    assert results[0].prev_inference_time == "10.0"
    assert results[0].new_inference_time == "20.0"
    # Compile job IDs are routed into the new aliases.
    assert results[0].job_id == JOB_ID
    assert results[0].compile_job_id == COMPILE_JOB_ID
    assert results[0].previous_job_id == PREV_JOB_ID
    assert results[0].previous_compile_job_id == PREV_COMPILE_JOB_ID


def test_get_severe_regressions_compile_id_fallback_without_component() -> None:
    """Compile-job cache keyed with component=None still resolves for a
    component-named perf.yaml row.

    Production caches store single-component models with component=None (see
    ScorecardJobYaml.update_from_export_output), but perf.yaml surfaces a
    display component name for the same rows. The lookup must fall back to
    component=None so the compile-job link is populated in regression issues.
    """
    prev = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=10.0,
        onnx_ort_qnn_inference_time=5.123,
    )
    new = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    params_no_component = ScJobParams(
        model_id=MODEL_ID,
        path=ScorecardInferencePath.TFLITE,
        precision=Precision.float,
        device=cs_8_gen_3,
        component=None,
    )
    perf_diff = PerformanceDiff(
        current_compile_jobs=CompileScorecardJobYaml(
            {params_no_component.compile_job_id: COMPILE_JOB_ID}
        ),
        previous_compile_jobs=CompileScorecardJobYaml(
            {params_no_component.compile_job_id: PREV_COMPILE_JOB_ID}
        ),
    )
    perf_diff.update_summary(MODEL_ID, prev, new)

    results = perf_diff.get_severe_regressions(min_factor=2.0)
    assert len(results) == 1
    assert results[0].compile_job_id == COMPILE_JOB_ID
    assert results[0].previous_compile_job_id == PREV_COMPILE_JOB_ID


def test_get_severe_regressions_excludes_small() -> None:
    """Regressions below the min_factor threshold are excluded."""
    prev = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=10.0,
        onnx_ort_qnn_inference_time=5.123,
    )
    new = get_basic_speedup_report(
        onnx_tf_inference_time=15.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff()
    perf_diff.update_summary(MODEL_ID, prev, new)

    # 1.5x regression should not appear when min_factor=2.0
    results = perf_diff.get_severe_regressions(min_factor=2.0)
    assert len(results) == 0


def test_dump_severe_regressions_json(tmp_path: Path) -> None:
    """dump_severe_regressions_json writes valid JSON."""
    prev = get_basic_speedup_report(
        PREV_JOB_ID,
        onnx_tf_inference_time=10.0,
        onnx_ort_qnn_inference_time=5.123,
    )
    new = get_basic_speedup_report(
        onnx_tf_inference_time=20.0, onnx_ort_qnn_inference_time=5.123
    )

    perf_diff = PerformanceDiff(
        current_compile_jobs=_compile_mapping_for(
            COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
        previous_compile_jobs=_compile_mapping_for(
            PREV_COMPILE_JOB_ID, ScorecardInferencePath.TFLITE
        ),
    )
    perf_diff.update_summary(MODEL_ID, prev, new)

    json_path = str(tmp_path / "regressions.json")
    perf_diff.dump_severe_regressions_json(json_path)

    with open(json_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["Model ID"] == MODEL_ID
    # The dumped JSON uses the canonical alias names (no env suffix) — the
    # issue-body builder appends (env) at render time based on each side's
    # deployment, since 'previous' isn't always prod.
    assert data[0]["Compile Job ID"] == COMPILE_JOB_ID
    assert data[0]["Previous Compile Job ID"] == PREV_COMPILE_JOB_ID


def test_tool_versions_diff_detects_changes() -> None:
    """ToolVersionsByPathYaml.diff returns one row per changed field, skips equal fields."""
    previous = ToolVersionsByPathYaml(
        tool_versions={
            ScorecardInferencePath.TFLITE: ToolVersions(tflite="2.10.0", onnx="1.16.0"),
            ScorecardInferencePath.ONNX: ToolVersions(onnx="1.16.0"),
        }
    )
    current = ToolVersionsByPathYaml(
        tool_versions={
            ScorecardInferencePath.TFLITE: ToolVersions(tflite="2.11.0", onnx="1.16.0"),
            ScorecardInferencePath.ONNX: ToolVersions(onnx="1.17.0"),
            ScorecardInferencePath.QNN_DLC: ToolVersions(onnx="1.17.0"),
        }
    )

    rows = current.diff(previous)
    assert (
        ToolVersionChange(
            ScorecardInferencePath.TFLITE.name, "tflite", "2.10.0", "2.11.0"
        )
        in rows
    )
    assert (
        ToolVersionChange(ScorecardInferencePath.ONNX.name, "onnx", "1.16.0", "1.17.0")
        in rows
    )
    assert (
        ToolVersionChange(ScorecardInferencePath.QNN_DLC.name, "onnx", "N/A", "1.17.0")
        in rows
    )
    # Unchanged fields must not appear (TFLITE.onnx stayed 1.16.0).
    assert not any(
        r.path == ScorecardInferencePath.TFLITE.name and r.tool == "onnx" for r in rows
    )


def test_tool_versions_diff_empty_when_identical() -> None:
    yaml = ToolVersionsByPathYaml(
        tool_versions={
            ScorecardInferencePath.TFLITE: ToolVersions(tflite="2.11.0"),
        }
    )
    assert yaml.diff(yaml) == []


def test_dump_summary_includes_toolchain_section(tmp_path: Path) -> None:
    """dump_summary renders a Toolchain Version Changes section when given changes."""
    perf_diff = PerformanceDiff()
    summary_path = str(tmp_path / "summary.txt")
    toolchain_changes = [
        ToolVersionChange(
            ScorecardInferencePath.TFLITE.name, "tflite", "2.10.0", "2.11.0"
        ),
    ]
    perf_diff.dump_summary(summary_path, toolchain_changes=toolchain_changes)

    contents = Path(summary_path).read_text()
    assert "Toolchain Version Changes" in contents
    assert "tflite" in contents
    assert "2.10.0" in contents
    assert "2.11.0" in contents


def test_dump_summary_omits_toolchain_section_when_empty(tmp_path: Path) -> None:
    """No Toolchain Version Changes section when toolchain_changes is None or []."""
    perf_diff = PerformanceDiff()

    none_path = str(tmp_path / "none.txt")
    perf_diff.dump_summary(none_path, toolchain_changes=None)
    assert "Toolchain Version Changes" not in Path(none_path).read_text()

    empty_path = str(tmp_path / "empty.txt")
    perf_diff.dump_summary(empty_path, toolchain_changes=[])
    assert "Toolchain Version Changes" not in Path(empty_path).read_text()
