# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Generator
from unittest.mock import patch

import pytest

from qai_hub_models_cli.cli import main
from qai_hub_models_cli.proto.numerics_pb2 import ModelNumerics
from qai_hub_models_cli.proto.perf_pb2 import ModelPerf
from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    PlatformInfo,
)
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.range_pb2 import DoubleRange, IntRange
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions
from qai_hub_models_cli.proto_helpers.numerics import (
    filter_numerics,
    format_numerics_table,
)
from qai_hub_models_cli.proto_helpers.perf import (
    filter_perf,
    format_perf_table,
)


def _devices(perf: ModelPerf) -> set[str]:
    """Devices present in a (filtered) perf proto."""
    return {r.device for r in perf.performance_metrics}


def _platform() -> PlatformInfo:
    return PlatformInfo(
        devices=[
            DeviceInfo(
                name="Samsung Galaxy S24", chipset="qualcomm-snapdragon-8-gen-3"
            ),
            DeviceInfo(
                name="Samsung Galaxy S23", chipset="qualcomm-snapdragon-8-gen-2"
            ),
        ],
        chipsets=[
            ChipsetInfo(
                name="qualcomm-snapdragon-8-gen-3",
                marketing_name="Snapdragon 8 Gen 3",
                aliases=["sd8gen3"],
            ),
            ChipsetInfo(
                name="qualcomm-snapdragon-8-gen-2",
                marketing_name="Snapdragon 8 Gen 2",
            ),
        ],
    )


def _perf() -> ModelPerf:
    return ModelPerf(
        model_id="mobilenet_v2",
        performance_metrics=[
            ModelPerf.PerformanceDetails(
                precision=Precision.PRECISION_FLOAT,
                device="Samsung Galaxy S24",
                runtime=Runtime.RUNTIME_TFLITE,
                tool_versions=ToolVersions(tflite="2.16"),
                metrics=ModelPerf.PerfMetrics(
                    inference_time_milliseconds=2.5,
                    estimated_peak_memory_range_mb=IntRange(min=10, max=20),
                    primary_compute_unit="NPU",
                ),
            ),
            ModelPerf.PerformanceDetails(
                precision=Precision.PRECISION_W8A8,
                device="Samsung Galaxy S23",
                runtime=Runtime.RUNTIME_QNN_DLC,
                tool_versions=ToolVersions(qairt="2.31"),
                metrics=ModelPerf.PerfMetrics(
                    inference_time_milliseconds=1.2,
                    primary_compute_unit="NPU",
                ),
            ),
        ],
    )


def _llm_perf() -> ModelPerf:
    return ModelPerf(
        model_id="llama",
        performance_metrics=[
            ModelPerf.PerformanceDetails(
                precision=Precision.PRECISION_W4A16,
                device="Samsung Galaxy S24",
                runtime=Runtime.RUNTIME_GENIE,
                tool_versions=ToolVersions(qairt="2.31"),
                llm_metrics=[
                    # desired_compute_unit unset -> rendered as "npu".
                    ModelPerf.LLMPerfMetrics(
                        context_length=4096,
                        tokens_per_second=20.5,
                        time_to_first_token_range_milliseconds=DoubleRange(
                            min=100, max=150
                        ),
                    ),
                    ModelPerf.LLMPerfMetrics(
                        context_length=4096,
                        tokens_per_second=8.0,
                        time_to_first_token_range_milliseconds=DoubleRange(
                            min=200, max=300
                        ),
                        desired_compute_unit="cpu",
                    ),
                ],
            ),
        ],
    )


def _numerics() -> ModelNumerics:
    return ModelNumerics(
        model_id="mobilenet_v2",
        metrics=[
            ModelNumerics.NumericsMetric(
                dataset_name="imagenet",
                metric_name="Top-1 Accuracy",
                metric_unit="%",
                partial_torch_metric=71.8,
                device_metrics=[
                    ModelNumerics.NumericsMetric.DeviceNumericsMetrics(
                        device="Samsung Galaxy S24",
                        precision=Precision.PRECISION_FLOAT,
                        runtime=Runtime.RUNTIME_TFLITE,
                        partial_metric=71.5,
                        # Cross-referenced from the matching perf record.
                        tool_versions=ToolVersions(tflite="2.16"),
                    ),
                    ModelNumerics.NumericsMetric.DeviceNumericsMetrics(
                        device="Samsung Galaxy S23",
                        precision=Precision.PRECISION_W8A8,
                        runtime=Runtime.RUNTIME_QNN_DLC,
                        partial_metric=70.9,
                        tool_versions=ToolVersions(qairt="2.31"),
                    ),
                ],
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _mocks() -> Generator[None]:
    with (
        patch("qai_hub_models_cli.cli._check_version_match"),
        patch("qai_hub_models_cli.cli.get_platform", return_value=_platform()),
        patch("qai_hub_models_cli.cli.get_model_perf", return_value=_perf()),
        patch("qai_hub_models_cli.cli.get_model_numerics", return_value=_numerics()),
    ):
        yield


def test_perf_filter_runtime_and_chipset() -> None:
    # runtime + chipset (by alias) narrows to the single tflite/S24 record.
    filtered = filter_perf(_perf(), _platform(), runtime="tflite", chipset="sd8gen3")
    assert _devices(filtered) == {"Samsung Galaxy S24"}


def test_perf_filter_multiple_runtimes() -> None:
    # Multiple runtimes match as an OR; both records are kept.
    filtered = filter_perf(_perf(), _platform(), runtime=["tflite", "qnn_dlc"])
    assert _devices(filtered) == {"Samsung Galaxy S24", "Samsung Galaxy S23"}


def test_perf_invalid_device_errors(capsys: pytest.CaptureFixture[str]) -> None:
    # An unknown device errors with the helpful 'devices' pointer (like fetch).
    with pytest.raises(SystemExit):
        main(["perf", "mobilenet_v2", "-d", "Samsung Galaxy S24", "Bogus"])
    assert "Unknown device 'Bogus'" in capsys.readouterr().out


def test_perf_llm_table_includes_sdk_versions() -> None:
    # LLM perf table renders an SDK Versions column populated from tool_versions.
    table = format_perf_table(_llm_perf())
    assert "Tokens/sec" in table
    assert "SDK Versions" in table
    assert "QAIRT 2.31" in table


def test_perf_llm_table_includes_compute_unit() -> None:
    # The LLM table shows a Compute Unit column (values uppercased); an unset
    # value renders as NPU, and an explicit one (e.g. cpu) shows as CPU.
    table = format_perf_table(_llm_perf())
    assert "Compute Unit" in table
    assert "NPU" in table
    assert "CPU" in table


def test_numerics_filter() -> None:
    filtered = filter_numerics(_numerics(), _platform(), precision="float")
    # Only the float (S24/tflite) device result survives.
    results = [dm for m in filtered.metrics for dm in m.device_metrics]
    assert len(results) == 1
    assert results[0].device == "Samsung Galaxy S24"


def test_numerics_sdk_version_filter() -> None:
    # SDK versions are cross-referenced onto numerics at build time, so numerics
    # filters on its own tool_versions (no perf needed).
    filtered = filter_numerics(_numerics(), _platform(), sdk_versions={"qairt": "2.31"})
    results = [dm for m in filtered.metrics for dm in m.device_metrics]
    assert [dm.device for dm in results] == ["Samsung Galaxy S23"]


def test_numerics_table_includes_sdk_versions() -> None:
    table = format_numerics_table(_numerics())
    assert "SDK Versions" in table
    assert "TFLite 2.16" in table and "QAIRT 2.31" in table


def test_footer(capsys: pytest.CaptureFixture[str]) -> None:
    main(["perf", "mobilenet_v2", "-r", "qnn_dlc", "-v", "0.22.0"])
    output = capsys.readouterr().out
    # Example command pins the version and echoes known filters, placeholders
    # for the rest.
    assert "qai-hub-models perf mobilenet_v2 -v 0.22.0 -r 'qnn_dlc'" in output
    assert "-p <precision>" in output
    # Cross-links to the sibling command, version pinned.
    assert "qai-hub-models numerics mobilenet_v2 -v 0.22.0" in output


def test_no_footer_when_empty(capsys: pytest.CaptureFixture[str]) -> None:
    main(["perf", "mobilenet_v2", "-r", "onnx"])
    output = capsys.readouterr().out
    assert "No performance metrics match" in output
    assert "Filter these results" not in output


def _component_perf() -> ModelPerf:
    def rec(component: str) -> ModelPerf.PerformanceDetails:
        return ModelPerf.PerformanceDetails(
            precision=Precision.PRECISION_W4A16,
            device="Samsung Galaxy S24",
            runtime=Runtime.RUNTIME_GENIE,
            component=component,
            metrics=ModelPerf.PerfMetrics(inference_time_milliseconds=1.0),
        )

    return ModelPerf(performance_metrics=[rec("encoder"), rec("decoder"), rec("tok")])


def test_perf_component_filter() -> None:
    filtered = filter_perf(
        _component_perf(), _platform(), components=["encoder", "decoder"]
    )
    components = {r.component for r in filtered.performance_metrics}
    assert components == {"encoder", "decoder"}


def test_perf_invalid_component_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("qai_hub_models_cli.cli.get_model_perf", return_value=_component_perf()),
        pytest.raises(SystemExit),
    ):
        main(["perf", "llama", "--component", "bogus"])
    # The error lists the valid components.
    assert "Unknown component(s): bogus" in capsys.readouterr().out
