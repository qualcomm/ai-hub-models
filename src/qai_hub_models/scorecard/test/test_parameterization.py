# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.envvars import (
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialPathSetting,
    SpecialPrecisionSetting,
)
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_default_quantized_precision,
    get_model_test_precisions,
    get_profile_parameterized_pytest_config,
    get_quantize_parameterized_pytest_config,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath


@pytest.fixture(autouse=True)
def set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    IgnoreKnownFailuresEnvvar.patchenv(monkeypatch, True)


def test_get_quantize_precisions(monkeypatch: pytest.MonkeyPatch) -> None:
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT, "w8a8"}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float]},
        {k: [] for k in [Precision.float]},
    )
    assert set(quantize_precisions) == {Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
    )
    assert set(quantize_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
        {k: [] for k in [Precision.float, Precision.w8a8, Precision.w8a16]},
    )
    # The first quantized precision the model lists, not a hardcoded w8a16.
    assert set(quantize_precisions) == {Precision.w8a8}

    quantize_precisions = get_quantize_parameterized_pytest_config(
        "",
        {k: [] for k in [Precision.float]},
        {k: [] for k in [Precision.float]},
    )
    assert set(quantize_precisions) == {Precision.w8a8}


def test_get_compile_precisions(monkeypatch: pytest.MonkeyPatch) -> None:
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn_context_binary"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    precision_mapping = {
        k: [TargetRuntime.QNN_CONTEXT_BINARY]
        for k in [Precision.float, Precision.w8a8, Precision.w8a16]
    }
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8, Precision.w8a16}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    precision_mapping = {
        k: [TargetRuntime.QNN_CONTEXT_BINARY] for k in [Precision.float]
    }
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT, "w8a8"}
    )
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}

    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.BENCH})
    precision_mapping = {
        k: [TargetRuntime.QNN_CONTEXT_BINARY] for k in [Precision.float, Precision.w8a8]
    }
    compile_paths = get_compile_parameterized_pytest_config(
        "", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float}

    compile_paths = get_compile_parameterized_pytest_config(
        "resnet18", precision_mapping, precision_mapping
    )
    compile_precisions = [path[0] for path in compile_paths]
    assert set(compile_precisions) == {Precision.float, Precision.w8a8}


@pytest.mark.parametrize(
    ("supported_precisions", "expected"),
    [
        # The model's first quantized precision wins, rather than a hardcoded w8a16.
        ([Precision.float, Precision.w8a8, Precision.w8a16], Precision.w8a8),
        # Order is honored, so a model that lists w8a16 first runs w8a16.
        ([Precision.float, Precision.w8a16, Precision.w8a8], Precision.w8a16),
        # Models that list a mixed variant first prefer it (eg. vit, yolov8_det).
        (
            [
                Precision.float,
                Precision.w8a8_mixed_int16,
                Precision.w8a16,
                Precision.w8a8,
            ],
            Precision.w8a8_mixed_int16,
        ),
        # GGUF precisions declare no activations dtype, so they fall through to the
        # "first non-float" rule instead of defaulting to an unproducible w8a8.
        ([Precision.q4_0], Precision.q4_0),
        ([Precision.mxfp4], Precision.mxfp4),
        # w4 has no quantized activations but w4a16 does (eg. llama_v3_2_1b_instruct).
        ([Precision.w4, Precision.w4a16], Precision.w4a16),
        ([Precision.w4], Precision.w4),
        # Collection models with mixed precision (eg. zipformer, bevdet).
        ([Precision.float, Precision.mixed], Precision.mixed),
        # Regression for w4a16-only LLMs: they have no QuantizeJob, so forcing an
        # unsupported w8a8 silently built wrong-precision bundles (tetracode#20506).
        ([Precision.w4a16], Precision.w4a16),
        # Float-only models rely on quantize job to produce w8a8.
        ([Precision.float], Precision.w8a8),
    ],
    ids=str,
)
def test_default_quantized_precision_selection(
    monkeypatch: pytest.MonkeyPatch,
    supported_precisions: list[Precision],
    expected: Precision,
) -> None:
    """default_quantized runs the model's first quantized precision, in listed order."""
    EnabledPrecisionsEnvvar.patchenv(
        monkeypatch, {SpecialPrecisionSetting.DEFAULT_QUANTIZED}
    )
    # can_use_quantize_job is irrelevant: the selected precision comes from the model's
    # own supported list, so it never depends on quantize job being available.
    for can_use_quantize_job in [True, False]:
        assert get_model_test_precisions(
            "",
            supported_precisions,
            None,
            can_use_quantize_job=can_use_quantize_job,
        ) == [expected]


@pytest.mark.parametrize(
    "model_id",
    ["vit", "face_det_lite", "llama_v3_2_1b_instruct", "zipformer", "resnet50"],
)
def test_default_quantized_matches_manifest_property(model_id: str) -> None:
    """
    The precision default_quantized runs must match the one tagged "default_quantized"
    in the results spreadsheet, which uses default_quantized_precision.
    """
    manifest = QAIHMModelManifest.from_model(model_id)
    assert (
        get_default_quantized_precision(manifest.supported_precisions)
        == manifest.default_quantized_precision
    )


def test_test_precisions_preserve_supported_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test precisions are returned in the model's listed order, not set-hash order."""
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})
    ordered = [Precision.float, Precision.w8a16, Precision.w8a8]
    assert get_model_test_precisions("", ordered, None) == ordered
    assert get_model_test_precisions("", list(reversed(ordered)), None) == list(
        reversed(ordered)
    )


# ---- Tests for should_run_path_for_model ----

JIT_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [TargetRuntime.QNN_DLC, TargetRuntime.ONNX, TargetRuntime.TFLITE],
}
AOT_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.float: [
        TargetRuntime.QNN_CONTEXT_BINARY,
        TargetRuntime.PRECOMPILED_QNN_ONNX,
    ],
}


def test_default_paths_jit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default paths for JIT models include JIT paths, not AOT."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {SpecialPathSetting.DEFAULT})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", JIT_RUNTIMES, JIT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc" in path_values
    assert "onnx" in path_values
    assert "tflite" in path_values
    assert "qnn_context_binary" not in path_values
    assert "precompiled_qnn_onnx" not in path_values


def test_default_paths_aot_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default paths for AOT models include AOT paths, not JIT."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {SpecialPathSetting.DEFAULT})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", AOT_RUNTIMES, AOT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_context_binary" in path_values
    assert "precompiled_qnn_onnx" in path_values
    assert "qnn_dlc" not in path_values
    assert "onnx" not in path_values
    assert "tflite" not in path_values


def test_engine_prefix_aot_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine prefix 'qnn' only enables paths whose runtime is in supported."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", AOT_RUNTIMES, AOT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    # qnn_context_binary runtime IS in supported runtimes
    assert "qnn_context_binary" in path_values
    # qnn_dlc runtime is NOT in supported runtimes (strict match)
    assert "qnn_dlc" not in path_values


def test_engine_prefix_jit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Engine prefix 'qnn' only enables paths whose runtime is in supported."""
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", JIT_RUNTIMES, JIT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc" in path_values
    assert "qnn_context_binary" not in path_values


def test_engine_prefix_does_not_enable_unpublished_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    An engine prefix must not select paths outside default_paths().

    qnn_dlc_gpu shares the QNN_DLC runtime, so a bare 'qnn' used to select it
    via should_run_path_for_model while the `enabled` property (which gates
    the same rule on default_paths) reported it off.
    count_device_jobs keys its tally on `enabled` and raised KeyError.
    """
    EnabledPathsEnvvar.patchenv(monkeypatch, {"qnn"})
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    profile_paths = get_profile_parameterized_pytest_config(
        "", JIT_RUNTIMES, JIT_RUNTIMES
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc" in path_values
    assert "qnn_dlc_gpu" not in path_values


@pytest.mark.parametrize(
    "requested",
    [
        {SpecialPathSetting.DEFAULT},
        {"qnn"},
        {"onnx"},
        {"tflite"},
        {"qnn_dlc_gpu"},
        {SpecialPathSetting.DEFAULT, "qnn_dlc_gpu"},
        {"qnn", "qnn_dlc_gpu"},
        {SpecialPathSetting.DEFAULT, "onnx_dml_gpu"},
    ],
)
def test_should_run_paths_are_all_enabled(
    monkeypatch: pytest.MonkeyPatch, requested: set[str | SpecialPathSetting]
) -> None:
    """
    Every path should_run_path_for_model selects must also report `enabled`.

    Callers build lookup tables keyed on `enabled` (e.g. count_device_jobs'
    job_count_by_path) and then index them with the paths parameterization
    yields, so any path in should_run but not enabled is a crash.
    """
    EnabledPathsEnvvar.patchenv(monkeypatch, requested)

    all_runtimes = {Precision.float: list(TargetRuntime)}
    enabled = {p for p in ScorecardProfilePath if p.enabled}
    should_run = {
        p
        for p in ScorecardProfilePath
        if p.should_run_path_for_model(Precision.float, all_runtimes)
    }
    assert should_run, "expected at least one path to run"
    assert not (should_run - enabled)


def test_llm_runtimes_do_not_match_qnn_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Multi-graph LLMs list only GENIE/GENIEX_QAIRT runtimes. These live on the
    GENIEX inference engine, not QNN, so QNN compile paths like
    qnn_dlc_gpu must NOT run against them (they would hit the AOT-only
    assertion in base_multi_graph_model.get_graph_hub_compile_options).
    """
    EnabledPathsEnvvar.patchenv(
        monkeypatch, {SpecialPathSetting.DEFAULT, "qnn_dlc_gpu"}
    )
    EnabledPrecisionsEnvvar.patchenv(monkeypatch, {SpecialPrecisionSetting.DEFAULT})

    llm_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.w4a16: [TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT],
    }
    profile_paths = get_profile_parameterized_pytest_config(
        "", llm_runtimes, llm_runtimes
    )
    path_values = {p[1].value for p in profile_paths}
    assert "qnn_dlc_gpu" not in path_values
    assert "qnn_dlc" not in path_values
    assert "qnn_context_binary" not in path_values
    # GENIE is excluded from the default sweep; see default_sweep_paths().
    assert "genie" not in path_values
    assert "geniex_qairt" in path_values
