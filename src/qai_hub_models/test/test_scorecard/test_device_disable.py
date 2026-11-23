# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.model_disable_reasons import ModelDisableReasonsMapping
from qai_hub_models.models.templates.llm.perf_collection import (
    get_llm_perf_parametrization,
)
from qai_hub_models.scorecard import ScorecardInferencePath
from qai_hub_models.scorecard.device import LLM_COMPILE_DEVICES, ScorecardDevice
from qai_hub_models.scorecard.envvars import IgnoreKnownFailuresEnvvar
from qai_hub_models.scorecard.execution_helpers import (
    get_compile_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
)
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.code_gen import (
    remove_asset_failures,
    remove_numerics_failures,
)
from qai_hub_models.utils.set_env import set_temp_env

MODEL_ID = "mobilenet_v2"
ISSUE = "https://github.com/qcom-ai-hub/tetracode/issues/1234"
DISABLED_DEVICE = "cs_8_elite"
DISABLED_HUB_DEVICE = ScorecardDevice.get(DISABLED_DEVICE).device_name
OTHER_HUB_DEVICE = ScorecardDevice.get("cs_8_gen_3").device_name
DISABLED_CHIPSET = "qualcomm-snapdragon-8-elite-for-galaxy"
KEPT_CHIPSET = "qualcomm-snapdragon-x-elite"


def _manifest_with_device_disable(timeout: bool = False) -> QAIHMModelManifest:
    manifest = QAIHMModelManifest.from_model(MODEL_ID)
    reasons = manifest.disabled_paths.get_disable_reasons(
        Precision.float, TargetRuntime.TFLITE
    )
    reasons.issue = ISSUE
    reasons.causes_timeout = timeout
    reasons.disable_devices = [DISABLED_HUB_DEVICE]
    return manifest


def test_is_supported_respects_device() -> None:
    manifest = _manifest_with_device_disable()
    assert not manifest.is_supported(
        Precision.float, TargetRuntime.TFLITE, device=DISABLED_HUB_DEVICE
    )
    assert manifest.is_supported(
        Precision.float, TargetRuntime.TFLITE, device=OTHER_HUB_DEVICE
    )
    assert manifest.is_supported(Precision.float, TargetRuntime.TFLITE)


def test_device_issue_ignored_when_user_failures_excluded() -> None:
    manifest = _manifest_with_device_disable()
    assert manifest.is_supported(
        Precision.float,
        TargetRuntime.TFLITE,
        consider_user_defined_failures=False,
        device=DISABLED_HUB_DEVICE,
    )


def test_device_timeout_applies_even_when_user_failures_excluded() -> None:
    manifest = _manifest_with_device_disable(timeout=True)
    assert not manifest.is_supported(
        Precision.float,
        TargetRuntime.TFLITE,
        consider_user_defined_failures=False,
        device=DISABLED_HUB_DEVICE,
    )


def _profile_devices(manifest: QAIHMModelManifest) -> set[str]:
    paths = manifest.get_supported_paths_for_testing(only_include_passing=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(QAIHMModelManifest, "from_model", lambda _id: manifest)
        configs = get_profile_parameterized_pytest_config(MODEL_ID, paths, paths)
    return {
        device.name
        for precision, path, device in configs
        if precision == Precision.float
        and isinstance(path, ScorecardInferencePath)
        and path.runtime == TargetRuntime.TFLITE
    }


def test_parameterization_skips_disabled_device() -> None:
    with set_temp_env({IgnoreKnownFailuresEnvvar.VARNAME: "0"}):
        baseline = _profile_devices(QAIHMModelManifest.from_model(MODEL_ID))
        disabled = _profile_devices(_manifest_with_device_disable())
    assert DISABLED_DEVICE in baseline
    assert DISABLED_DEVICE not in disabled
    assert disabled == baseline - {DISABLED_DEVICE}


def test_parameterization_keeps_device_when_ignoring_known_failures() -> None:
    with set_temp_env({IgnoreKnownFailuresEnvvar.VARNAME: "1"}):
        assert DISABLED_DEVICE in _profile_devices(_manifest_with_device_disable())
        assert DISABLED_DEVICE not in _profile_devices(
            _manifest_with_device_disable(timeout=True)
        )


LLM_MODEL_IDS = ["qwen3_8b", "qwen3_vl_8b_instruct"]
LLM_DISABLED_DEVICES = {"cs_8_elite", "cs_8_elite_gen_5"}
LLM_PERF_ENV = {"QAIHM_TEST_DEVICES": "all", "QAIHM_LLM_MODELS": ""}


def _llm_perf_devices(model_id: str, ignore_known_failures: str) -> set[str]:
    env: dict[str, str | None] = {
        IgnoreKnownFailuresEnvvar.VARNAME: ignore_known_failures,
        **LLM_PERF_ENV,
    }
    with set_temp_env(env):
        return {d.name for _, d in get_llm_perf_parametrization(model_id)}


@pytest.mark.parametrize("model_id", LLM_MODEL_IDS)
def test_llm_compile_parameterization_skips_disabled_devices(model_id: str) -> None:
    manifest = QAIHMModelManifest.from_model(model_id)
    paths = manifest.get_supported_paths_for_testing(only_include_passing=False)
    with set_temp_env({IgnoreKnownFailuresEnvvar.VARNAME: "0"}):
        configs = get_compile_parameterized_pytest_config(
            model_id, paths, paths, is_llm=True
        )
    assert not {device.name for _, _, device in configs} & LLM_DISABLED_DEVICES


@pytest.mark.parametrize("model_id", LLM_MODEL_IDS)
def test_llm_perf_parametrization_skips_disabled_devices(model_id: str) -> None:
    assert not _llm_perf_devices(model_id, "0") & LLM_DISABLED_DEVICES


@pytest.mark.parametrize("model_id", LLM_MODEL_IDS)
def test_llm_perf_parametrization_keeps_devices_when_ignoring_known_failures(
    model_id: str,
) -> None:
    expected = {d.name for d in LLM_COMPILE_DEVICES if d.qdc_enabled}
    assert LLM_DISABLED_DEVICES & expected <= _llm_perf_devices(model_id, "1")


def test_llm_perf_parametrization_follows_geniex_qairt_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = QAIHMModelManifest.from_model("qwen3_4b")
    reasons = manifest.disabled_paths.get_disable_reasons(
        Precision.w4a16, TargetRuntime.GENIEX_QAIRT
    )
    reasons.issue = ISSUE
    reasons.disable_devices = [
        ScorecardDevice.get(name).device_name for name in sorted(LLM_DISABLED_DEVICES)
    ]
    monkeypatch.setattr(QAIHMModelManifest, "from_model", lambda _id: manifest)
    assert not _llm_perf_devices("qwen3_4b", "0") & LLM_DISABLED_DEVICES


def _qwen3_8b_scoped_reasons() -> ModelDisableReasonsMapping:
    return QAIHMModelManifest.from_model("qwen3_8b").disabled_paths


def _asset(runtime: str) -> QAIHMModelReleaseAssets.AssetDetails:
    return QAIHMModelReleaseAssets.AssetDetails(s3_key=f"x/{runtime}.zip")


def test_remove_asset_failures_keeps_assets_off_disabled_devices() -> None:
    assets = QAIHMModelReleaseAssets(
        precisions={
            Precision.w4a16: QAIHMModelReleaseAssets.PrecisionDetails(
                chipset_assets={
                    chipset: {
                        ScorecardInferencePath.GENIE: _asset("genie"),
                        ScorecardInferencePath.GENIEX_QAIRT: _asset("geniex_qairt"),
                    }
                    for chipset in [DISABLED_CHIPSET, KEPT_CHIPSET]
                }
            )
        }
    )
    out = remove_asset_failures(assets, _qwen3_8b_scoped_reasons())
    chipset_assets = out.precisions[Precision.w4a16].chipset_assets
    assert DISABLED_CHIPSET not in chipset_assets
    assert set(chipset_assets[KEPT_CHIPSET]) == {
        ScorecardInferencePath.GENIE,
        ScorecardInferencePath.GENIEX_QAIRT,
    }


def test_remove_numerics_failures_keeps_metrics_off_disabled_devices() -> None:
    details = QAIHMModelNumerics.DeviceDetails(partial_metric=80.0)
    numerics = QAIHMModelNumerics(
        metrics=[
            QAIHMModelNumerics.MetricDetails(
                dataset_name="grace2",
                dataset_link="https://example.com",
                dataset_split_description="split",
                metric_name="Grace2",
                metric_description="desc",
                metric_unit="%",
                metric_range=QAIHMModelNumerics.Range(min=0.0, max=100.0),
                num_partial_samples=100,
                partial_torch_metric=96.4,
                device_metric={
                    ScorecardDevice.get(name): {
                        Precision.w4a16: {ScorecardInferencePath.GENIEX_QAIRT: details}
                    }
                    for name in [DISABLED_DEVICE, "cs_x_elite"]
                },
            )
        ]
    )
    out = remove_numerics_failures(numerics, _qwen3_8b_scoped_reasons())
    assert {d.name for d in out.metrics[0].device_metric} == {"cs_x_elite"}


def _genie_and_geniex_assets() -> QAIHMModelReleaseAssets:
    return QAIHMModelReleaseAssets(
        precisions={
            Precision.w4a16: QAIHMModelReleaseAssets.PrecisionDetails(
                chipset_assets={
                    KEPT_CHIPSET: {
                        ScorecardInferencePath.GENIE: _asset("genie"),
                        ScorecardInferencePath.GENIEX_QAIRT: _asset("geniex_qairt"),
                    }
                }
            )
        }
    )


def test_remove_asset_failures_still_drops_runtime_wide_failures() -> None:
    reasons = ModelDisableReasonsMapping()
    reasons.get_disable_reasons(
        Precision.w4a16, TargetRuntime.GENIE
    ).scorecard_accuracy_failure = "too low"
    out = remove_asset_failures(_genie_and_geniex_assets(), reasons)
    assert set(out.precisions[Precision.w4a16].chipset_assets[KEPT_CHIPSET]) == {
        ScorecardInferencePath.GENIEX_QAIRT
    }


def test_geniex_qairt_failure_also_strips_genie() -> None:
    reasons = ModelDisableReasonsMapping()
    reasons.get_disable_reasons(
        Precision.w4a16, TargetRuntime.GENIEX_QAIRT
    ).scorecard_accuracy_failure = "too low"
    out = remove_asset_failures(_genie_and_geniex_assets(), reasons)
    assert Precision.w4a16 not in out.precisions
