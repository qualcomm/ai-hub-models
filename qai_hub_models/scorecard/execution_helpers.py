# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import qai_hub as hub

from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard import ScorecardCompilePath, ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.envvars import (
    EnabledPrecisionsEnvvar,
    IgnoreKnownFailuresEnvvar,
    SpecialPrecisionSetting,
)
from qai_hub_models.scorecard.results.scorecard_job import ScorecardPathOrNoneTypeVar
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

try:
    from qai_hub_models.scorecard.internal.list_models import (
        get_bench_pytorch_w8a8_models,
        get_bench_pytorch_w8a16_models,
    )

except ImportError:

    def get_bench_pytorch_w8a8_models() -> list[str]:  # type: ignore[misc]
        return []

    def get_bench_pytorch_w8a16_models() -> list[str]:  # type: ignore[misc]
        return []


def for_each_scorecard_path_and_device(
    path_type: type[ScorecardPathOrNoneTypeVar],
    callback: Callable[[Precision, ScorecardPathOrNoneTypeVar, ScorecardDevice], None],
    precisions: list[Precision] | None = None,
    include_paths: list[ScorecardPathOrNoneTypeVar] | None = None,
    include_devices: list[ScorecardDevice] | None = None,
    exclude_paths: list[ScorecardPathOrNoneTypeVar] | None = None,
    exclude_devices: list[ScorecardDevice] | None = None,
    include_mirror_devices: bool = False,
) -> None:
    if precisions is None:
        precisions = [Precision.float]
    for precision in precisions:
        if path_type is not type(None) and path_type is not None:
            all_paths = path_type.all_paths(enabled=True, supports_precision=precision)  # type: ignore[attr-defined]
        else:
            all_paths = [None]

        for path in all_paths:
            if include_paths is not None and path not in include_paths:
                continue
            if exclude_paths is not None and path in exclude_paths:
                continue

            for device in ScorecardDevice.all_devices(
                enabled=True,
                npu_supports_precision=precision,
                supports_compile_path=(
                    path if isinstance(path, ScorecardCompilePath) else None
                ),
                supports_profile_path=(
                    path if isinstance(path, ScorecardProfilePath) else None
                ),
                is_mirror=None if include_mirror_devices else False,
            ):
                if include_devices is not None and device not in include_devices:
                    continue
                if exclude_devices is not None and device in exclude_devices:
                    continue

                callback(precision, path, device)


def get_enabled_test_precisions() -> tuple[
    SpecialPrecisionSetting | None, list[Precision]
]:
    """
    Determine what precisions are enabled based on the test environment.

    Returns
    -------
    special_precision_setting : SpecialPrecisionSetting | None
        Any special precision setting with which the run was configured.
    extra_enabled_precisions : list[Precision]
        Precisions that should be enabled beyond the defaults, if a model supports quantize job.
    """
    precisions_set = EnabledPrecisionsEnvvar.get()
    precisions_special_settings = [
        p for p in precisions_set if isinstance(p, SpecialPrecisionSetting)
    ]
    if len(precisions_special_settings) > 1:
        raise ValueError(
            "Multiple special settings found in precision list."
            f"Cannot set both {precisions_special_settings[0].value} and {precisions_special_settings[1].value}."
        )

    return (
        precisions_special_settings[0] if precisions_special_settings else None,
        [Precision.parse(p.strip()) for p in precisions_set if isinstance(p, str)],
    )


def get_quantized_bench_models_path() -> Path:
    return (
        QAIHM_MODELS_ROOT / "scorecard" / "internal" / "pytorch_bench_models_w8a8.txt"
    )


@lru_cache(maxsize=1)
def get_quantized_bench_models() -> set[str]:
    with open(get_quantized_bench_models_path()) as f:
        return set(f.read().strip().split("\n"))


def get_model_test_precisions(
    model_id: str,
    model_supported_precisions: set[Precision],
    can_use_quantize_job: bool = True,
) -> list[Precision]:
    """
    Get the list of precisions that should be tested in this environment.

    Parameters
    ----------
    model_id
        The model ID.
    model_supported_precisions
        The set of Precisions that this model can support.
    can_use_quantize_job
        Whether a model can use quantize job.
        If true, extra precisions set in parameter `enabled_test_precisions` will be included.

    Returns
    -------
    model_test_precisions : list[Precision]
        The list of precisions to test for this model.
    """
    enabled_test_precisions = get_enabled_test_precisions()
    special_precision_setting, extra_enabled_precisions = enabled_test_precisions
    enabled_precisions: set[Precision] = set()
    if special_precision_setting in [
        SpecialPrecisionSetting.DEFAULT,
        SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT,
    ]:
        # If default precisions are enabled, always run tests with default precisions.
        enabled_precisions.update(model_supported_precisions)

    if (
        special_precision_setting == SpecialPrecisionSetting.DEFAULT_MINUS_FLOAT
        and Precision.float in enabled_precisions
    ):
        enabled_precisions.remove(Precision.float)
    if special_precision_setting == SpecialPrecisionSetting.DEFAULT_QUANTIZED:
        enabled_precisions.add(
            Precision.w8a16
            if (Precision.w8a16 in model_supported_precisions)
            else Precision.w8a8
        )
    if special_precision_setting == SpecialPrecisionSetting.BENCH:
        if Precision.float in model_supported_precisions:
            enabled_precisions.add(Precision.float)
        if (
            Precision.w8a8 in model_supported_precisions
            and model_id in get_bench_pytorch_w8a8_models()
        ):
            enabled_precisions.add(Precision.w8a8)
        if (
            Precision.w8a16 in model_supported_precisions
            and model_id in get_bench_pytorch_w8a16_models()
        ):
            enabled_precisions.add(Precision.w8a16)
    if can_use_quantize_job:
        # If quantize job is supported, this model can run tests on any desired precision.
        enabled_precisions.update(extra_enabled_precisions)
    else:
        # If quantize job is not supported, we can still run enabled precisions that happen to be in the model's supported precisions list.
        enabled_precisions.update(
            set(model_supported_precisions).intersection(extra_enabled_precisions)
        )

    return list(enabled_precisions)


ScorecardPathTypeVar = TypeVar(
    "ScorecardPathTypeVar", ScorecardCompilePath, ScorecardProfilePath, None
)


def get_model_test_parameterizations(
    model_id: str,
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    path_type: type[ScorecardPathTypeVar],
    can_use_quantize_job: bool = True,
    devices: list[ScorecardDevice] | None = None,
    include_unsupported_paths: bool | None = None,
    requires_aot_prepare: bool = False,
    only_include_genai_paths: bool = False,
) -> list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]]:
    """
    Get a list of parameterizations for testing a model.

    Parameters
    ----------
    model_id
        model_id of the relevant model.
    supported_paths
        The list of (Precision, Runtime) pairs that this model can support.
    timeout_paths
        The list of (Precision, Runtime) pairs that time out. These will never run regardless of scorecard settings.
    path_type
        The type of scorecard path to return (Compile or Profile)
    can_use_quantize_job
        Whether this model can be quantized with QuantizeJob.
        If true, extra precisions set in parameter `enabled_test_precisions` will be included.
    devices
        The list of devices to include. If None, all enabled devices are included.
    include_unsupported_paths
        If true, all enabled paths will be included, instead of the ones compatible with
        parameter supported_paths.
    requires_aot_prepare
        If True, only AOT (compilation to context binary on Hub) paths are included.
        If False, only JIT (compilation to context binary on-device) paths are included.
    only_include_genai_paths
        If True, only GenAI paths are included.

    Returns
    -------
    enabled_test_paths : list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]]
        A list of (Precision, ScorecardPath, Device) pairs to test.
        Each (Precision, ScorecardPath, Device) pair will:
        * Only include items enabled in this environment via env variables
            (each arg is a comma separated list)
            - QAIHM_TEST_PRECISIONS (enabled precisions, default is DEFAULT (only include precisions supported by each model)
            - QAIHM_TEST_PATHS (enabled runtimes, default is ALL)
            - QAIHM_TEST_DEVICES (enabled devices, default is ALL)
        * Be compatible with each other:
            - The ScorecardPath will be compatible with the Precision.
            - The ScorecardPath will be applicable to the Device.
            - The Precision can run on the Device's NPU.
        * Be compatible with the model:
            - See parameter documentation for details.
    """
    ret: list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]] = []
    if include_unsupported_paths is None:
        include_unsupported_paths = IgnoreKnownFailuresEnvvar.get()

    # Get the precisions enabled for this model in this test environment.
    model_supported_precisions = set(supported_paths.keys())
    test_precisions = get_model_test_precisions(
        model_id, model_supported_precisions, can_use_quantize_job
    )

    # For each enabled test precision...
    for precision in test_precisions:
        # Get all enabled paths that support this precision
        path_list = path_type.all_paths(  # type: ignore[attr-defined]
            enabled=True,
            supports_precision=precision,
            include_genai_paths=only_include_genai_paths,
        )

        if only_include_genai_paths:
            # Only include GenAI paths.
            path_list = [
                path for path in path_list if path.runtime.is_exclusively_for_genai
            ]
        elif requires_aot_prepare:
            # Only include AOT compiled paths.
            path_list = [path for path in path_list if path.runtime.is_aot_compiled]
        else:
            # Only include JIT compiled paths.
            path_list = [path for path in path_list if not path.runtime.is_aot_compiled]

        # Filter the list to include only paths that are supported by this model.
        if not include_unsupported_paths:
            supported_runtime_list = supported_paths.get(precision, [])
            path_list = [
                path for path in path_list if path.runtime in supported_runtime_list
            ]

        # Filter out timeout paths
        timeout_precision_paths = timeout_paths.get(precision, [])
        path_list = [
            path for path in path_list if path.runtime not in timeout_precision_paths
        ]

        # For each test path...

        for sc_path in path_list:
            for device in devices or ScorecardDevice.all_devices(is_mirror=False):
                if not device.enabled or not device.npu_supports_precision(precision):
                    continue
                if (
                    sc_path not in device.compile_paths
                    and sc_path not in device.profile_paths
                ):
                    continue

                ret.append((precision, sc_path, device))
    return ret


def pytest_device_idfn(val: object) -> str | None:
    """
    Pytest generates test titles based on the parameterization of each test.
    This title can both be used as a filter during test selection and is
    printed to console to identify the test. An example title:
    qai_hub_models/models/whisper_base/test_generated.py::test_compile[qnn-cs_8_gen_3]

    Several unit tests parameterize based on device objects. Pytest is not capable by default
    of understanding what string identifier to use for a device object, so it will print
    `device##` in the title of those tests rather than the actual device name.

    Passing this function to the @pytest.mark.parametrize hook (ids=pytest_device_idfn) will
    instruct pytest to print the name of the device in the test title instead.

    See https://docs.pytest.org/en/stable/example/parametrize.html#different-options-for-test-ids
    """
    if isinstance(val, ScorecardDevice):
        return val.name
    if isinstance(val, Precision):
        return str(val)
    return None


def needs_pre_quantize_compile(
    model_id: str,
    supported_paths: dict[Precision, list[TargetRuntime]],
) -> bool:
    return len(get_quantize_parameterized_pytest_config(model_id, supported_paths)) > 0


def get_quantize_parameterized_pytest_config(
    model_id: str,
    supported_paths: dict[Precision, list[TargetRuntime]],
) -> list[Precision]:
    precisions = get_model_test_precisions(
        model_id,
        set(supported_paths.keys()),
        can_use_quantize_job=True,
    )
    return [x for x in precisions if x.has_quantized_activations]


def get_compile_parameterized_pytest_config(
    model_id: str,
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    requires_aot_prepare: bool = False,
    only_include_genai_paths: bool = False,
) -> list[tuple[Precision, ScorecardCompilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, compile path) pairs."""
    return get_model_test_parameterizations(
        model_id,
        supported_paths,
        timeout_paths,
        ScorecardCompilePath,
        can_use_quantize_job,
        requires_aot_prepare=requires_aot_prepare,
        only_include_genai_paths=only_include_genai_paths,
    )


def get_profile_parameterized_pytest_config(
    model_id: str,
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    requires_aot_prepare: bool = False,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, profile path) pairs."""
    return get_model_test_parameterizations(
        model_id,
        supported_paths,
        timeout_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        requires_aot_prepare=requires_aot_prepare,
    )


def get_export_parameterized_pytest_config(
    model_id: str,
    device: ScorecardDevice,
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    requires_aot_prepare: bool = False,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, profile path) pairs."""
    return get_model_test_parameterizations(
        model_id,
        supported_paths,
        timeout_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        ScorecardDevice.all_devices(enabled=True, include_universal=False)
        if requires_aot_prepare
        else [device],
        requires_aot_prepare=requires_aot_prepare,
    )


def get_evaluation_parameterized_pytest_config(
    model_id: str,
    device: ScorecardDevice,
    supported_paths: dict[Precision, list[TargetRuntime]],
    timeout_paths: dict[Precision, list[TargetRuntime]],
    can_use_quantize_job: bool = True,
    requires_aot_prepare: bool = False,
) -> list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    """Get a pytest parameterization list of all enabled (device, profile path) pairs."""
    enabled_devices = ScorecardDevice.all_devices(enabled=True, include_universal=False)
    if device not in enabled_devices:
        if len(enabled_devices) == 1:
            device = enabled_devices[0]
        else:
            # "When running numerical evaluation, must specify exactly one device or have {device} as part of the device list."
            return []

    return get_model_test_parameterizations(
        model_id,
        supported_paths,
        timeout_paths,
        ScorecardProfilePath,
        can_use_quantize_job,
        [device],
        requires_aot_prepare=requires_aot_prepare,
    )


def get_async_job_cache_name(
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    model_id: str,
    device: ScorecardDevice,
    precision: Precision = Precision.float,
    component: str | None = None,
) -> str:
    """
    Get the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

    Parameters
    ----------
    path
        Applicable scorecard path
    model_id
        The ID of the QAIHM model being tested
    device
        The targeted device
    precision
        The precision in which this model is running
    component
        The name of the model component being tested, if applicable

    Returns
    -------
    cache_key : str
        The cache key for this job.
    """
    return (
        f"{model_id}"
        + ("_" + str(precision) if precision != Precision.float else "")
        + ("_" + path.name if path else "")
        + ("-" + device.name if device != cs_universal else "")
        + ("_" + component if component else "")
    )


def _on_staging() -> bool:
    """
    Returns whether the workbench client is pointing to staging.
    Can be sometimes useful to diverge logic between PR CI (prod) and nightly (staging).
    """
    client = hub.client.Client()
    client.get_devices()
    client_config = client._config
    assert client_config is not None
    return "staging" in client_config.api_url


@dataclass
class ClientState:
    on_staging: bool


class ClientStateSingleton:
    _instance: ClientState | None = None

    def __init__(self) -> None:
        if self._instance is None:
            self._instance = ClientState(on_staging=_on_staging())

    def on_staging(self) -> bool:
        assert self._instance is not None
        return self._instance.on_staging
