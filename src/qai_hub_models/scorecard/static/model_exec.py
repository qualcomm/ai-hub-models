# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from qai_hub import (
    CompileJob,
    Dataset,
    InferenceJob,
    InputSpecs,
    Job,
    JobStatus,
    JobType,
    Model,
)
from qai_hub.client import Client
from qai_hub.hub import _global_client

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import ScorecardPathTypeVar
from qai_hub_models.scorecard.params import ScJobParams
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
)


def _print_if_not_verbose(client: Client, pstr: str) -> None:
    if not client.verbose:
        print(pstr)


def _wait_for_job(job: Job, timeout: int | None) -> JobStatus:
    """Wait for a job and return job status after the wait period."""
    status = job.get_status()
    if not status.failure and not status.success and (not timeout or timeout > 0):
        _print_if_not_verbose(
            job._owner,
            f"Waiting for a max of {timeout or 'inf'} secs for {job.__class__.__name__} {job.job_id} ({job.url})...",
        )

        try:
            return job.wait(timeout)
        except TimeoutError:
            return job.get_status()
    else:
        return status


def _get_successful_compile_job_or_failure_message(
    hub: Client,
    compile_results: CompileScorecardJobYaml,
    compile_job_timeout: int | None,
    params: ScJobParams,
    cache: dict[str, tuple[CompileJob, JobStatus]],
) -> CompileJob | str:
    compile_job_id = compile_results.mapping.get(params.compile_job_id)
    if not compile_job_id:
        return "Associated Compile Job Not Found"

    # We cache compile job objects because hub.get_job() can be surprisingly slow sometimes.
    if compile_job_id in cache:
        compile_job, compile_job_status = cache[compile_job_id]
    else:
        compile_job = cast(CompileJob, hub.get_job(compile_job_id))
        compile_job_status = _wait_for_job(compile_job, compile_job_timeout)
        cache[compile_job_id] = compile_job, compile_job_status

    if compile_job_status.success:
        return compile_job
    return f"Compile Job {compile_job.job_id} {compile_job_status.state.name} ({compile_job.url})"


def for_each_static_model_test_parameterization(
    model_id: str,
    job_type: JobType,
    scorecard_path_type: type[ScorecardPathTypeVar],
    callback: Callable[[ScJobParams], None],
    precision: Precision = Precision.float,
    valid_devices: list[ScorecardDevice] | None = None,
    valid_paths: list[ScorecardPathTypeVar] | None = None,
    verbose: bool = True,
) -> None:
    """
    Execute the callback for all valid paramaterizations in current testing environment.
    Each paramaterization corresponds to a single profile job.
    """
    if valid_devices is not None:
        if scorecard_path_type == ScorecardCompilePath:
            # Always include the universal device, so we compile for all targets.
            valid_devices_set = set(valid_devices)
            valid_devices_set.add(cs_universal)
            valid_devices = list(valid_devices_set)

        if verbose:
            invalid_devices = {
                x
                for x in ScorecardDevice.all_devices(enabled=True)
                if x not in valid_devices
            }
            print(
                f"qaihm::{job_type.display_name} | {model_id} | This model's YAML config limits which devices are valid for testing. The following devices will be skipped: {invalid_devices}"
            )

    sc_paths = valid_paths if valid_paths is not None else scorecard_path_type
    for sc_path in sc_paths:
        if not sc_path.enabled or not sc_path.supports_precision(precision):
            continue
        devices = (
            valid_devices
            if valid_devices is not None
            else ScorecardDevice.all_devices()
        )
        for device in devices:
            if not device.enabled or not device.npu_supports_precision(precision):
                continue
            if (
                isinstance(sc_path, ScorecardCompilePath)
                and sc_path not in device.compile_paths
            ):
                continue
            if (
                isinstance(sc_path, ScorecardInferencePath)
                and sc_path not in device.profile_paths
            ):
                continue
            callback(
                ScJobParams(
                    model_id,
                    sc_path,
                    precision,
                    device,
                )
            )


def get_static_model_test_parameterizations(
    model_id: str,
    job_type: JobType,
    scorecard_path_type: type[ScorecardPathTypeVar],
    precision: Precision = Precision.float,
    valid_devices: list[ScorecardDevice] | None = None,
    valid_paths: list[ScorecardPathTypeVar] | None = None,
) -> list[tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]]:
    """
    Collect all valid paramaterizations in current testing environment for a given model and job type.
    Each paramaterization corresponds to a single profile job.
    """
    parameterizations: list[
        tuple[Precision, ScorecardPathTypeVar, ScorecardDevice]
    ] = []

    def collect_parameterization(params: ScJobParams[ScorecardPathTypeVar]) -> None:
        assert params.path is not None
        assert params.device is not None
        parameterizations.append((precision, params.path, params.device))

    for_each_static_model_test_parameterization(
        model_id,
        job_type,
        scorecard_path_type,
        collect_parameterization,
        precision,
        valid_devices,
        valid_paths,
        False,
    )

    return parameterizations


def compile_model(
    model_id: str,
    hub_model: Model | str,
    hub: Client = _global_client,
    input_specs: InputSpecs | None = None,
    precision: Precision = Precision.float,
    valid_devices: list[ScorecardDevice] | None = None,
    valid_paths: list[ScorecardCompilePath] | None = None,
    output_names: list[str] | None = None,
    channel_first_inputs: list[str] | None = None,
    channel_first_outputs: list[str] | None = None,
    extra_compile_options: dict[TargetRuntime, list[str]] | None = None,
    results: CompileScorecardJobYaml | None = None,
) -> CompileScorecardJobYaml:
    if results is None:
        results = CompileScorecardJobYaml()
    if extra_compile_options is None:
        extra_compile_options = {}
    if channel_first_outputs is None:
        channel_first_outputs = []
    if channel_first_inputs is None:
        channel_first_inputs = []
    if output_names is None:
        output_names = []
    if not isinstance(hub_model, Model):
        hub_model = hub.get_model(hub_model)

    output_names_option = None
    if output_names:
        output_names_option = f"--output_names {','.join(output_names)}"

    cl_input_option = None
    if channel_first_inputs:
        cl_input_option = f"--force_channel_last_input {','.join(channel_first_inputs)}"

    cl_output_option = None
    if channel_first_outputs:
        cl_output_option = (
            f"--force_channel_last_output {','.join(channel_first_outputs)}"
        )

    def submit_compile_job(params: ScJobParams) -> None:
        job_name = f"qaihm::compile | {params}"
        assert isinstance(params.path, ScorecardCompilePath)
        assert params.device is not None

        compile_options = []

        compile_options.append(
            params.path.get_compile_options(
                include_target_runtime=True,
                include_default_qaihm_qnn_version=True,
            )
        )

        if output_names_option:
            compile_options.append(output_names_option)

        if params.path.runtime.channel_last_native_execution:
            if cl_input_option:
                compile_options.append(cl_input_option)
            if cl_output_option:
                compile_options.append(cl_output_option)

        extra_options = extra_compile_options.get(params.path.runtime, [])
        compile_options.extend(extra_options)

        job = cast(
            CompileJob,
            hub.submit_compile_job(
                hub_model,
                params.device.execution_device,
                job_name,
                input_specs,
                " ".join(compile_options),
            ),
        )
        _print_if_not_verbose(hub, f"{job_name} | Submitted: {job.job_id} | {job.url}")
        results.set_job_id(job.job_id, params)

    for_each_static_model_test_parameterization(
        model_id,
        JobType.COMPILE,
        ScorecardCompilePath,
        submit_compile_job,
        precision,
        valid_devices,
        valid_paths,
    )

    return results


def inference_model(
    model_id: str,
    hub_dataset: Dataset | str,
    hub_channel_last_dataset: Dataset | str | None,
    compile_results: CompileScorecardJobYaml,
    hub: Client = _global_client,
    precision: Precision = Precision.float,
    valid_devices: list[ScorecardDevice] | None = None,
    valid_paths: list[ScorecardInferencePath] | None = None,
    results: InferenceScorecardJobYaml | None = None,
    compile_job_timeout: int | None = None,
) -> InferenceScorecardJobYaml:
    """
    Submit one inference job per (device, path) with the real validation
    dataset attached. Profiling is folded into every inference job.
    """
    if results is None:
        results = InferenceScorecardJobYaml()
    if not isinstance(hub_dataset, Dataset):
        hub_dataset = hub.get_dataset(hub_dataset)
    if hub_channel_last_dataset and not isinstance(hub_channel_last_dataset, Dataset):
        hub_channel_last_dataset = hub.get_dataset(hub_channel_last_dataset)

    compile_job_cache: dict[str, tuple[CompileJob, JobStatus]] = {}

    def submit_inference_job(params: ScJobParams) -> None:
        assert isinstance(params.path, ScorecardInferencePath)
        assert params.device is not None
        job_name = f"qaihm::inference | {params.inference_job_id}"
        compile_job = _get_successful_compile_job_or_failure_message(
            hub,
            compile_results,
            compile_job_timeout,
            params,
            compile_job_cache,
        )
        if isinstance(compile_job, str):
            print(f"{job_name} | Skipped: {compile_job}")
            return

        dataset = hub_dataset
        if params.path.runtime.channel_last_native_execution:
            dataset = hub_channel_last_dataset or dataset

        job = cast(
            InferenceJob,
            hub.submit_inference_job(
                compile_job.get_target_model(),
                params.device.execution_device,
                dataset,
                job_name,
                params.path.get_inference_options(
                    precision, params.device, include_default_qaihm_qnn_version=True
                ),
            ),
        )
        _print_if_not_verbose(hub, f"{job_name} | Submitted: {job.job_id} | {job.url}")
        results.set_job_id(job.job_id, params)

    for_each_static_model_test_parameterization(
        model_id,
        JobType.INFERENCE,
        ScorecardInferencePath,
        submit_inference_job,
        precision,
        valid_devices,
        valid_paths,
    )

    return results
