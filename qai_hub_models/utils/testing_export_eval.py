# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import itertools
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from typing import Any, Literal, cast
from unittest import mock

import numpy as np
import qai_hub as hub
import torch
from typing_extensions import assert_never

from qai_hub_models.configs.code_gen_yaml import QAIHMModelCodeGen
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.datasets import DATASET_NAME_MAP
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.envvars import (
    IgnoreDeviceJobCacheEnvvar,
    S3ArtifactsDirEnvvar,
)
from qai_hub_models.scorecard.results.yaml import (
    INFERENCE_YAML_BASE,
    INTERMEDIATES_DIR,
    PROFILE_YAML_BASE,
    QAIHMModelReleaseAssets,
    ScorecardAssetYaml,
    ToolVersionsByPathYaml,
)
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.evaluate import (
    DEFAULT_NUM_EVAL_SAMPLES,
    evaluate_on_dataset,
    get_torch_val_dataloader,
)
from qai_hub_models.utils.export_result import CollectionExportResult, ExportResult
from qai_hub_models.utils.hub_clients import (
    deployment_is_prod,
    get_default_hub_deployment,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel
from qai_hub_models.utils.qai_hub_helpers import assert_success_and_get_target_models
from qai_hub_models.utils.testing import (
    get_and_sync_datasets_cache_dir,
    get_hub_val_dataset,
    get_profile_job_ids_file,
    mock_get_calibration_data,
    mock_on_device_model_call,
    mock_tabulate_fn,
)
from qai_hub_models.utils.testing_async_utils import (
    CachedScorecardJobError,
    CompileJobsAreIdenticalCache,
    append_line_to_file,
    assert_success_or_cache_job,
    cache_dataset,
    callable_side_effect,
    fetch_async_test_jobs,
    get_cached_dataset_entries,
    get_compile_jobs_are_identical_cache_file,
    get_cpu_accuracy_file,
    get_dataset_ids_file,
    get_release_assets_file,
    is_hub_testing_async,
    str_with_async_test_metadata,
    write_accuracy,
)

try:
    from qai_hub_models.utils._internal.aws import (
        QAIHM_PRIVATE_S3_BUCKET,
        get_qaihm_s3,
        s3_multipart_upload,
    )
except ImportError:
    get_qaihm_s3 = None  # type: ignore[assignment]
    s3_multipart_upload = None  # type: ignore[assignment]
    QAIHM_PRIVATE_S3_BUCKET = None  # type: ignore[assignment]

ExportFunc = Callable[..., ExportResult | CollectionExportResult]
JobFunc = Callable[..., hub.Job | dict[str, hub.Job]]


def _parse_export_result(
    result: CollectionExportResult | ExportResult,
) -> dict[str | None, ExportResult]:
    """
    Converts the result of an export script (export_model) to a consistent type.

    For models WITHOUT components, returns:
        { None: ExportResult }

    For models WITH components, returns:
        {
            'component_1_name: ExportResult,
            ...
        }
    """
    if isinstance(result, ExportResult):
        # Map: <Component Name: Component>
        # Use "None" since there are no components.
        return {None: result}
    return cast(dict[str | None, ExportResult], result.components)


def _invalid_job_submission(*args: Any, **kwargs: Any) -> None:
    raise ValueError(
        "Attempted to submit a job when a cached job should have been present."
    )


def _get_sim_cpu_key(model_id: str, precision: Precision) -> str:
    return f"{model_id}_{precision}_sim"


def _get_torch_cpu_key(model_id: str) -> str:
    return f"{model_id}_torch"


def patch_hub_with_cached_jobs(
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    patch_quantization: bool = False,
    patch_compile: bool = False,
    patch_profile: bool = False,
    patch_inference: bool = False,
) -> tuple[
    mock._patch,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
    mock._patch | nullcontext,
]:
    """
    Many tests use the export scripts to submit jobs.
    However, there is no path to break the export script into pieces; eg.
        * compile in one test
        * profile in another test
        * etc.
    We could modify the export script parameters to be more expressive, but this
    would come at the cost of readability.

    Instead, we "mock" various hub APIs to return "cached" jobs from previous tests.
    This allows us to test various parts of the export script "asyncronously" (without each test neeing to wait for Hub).

    This function:
        1. Gathers previous cached jobs
        2. Mocks several hub APIs (eg. submit_profile_job) to return those jobs instead of creating new ones.

    NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    model_id
        Model ID.
    precision
        Model precision.
    path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    patch_quantization
        Whether to patch previously cached quantization jobs. Default is False.
    patch_compile
        Whether to patch previously cached compile jobs. Default is False.
    patch_profile
        Whether to patch previously cached profile jobs. Default is False.
    patch_inference
        Whether to patch previously cached inference jobs. Default is False.

    Returns
    -------
    device_patch : mock._patch
        Patch for device selection.
    calibration_data_patch : mock._patch | nullcontext
        Patch for calibration data retrieval.
    quantize_job_patch : mock._patch | nullcontext
        Patch for quantization jobs.
    compile_job_patch : mock._patch | nullcontext
        Patch for compilation jobs.
    profile_job_patch : mock._patch | nullcontext
        Patch for profiling jobs.
    inference_job_patch : mock._patch | nullcontext
        Patch for inference jobs.

    Notes
    -----
    For each "type" of job, returns a patch.
    If the associated "patch_job_type" param is False, the corresponding patch will do nothing.
    If cached jobs of a specific type aren't found, the corresponding patch will do nothing.

    Raises
    ------
    ValueError
        If jobs are still running or if any job failed.
    """
    device_patch = mock.patch(
        "qai_hub.get_devices", return_value=[device.reference_device]
    )

    if not is_hub_testing_async():
        return (device_patch, *[nullcontext() for _ in range(5)])  # type: ignore[arg-type,return-value]
    calibration_datas_to_patch: list[hub.Dataset] = []
    quantize_jobs_to_patch: list[hub.QuantizeJob] = []
    compile_jobs_to_patch: list[hub.CompileJob] = []
    profile_jobs_to_patch: list[hub.ProfileJob] = []
    inference_jobs_to_patch: list[hub.InferenceJob] = []

    # Collect pre-quantization (to ONNX) compile jobs & quantize jobs
    if patch_quantization:
        if quantize_jobs := fetch_async_test_jobs(
            hub.JobType.QUANTIZE,
            model_id,
            precision,
            None,
            device,
            component_names,
            raise_if_not_successful=True,
        ):
            pre_quantize_compile_jobs = {
                component_name: cast(
                    hub.CompileJob,
                    component_job.model.producer,
                )
                for component_name, component_job in quantize_jobs.items()
            }

            # Don't create a compile patch here yet since we may need to also patch the main compile jobs later.
            compile_jobs_to_patch.extend(pre_quantize_compile_jobs.values())
            quantize_jobs_to_patch.extend(quantize_jobs.values())
            calibration_datas_to_patch.extend(
                [x.calibration_dataset for x in quantize_jobs.values()]
            )
        elif precision != Precision.float:
            raise CachedScorecardJobError("Could not find cached quantize jobs.")

    if patch_compile:
        assert path
        if compile_jobs := fetch_async_test_jobs(
            hub.JobType.COMPILE,
            model_id,
            precision,
            path,
            device,
            component_names,
            raise_if_not_successful=True,
        ):
            compile_jobs_to_patch.extend(compile_jobs.values())
        else:
            raise CachedScorecardJobError("Could not find cached compile jobs.")

    if patch_profile:
        assert path
        if profile_jobs := fetch_async_test_jobs(
            hub.JobType.PROFILE,
            model_id,
            precision,
            path,
            device,
            component_names,
            raise_if_not_successful=True,
        ):
            profile_jobs_to_patch.extend(profile_jobs.values())
        else:
            raise CachedScorecardJobError("Could not find cached profile jobs.")

    if patch_inference:
        assert path
        if inference_jobs := fetch_async_test_jobs(
            hub.JobType.INFERENCE,
            model_id,
            precision,
            path,
            device,
            component_names,
            raise_if_not_successful=True,
        ):
            inference_jobs_to_patch.extend(inference_jobs.values())
        else:
            raise CachedScorecardJobError("Could not find cached inference jobs.")

    calib_side_effect = itertools.chain(
        calibration_datas_to_patch, itertools.repeat(mock_get_calibration_data)
    )
    calibration_data_patch = mock.patch(
        "qai_hub_models.utils.quantization.get_calibration_data",
        side_effect=callable_side_effect(calib_side_effect),
    )

    quantize_side_effect = itertools.chain(
        quantize_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    quantize_job_patch = (
        mock.patch(
            "qai_hub.submit_quantize_job",
            side_effect=callable_side_effect(quantize_side_effect),
        )
        if patch_quantization or quantize_jobs_to_patch
        else nullcontext()
    )

    # When patching quantize but not compile, the first set of compile jobs
    # need to be patched and the following calls to `submit_compile_job` should
    # actually submit jobs. Any subsequent calls should throw an error.
    if not patch_compile and compile_jobs_to_patch:
        compile_side_effect = itertools.chain(
            compile_jobs_to_patch,
            itertools.repeat(hub.submit_compile_job, len(component_names or [None])),
            itertools.repeat(_invalid_job_submission),
        )
    else:
        compile_side_effect = itertools.chain(
            compile_jobs_to_patch, itertools.repeat(_invalid_job_submission)
        )
    compile_job_patch = (
        mock.patch(
            "qai_hub.submit_compile_job",
            side_effect=callable_side_effect(compile_side_effect),
        )
        if patch_compile or compile_jobs_to_patch
        else nullcontext()
    )

    profile_side_effect = itertools.chain(
        profile_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    profile_job_patch = (
        mock.patch(
            "qai_hub.submit_profile_job",
            side_effect=callable_side_effect(profile_side_effect),
        )
        if patch_profile or profile_jobs_to_patch
        else nullcontext()
    )

    inference_side_effect = itertools.chain(
        inference_jobs_to_patch, itertools.repeat(_invalid_job_submission)
    )
    inference_job_patch = (
        mock.patch(
            "qai_hub.submit_inference_job",
            side_effect=callable_side_effect(inference_side_effect),
        )
        if patch_inference or inference_jobs_to_patch
        else nullcontext()
    )

    return (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        profile_job_patch,
        inference_job_patch,
    )


def pre_quantize_compile_via_export(
    compile_model: Callable[..., hub.CompileJob | dict[str, hub.CompileJob]],
    model_id: str,
    model: CollectionModel | BaseModel,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit ONNX compile jobs (before quantization).

    If async testing is enabled:
        Submitted jobs are added to the async testing cache,
        and this method returns immediately.

    Otherwise:
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    compile_model
        Export script function to submit compile jobs.
    model_id
        Model ID.
    model
        QAIHM instance of the model.
    device
        Scorecard device.

    """
    # Run ONNX compile jobs
    compile_output = compile_model(
        model,
        model_id,
        device.execution_device,
        TargetRuntime.ONNX,
        Precision.float,
    )
    assert compile_output is not None
    compile_jobs = (
        cast(dict[str | None, hub.Job], compile_output)
        if isinstance(compile_output, dict)
        else {None: compile_output}
    )

    # Verify success or cache job IDs to a file.
    for component, job in compile_jobs.items():
        assert_success_or_cache_job(
            model_id,
            job,
            Precision.float,
            ScorecardCompilePath.ONNX_FOR_QUANTIZATION,
            device,
            component,
        )


def quantize_via_export(
    quantize_model: Callable[..., hub.QuantizeJob | dict[str, hub.QuantizeJob] | None],
    model_id: str,
    model: CollectionModel | BaseModel,
    precision: Precision,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit quantize jobs.

    If async testing is enabled:
        Submitted jobs are added to the async testing cache,
        and this method returns immediately.

    Otherwise:
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    quantize_model
        Export script function to submit quantize jobs.
    model_id
        Model ID.
    model
        QAIHM instance of the model.
    precision
        Model precision.
    device
        Scorecard device.

    """
    # Fetch ONNX compile jobs from pre_quantize_compile_via_export
    has_components = isinstance(model, CollectionModel)
    onnx_compile_jobs = fetch_async_test_jobs(
        hub.JobType.COMPILE,
        model_id,
        Precision.float,
        ScorecardCompilePath.ONNX_FOR_QUANTIZATION,
        device,
        model.component_class_names if isinstance(model, CollectionModel) else None,
        raise_if_not_successful=True,
    )
    if onnx_compile_jobs is None:
        raise CachedScorecardJobError(
            str_with_async_test_metadata(
                "Could not find cached pre-quantization ONNX compile jobs.",
                model_id,
                precision,
                ScorecardCompilePath.ONNX_FOR_QUANTIZATION,
                device,
            )
        )

    # Extract ONNX models from compile jobs
    onnx_model_input_map = assert_success_and_get_target_models(onnx_compile_jobs)
    onnx_model_input = (
        onnx_model_input_map if has_components else onnx_model_input_map[None]
    )

    # Run quantize jobs
    with mock.patch(
        "qai_hub_models.utils.quantization.get_calibration_data",
        mock_get_calibration_data,
    ):
        quantize_output = quantize_model(
            precision,
            model,
            model_id,
            onnx_model_input,
            None,
        )
        assert quantize_output is not None
        quantize_jobs = (
            cast(dict[str | None, hub.Job], quantize_output)
            if isinstance(quantize_output, dict)
            else {None: quantize_output}
        )

    # Verify success or cache job IDs to a file.
    for component, job in quantize_jobs.items():
        assert_success_or_cache_job(model_id, job, precision, None, device, component)


def compile_via_export(
    compile_model: Callable[..., hub.CompileJob | dict[str, hub.CompileJob]],
    model_id: str,
    model: CollectionModel | BaseModel,
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    is_aimet: bool = False,
) -> None:
    """
    Use the provided export script function to submit compile jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the compile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    compile_model
        Export script function to submit compile jobs.
    model_id
        Model ID.
    model:
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    is_aimet
        Whether the model uses local aimet encodings during compilation.
    """
    has_components = isinstance(model, CollectionModel)
    if is_aimet:
        with tempfile.TemporaryDirectory() as tmpdir:
            compile_output = compile_model(
                model,
                model_id,
                device.execution_device,
                scorecard_path.runtime,
                tmpdir,
                extra_options=scorecard_path.get_compile_options(),
            )
    else:
        quantize_jobs = fetch_async_test_jobs(
            hub.JobType.QUANTIZE,
            model_id,
            precision,
            None,
            device,
            model.component_class_names if isinstance(model, CollectionModel) else None,
            raise_if_not_successful=True,
        )
        if precision != Precision.float and quantize_jobs is None:
            raise CachedScorecardJobError(
                str_with_async_test_metadata(
                    "Could not find cached quantize jobs.",
                    model_id,
                    precision,
                    scorecard_path,
                    device,
                )
            )

        quantize_job_input: hub.Model | dict[str | None, hub.Model] | None = None
        if quantize_jobs is not None:
            # Extract ONNX models from quantize jobs
            quantize_job_input_map = assert_success_and_get_target_models(quantize_jobs)
            quantize_job_input = (
                quantize_job_input_map
                if has_components
                else quantize_job_input_map[None]
            )

        compile_output = compile_model(
            model,
            model_id,
            device.execution_device,
            scorecard_path.runtime,
            precision,
            quantize_job_input,
            extra_options=scorecard_path.get_compile_options(),
        )

    compile_jobs = (
        cast(dict[str | None, hub.Job], compile_output)
        if isinstance(compile_output, dict)
        else {None: compile_output}
    )

    # Verify success or cache job IDs to a file.
    for component, job in compile_jobs.items():
        assert_success_or_cache_job(
            model_id, job, precision, scorecard_path, device, component
        )


def run_llm_compile(
    export_model: ExportFunc,
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    skip_compile_options: bool = False,
    extra_model_arguments: dict[str, Any] | None = None,
    skip_downloading: bool = True,
) -> None:
    """
    Use the provided export script function to submit compile jobs for llm tests.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the compile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    export_model
        Export script function.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    skip_compile_options
        Whether to skip compile options. Default is False.
    extra_model_arguments
        Additional model arguments to pass to export. Default is None.
    skip_downloading
        Whether to skip downloading. Default is True.

    """
    export_result = _parse_export_result(
        export_model(
            device=device.execution_device,
            precision=precision,
            skip_downloading=skip_downloading,
            skip_profiling=True,
            skip_inferencing=True,
            skip_summary=True,
            compile_options=(
                scorecard_path.get_compile_options() if not skip_compile_options else ""
            ),
            target_runtime=scorecard_path.runtime,
            **extra_model_arguments or {},
        )
    )

    # Verify success or cache job IDs to a file.
    for component, result in export_result.items():
        assert_success_or_cache_job(
            model_id, result.compile_job, precision, scorecard_path, device, component
        )


def fetch_cached_jobs_if_compile_jobs_are_identical(
    job_type_to_fetch_from_cache: (Literal[hub.JobType.PROFILE, hub.JobType.INFERENCE]),
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> Mapping[str | None, hub.Job] | None:
    """
    Checks if the compile jobs are the same, the QAIRT version matches, and the override flag is not set.
    If all conditions are met, returns the cached profile or inference job and saves the job to the YAML cache.
    Otherwise, returns None.

    Parameters
    ----------
    job_type_to_fetch_from_cache
        Type of job to fetch from cache (PROFILE or INFERENCE).
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.

    Returns
    -------
    cached_result : Mapping[str | None, hub.Job] | None
        The cached Jobs, or None if no cached job is found.
    """
    # Check if the QAIRT version matches the API version and if the override flag is set.
    # Previous scorecard QAIRT version is stored at /scorecard/intermediates/environment.env dump.
    is_override = IgnoreDeviceJobCacheEnvvar.get()
    if (
        #
        # don't run if user disabled caching
        is_override
        #
        # only prod jobs are cached
        or not deployment_is_prod(get_default_hub_deployment() or "")
        #
        # if the tool versions do not match, profiling for all paths must be re-run
        or scorecard_path.tool_versions
        != ToolVersionsByPathYaml.from_dir(INTERMEDIATES_DIR).tool_versions.get(
            scorecard_path, ToolVersions()
        )
    ):
        return None

    if job_type_to_fetch_from_cache == hub.JobType.INFERENCE:
        yaml = INFERENCE_YAML_BASE
    elif job_type_to_fetch_from_cache == hub.JobType.PROFILE:
        yaml = PROFILE_YAML_BASE
    else:
        assert_never(job_type_to_fetch_from_cache)

    compile_jobs_identical_cache_file = get_compile_jobs_are_identical_cache_file()
    compile_jobs_identical_cache = CompileJobsAreIdenticalCache.from_yaml(
        compile_jobs_identical_cache_file, create_empty_if_no_file=True
    )
    compile_jobs_are_identical = compile_jobs_identical_cache.is_identical(
        model_id, precision, scorecard_path, device, component_names
    )
    compile_jobs_identical_cache.to_yaml(compile_jobs_identical_cache_file)
    if not compile_jobs_are_identical:
        return None

    cached_jobs = fetch_async_test_jobs(
        job_type_to_fetch_from_cache,
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        yaml,
    )

    if cached_jobs is None:
        # No cached profile jobs for this model.
        return None

    for job in cached_jobs.values():
        if (status_message := job.get_status().message) and (
            "unexpected device error" in status_message
            or "Waiting for device timed out" in status_message
            or "Job timed out" in status_message
        ):
            # don't use this cached job if it is a random device or timeout failure
            return None

    return cached_jobs


def profile_via_export(
    profile_model: Callable[..., hub.ProfileJob | dict[str, hub.ProfileJob]],
    model_id: str,
    model: CollectionModel | BaseModel,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit profile jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the profile job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    profile_model
        Export script function to submit profile jobs.
    model_id
        Model ID.
    model:
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    """
    if profile_jobs := fetch_cached_jobs_if_compile_jobs_are_identical(
        hub.JobType.PROFILE,
        model_id,
        precision,
        scorecard_path,
        device,
    ):
        print(
            str_with_async_test_metadata(
                f"The compiled assets from the previous scorecard are identical. Copying over profile job(s): {' '.join([job.job_id for job in profile_jobs.values()])}",
                model_id,
                precision,
                scorecard_path,
                device,
            )
        )
    else:
        # If there are no cached profile jobs, use the export script to create a new profile job
        has_components = isinstance(model, CollectionModel)
        compile_jobs = fetch_async_test_jobs(
            hub.JobType.COMPILE,
            model_id,
            precision,
            scorecard_path,
            device,
            model.component_class_names if isinstance(model, CollectionModel) else None,
            raise_if_not_successful=True,
        )
        if not compile_jobs:
            raise CachedScorecardJobError(
                str_with_async_test_metadata(
                    "Could not find cached compile jobs.",
                    model_id,
                    precision,
                    scorecard_path,
                    device,
                )
            )

        profile_output = profile_model(
            model_id,
            device.execution_device,
            model.get_hub_profile_options(
                scorecard_path.runtime, scorecard_path.get_profile_options()
            ),
            compile_jobs if has_components else compile_jobs[None],
        )
        profile_jobs = (
            cast(dict[str | None, hub.Job], profile_output)
            if isinstance(profile_output, dict)
            else {None: profile_output}
        )

    # Verify success or cache job IDs to a file.
    for component, job in profile_jobs.items():
        assert_success_or_cache_job(
            model_id,
            job,
            precision,
            scorecard_path,
            device,
            component,
        )


def inference_via_export(
    inference_model: Callable[..., hub.InferenceJob | dict[str, hub.InferenceJob]],
    model_id: str,
    model: CollectionModel | BaseModel,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Use the provided export script function to submit inference jobs.

    If async testing is enabled:
        * If found, previously cached compile & quantize jobs
          are used, rather than submitting new ones.

        * Submitted jobs are added to the async testing cache,
          and this method returns immediately.

    Otherwise:
        * Submits all pre-requisite jobs as well as the inference job
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    inference_model
        Export script function to submit inference jobs.
    model_id
        Model ID.
    model:
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    """
    # TODO(#15111): Reenable caching for Inference Jobs. Inference jobs also must check that input datasets are the same, not just the compiled assets.
    has_components = isinstance(model, CollectionModel)
    compile_jobs = fetch_async_test_jobs(
        hub.JobType.COMPILE,
        model_id,
        precision,
        scorecard_path,
        device,
        model.component_class_names if isinstance(model, CollectionModel) else None,
        raise_if_not_successful=True,
    )
    if not compile_jobs:
        raise CachedScorecardJobError(
            str_with_async_test_metadata(
                "Could not find cached compile jobs.",
                model_id,
                precision,
                scorecard_path,
                device,
            )
        )

    runtime = scorecard_path.runtime
    inference_output = inference_model(
        model.sample_inputs(
            use_channel_last_format=runtime.channel_last_native_execution
        ),
        model_id,
        device.execution_device,
        model.get_hub_profile_options(runtime, scorecard_path.get_profile_options()),
        compile_jobs if has_components else compile_jobs[None],
    )
    inference_jobs: dict[str | None, hub.Job] = (
        cast(dict[str | None, hub.Job], inference_output)
        if isinstance(inference_output, dict)
        else {None: inference_output}
    )
    # Verify success or cache job IDs to a file.
    for component, job in inference_jobs.items():
        assert_success_or_cache_job(
            model_id,
            job,
            precision,
            scorecard_path,
            device,
            component,
        )


def export_test_e2e(
    export_model: ExportFunc,
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Verifies the export script function provided works end to end.

    If async testing is enabled:
        * If found, existing Hub jobs are are used, rather than submitting new ones.

    Otherwise:
        * Submits all (quantize, compile, profile) jobs on Hub
        Waits for the submitted jobs and asserts success.
        NOTE: This method will wait infinitely long for running jobs.

    Parameters
    ----------
    export_model
        Export script function.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    """
    # Some scorecards will run without the profiling step.
    has_cached_profile_jobs = (
        os.path.exists(get_profile_job_ids_file())
        and os.stat(get_profile_job_ids_file()).st_size > 0
    )

    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        profile_job_patch,
        _,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        patch_quantization=not QAIHMModelCodeGen.from_model(model_id).is_aimet,
        patch_compile=True,
        patch_profile=has_cached_profile_jobs,
        patch_inference=False,
    )

    # Test export script end to end
    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        profile_job_patch,
        tempfile.TemporaryDirectory() as tmpdir,
    ):
        skip_upload_to_s3 = (
            not scorecard_path.is_public
            or get_qaihm_s3 is None
            or s3_multipart_upload is None
            or QAIHM_PRIVATE_S3_BUCKET is None
            or S3ArtifactsDirEnvvar.is_default()
        )

        result = export_model(
            device=device.execution_device,
            precision=precision,
            target_runtime=scorecard_path.runtime,
            compile_options=scorecard_path.compile_path.get_compile_options(),
            profile_options=scorecard_path.get_profile_options(),
            skip_downloading=skip_upload_to_s3,
            skip_profiling=not has_cached_profile_jobs,
            skip_inferencing=True,
            output_dir=tmpdir,
            zip_assets=True,
        )

        if result.download_path is not None:
            assert s3_multipart_upload is not None
            assert get_qaihm_s3 is not None
            assert QAIHM_PRIVATE_S3_BUCKET is not None

            assets_cache_path = get_release_assets_file()
            assets_cache = ScorecardAssetYaml.from_yaml(
                assets_cache_path, create_empty_if_no_file=True
            )
            if assets_cache.get_asset(
                model_id,
                precision,
                device if scorecard_path.runtime.is_aot_compiled else cs_universal,
                scorecard_path,
            ):
                # Asset for this runtime (device agnostic) or runtime + chipset exists already.
                return

            s3_bucket = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
            s3_key = str(S3ArtifactsDirEnvvar.get() / result.download_path.name)

            s3_multipart_upload(
                bucket=s3_bucket,
                key=s3_key,
                local_file_path=result.download_path,
            )

            assets_cache.add_asset(
                QAIHMModelReleaseAssets.AssetDetails(
                    s3_key=s3_key, tool_versions=result.tool_versions
                ),
                model_id,
                precision,
                device if scorecard_path.runtime.is_aot_compiled else cs_universal,
                scorecard_path,
            )
            assets_cache.to_yaml(assets_cache_path)


def on_device_inference_for_accuracy_validation(
    model: type[BaseModel | CollectionModel],
    dataset_name: str,
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Runs an inference job on the given dataset.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model class to run inference on.
    dataset_name
        Name of the dataset to use for evaluation.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.

    """
    compile_jobs = fetch_async_test_jobs(
        hub.JobType.COMPILE,
        model_id,
        precision,
        scorecard_path.compile_path,
        device,
        raise_if_not_successful=True,
    )
    if not compile_jobs:
        raise CachedScorecardJobError(
            str_with_async_test_metadata(
                "Missing cached compile job",
                model_id,
                precision,
                scorecard_path,
                device,
            )
        )

    for component_name, job in compile_jobs.items():
        hub_val_dataset = get_hub_val_dataset(
            dataset_name,
            get_dataset_ids_file(),
            model,
            apply_channel_transpose=scorecard_path.runtime.channel_last_native_execution,
            num_samples=get_num_eval_samples(dataset_name),
        )
        ijob = hub.submit_inference_job(
            device=device.execution_device,
            inputs=hub_val_dataset,
            model=job.get_target_model(),
            name=model_id,
        )
        assert_success_or_cache_job(
            model_id, ijob, precision, scorecard_path, device, component_name
        )


def torch_inference_for_accuracy_validation(
    model: BaseModel | CollectionModel, dataset_name: str, model_id: str
) -> None:
    """
    Runs torch inference job on the given dataset.
    Uploads the results to hub and caches them.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model instance to run inference on.
    dataset_name
        Name of the dataset to use for evaluation.
    model_id
        Model ID.

    """
    assert isinstance(model, BaseModel), (
        "This function is not yet supported for CollectionModel."
    )
    # Get the first dim of the first input. This is always the batch size.
    compiled_batch_size = next(iter(model.get_input_spec().values()))[0][0]

    inputs, *_ = next(
        iter(
            get_torch_val_dataloader(
                dataset_name,
                get_num_eval_samples(dataset_name),
                model.get_input_spec(),
            )
        )
    )
    if not isinstance(inputs, list) and not isinstance(inputs, tuple):
        # Generalize: treat "single-input" model as a list of 1 input.
        # This allows us to support single and multi-input models in 1 loop.
        inputs = [inputs]

    num_batches = inputs[0].shape[0]
    output_names = model.get_output_names()
    outputs: list[list[np.ndarray]] = [[] for _ in output_names]
    for b in range(math.ceil(num_batches / compiled_batch_size)):
        # Complete N batches at a time to substantially reduces memory pressure
        # (not all inputs / outputs need to be in memory at once)
        # Without this, scorecard jobs can easily run OOM.
        #
        # TODO(#15497): We should strive to disable single-batch inference. Multi-batch inference is much faster.
        model_inputs = [
            x[b * compiled_batch_size : min((b + 1) * compiled_batch_size, x.shape[0])]
            for x in inputs
        ]
        model_outputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] = model(
            *model_inputs
        )
        if not isinstance(model_outputs, (list, tuple)):
            # Generalize: treat "single-output" model as a list of 1 output.
            # This allows us to support single and multi-output models in 1 loop.
            model_outputs = [model_outputs]
        for i, output in enumerate(model_outputs):
            outputs[i].append(output.numpy())
    hub_entries = dict(zip(output_names, outputs, strict=False))
    cache_dataset(model_id, "torch_val", hub.upload_dataset(hub_entries))


def torch_inference_for_accuracy_validation_outputs(model_id: str) -> list[np.ndarray]:
    """
    Fetches torch inference results computed by torch_inference_for_accuracy_validation
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model_id
        Model ID.

    Returns
    -------
    inference_outputs : list[np.ndarray]
        List of results, in order of output from the torch model.
        [ output_0_array, output_1_array, ... ]
    """
    dataset = get_cached_dataset_entries(model_id, "torch_val")
    if not dataset:
        raise ValueError(f"Missing inference output dataset for model {model_id}")

    # Hub DatasetEntries is a dict of format {'name' [ batch_val_1, batch_val_2, etc.]}
    #
    # This flattens the dict into a list of the same order,
    # and merges the list of batch outputs for each dictionary entry into a single tensor.
    return [
        np.concatenate(tensor_list, axis=0) if len(tensor_list) > 1 else tensor_list[0]
        for tensor_list in dataset.values()
    ]


def split_and_group_accuracy_validation_output_batches(
    torch_inference_outputs: list[np.ndarray],
) -> list[torch.Tensor | tuple[torch.Tensor, ...]]:
    """
    Converts output generated by torch_inference_for_accuracy_validation_outputs to a different format.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    torch_inference_outputs
        Return value of torch_inference_for_accuracy_validation_outputs.

    Returns
    -------
    batched_outputs : list[torch.Tensor | tuple[torch.Tensor, ...]]
        If torch_inference_outputs is length 1:
            [output_0::batch_0, output_0::batch_1, ...]

        otherwise:
            [(output_0::batch_0, output_1::batch_0, ...),
             (output_0::batch_1, output_1::batch_1, ...),
             ...]

        Note that the batch dimension is preserved in all returned tensors (it is always 1).
    """
    num_outputs = len(torch_inference_outputs)
    if num_outputs == 1:
        output = torch.tensor(torch_inference_outputs[0])
        return list(output.split(1))

    num_batches = len(torch_inference_outputs[0])
    outputs_per_batch: list[torch.Tensor | tuple[torch.Tensor, ...]] = [
        tuple(
            torch.Tensor(output_n[batch_idx]).unsqueeze(0)
            for output_n in torch_inference_outputs
        )
        for batch_idx in range(num_batches)
    ]
    return outputs_per_batch


def accuracy_on_sample_inputs_via_export(
    export_model: ExportFunc,
    model_id: str,
    model: BaseModel | CollectionModel,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
) -> None:
    """
    Computes accuracy for the given model's sample inputs and saves it to disk.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    export_model
        Code-generated export function from export.py.
    model_id
        Model ID.
    model
        QAIHM instance of the model
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None of there are no components.
        Default is None.
    """
    if isinstance(model, CollectionModel):
        component_names = component_names or model.component_class_names
    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        _,  # profile_job_patch
        inference_job_patch,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        component_names,
        patch_quantization=not QAIHMModelCodeGen.from_model(model_id).is_aimet,
        patch_compile=True,
        patch_profile=False,
        patch_inference=True,
    )

    psnr_values: list[str] = []

    def _mock_tabulate_fn(df: Any, **kwargs: Any) -> str:
        new_psnr_values, tabulate_results = mock_tabulate_fn(df)
        psnr_values.extend(new_psnr_values)
        return tabulate_results

    tabulate_patch = mock.patch(
        "qai_hub_models.utils.printing.tabulate",
        side_effect=_mock_tabulate_fn,
    )

    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        inference_job_patch,
        tabulate_patch,
    ):
        export_model(
            device=device.execution_device,
            target_runtime=scorecard_path.runtime,
            precision=precision,
            skip_downloading=True,
            skip_profiling=True,
        )

    write_accuracy(model_id, device.chipset, precision, scorecard_path, psnr_values)


def _get_dataset_cache_patch(
    dataset_name: str,
    scorecard_path: ScorecardProfilePath,
    model_cls: type[BaseModel | CollectionModel],
) -> mock._patch:
    # Patch input eval dataset to use a cached dataset if it exists
    dataset_dir = get_and_sync_datasets_cache_dir(
        scorecard_path.runtime.channel_last_native_execution,
        dataset_name,
        get_num_eval_samples(dataset_name),
        model_cls,
    )
    return mock.patch(
        "qai_hub_models.utils.evaluate.get_hub_datasets_path",
        return_value=dataset_dir.parent,
    )


def get_num_eval_samples(dataset_name: str) -> int:
    """
    Resolve how many samples to evaluate for a given dataset.

    This needs to be set in multiple callsites and for both num_samples and samples_per_job.

    Parameters
    ----------
    dataset_name
        Name of the dataset.

    Returns
    -------
    num_samples : int
        Number of samples to evaluate.
    """
    return min(
        DATASET_NAME_MAP[dataset_name].default_samples_per_job(),
        DEFAULT_NUM_EVAL_SAMPLES,
    )


def accuracy_on_dataset_via_evaluate_and_export(
    export_model: ExportFunc,
    model: BaseModel,
    dataset_name: str,
    torch_val_outputs: list[np.ndarray],
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
    model_id: str,
    precision: Precision,
    scorecard_path: ScorecardProfilePath,
    device: ScorecardDevice,
) -> None:
    """
    Computes accuracy for the given model and dataset and saves it to disk.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    export_model
        Code-generated export function from export.py.
    model
        Model instance to run inference on.
    dataset_name
        Name of the dataset to use for evaluation.
    torch_val_outputs
        Torch validation outputs.
    torch_evaluate_mock_outputs
        The outputs of the torch forward passes in the format expected by the evaluate function.
    model_id
        Model ID.
    precision
        Model precision.
    scorecard_path
        Scorecard path.
    device
        Scorecard device.

    """
    cache_path_patch = _get_dataset_cache_patch(
        dataset_name, scorecard_path, model.__class__
    )

    cpu_accuracy = load_yaml(get_cpu_accuracy_file())
    sim_acc = cpu_accuracy.get(_get_sim_cpu_key(model_id, precision), None)

    torch_key = _get_torch_cpu_key(model_id)
    if torch_key not in cpu_accuracy:
        raise CachedScorecardJobError(
            "Torch accuracy data is missing. Accuracy data collection (test_torch_accuracy) probably failed."
        )
    torch_acc = float(cpu_accuracy[torch_key])

    dataset_metadata = None
    metric_metadata = None
    num_samples = None
    with suppress(NotImplementedError):
        dataset_cls = DATASET_NAME_MAP[dataset_name]
        dataset_metadata = dataset_cls.get_dataset_metadata()
        num_samples = get_num_eval_samples(dataset_name)
        dataset_metadata = DATASET_NAME_MAP[dataset_name].get_dataset_metadata()
        metric_metadata = model.get_evaluator().get_metric_metadata()

    try:
        # Get existing inference jobs, then create related patches
        # This will raise a ValueError if any of the jobs failed
        inference_jobs = fetch_async_test_jobs(
            hub.JobType.INFERENCE,
            model_id,
            precision,
            scorecard_path,
            device,
            raise_if_not_successful=True,
        )
        if not inference_jobs:
            raise CachedScorecardJobError(  # noqa: TRY301
                str_with_async_test_metadata(
                    "Missing cached inference job",
                    model_id,
                    precision,
                    scorecard_path,
                    device,
                )
            )
    except (CachedScorecardJobError, ValueError):
        # If no on-device accuracy numbers, we still want to write torch, sim numbers
        write_accuracy(
            model_id,
            device.chipset,
            precision,
            scorecard_path,
            [],
            torch_acc,
            None,
            float(sim_acc) if sim_acc is not None else None,
            dataset_name,
            dataset_metadata,
            metric_metadata,
            num_samples,
        )
        raise

    inference_output_datas = [x.download_output_data() for x in inference_jobs.values()]
    dataset_download_patch = mock.patch(
        "qai_hub.client.Dataset.download", side_effect=inference_output_datas
    )
    inference_job_dataset_download_patch = mock.patch(
        "qai_hub.client.InferenceJob.download_output_data",
        side_effect=inference_output_datas,
    )
    on_device_call_patch = mock.patch.object(
        AsyncOnDeviceModel,
        "__call__",
        new=callable_side_effect(
            iter([mock_on_device_model_call(x) for x in inference_jobs.values()])
        ),
    )
    torch_call_patch = mock.patch(
        "qai_hub_models.utils.evaluate.BaseModel.__call__",
        side_effect=torch_evaluate_mock_outputs,
    )
    compare_torch_inference_patch = mock.patch(
        "qai_hub_models.utils.compare._torch_inference_impl",
        side_effect=[torch_val_outputs],
    )

    # Run eval script to collect accuracy metrics
    num_samples = get_num_eval_samples(dataset_name)
    with (
        cache_path_patch,
        dataset_download_patch,
        on_device_call_patch,
        torch_call_patch,
    ):
        inference_job = inference_jobs[None]
        evaluate_result = evaluate_on_dataset(
            evaluator_func=model.get_evaluator,
            compiled_model=inference_job.model,
            hub_device=inference_job.device,
            dataset_name=dataset_name,
            use_cache=True,
            num_samples=num_samples,
            samples_per_job=num_samples,
        )

    # Patch previous jobs
    (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        _,
        inference_job_patch,
    ) = patch_hub_with_cached_jobs(
        model_id,
        precision,
        scorecard_path,
        device,
        patch_quantization=not QAIHMModelCodeGen.from_model(model_id).is_aimet,
        patch_compile=True,
        patch_inference=True,
    )

    psnr_values: list[str] = []

    def _mock_tabulate_fn(df: Any, **kwargs: Any) -> str:
        new_psnr_values, tabulate_results = mock_tabulate_fn(df)
        psnr_values.extend(new_psnr_values)
        return tabulate_results

    tabulate_patch = mock.patch(
        "qai_hub_models.utils.printing.tabulate",
        side_effect=_mock_tabulate_fn,
    )
    with (
        device_patch,
        calibration_data_patch,
        quantize_job_patch,
        compile_job_patch,
        inference_job_patch,
        compare_torch_inference_patch,
        inference_job_dataset_download_patch,
        tabulate_patch,
    ):
        export_model(
            device=device.execution_device,
            target_runtime=scorecard_path.runtime,
            precision=precision,
            skip_downloading=True,
            skip_profiling=True,
        )

    write_accuracy(
        model_id,
        device.chipset,
        precision,
        scorecard_path,
        psnr_values,
        torch_acc,
        evaluate_result.device_accuracy,
        float(sim_acc) if sim_acc is not None else None,
        dataset_name,
        dataset_metadata,
        metric_metadata,
        num_samples,
    )


def torch_accuracy_on_dataset(
    model: BaseModel,
    dataset_name: str,
    torch_evaluate_mock_outputs: list[torch.Tensor | tuple[torch.Tensor, ...]],
    model_id: str,
) -> None:
    """
    Computes accuracy for the given model on pytorch.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model instance to run inference on.
    dataset_name
        Name of the dataset to use for evaluation.
    torch_evaluate_mock_outputs
        The outputs of the torch forward passes in the format
        expected by the evaluate function.
    model_id
        Model ID.

    """
    torch_call_patch = mock.patch(
        "qai_hub_models.utils.evaluate.BaseModel.__call__",
        side_effect=torch_evaluate_mock_outputs,
    )
    scorecard_path = ScorecardProfilePath.ONNX
    cache_path_patch = _get_dataset_cache_patch(
        dataset_name, scorecard_path, model.__class__
    )
    num_samples = get_num_eval_samples(dataset_name)
    with torch_call_patch, cache_path_patch:
        evaluate_result = evaluate_on_dataset(
            evaluator_func=model.get_evaluator,
            torch_model=model,
            use_cache=True,
            dataset_name=dataset_name,
            num_samples=num_samples,
            samples_per_job=num_samples,
        )
    cache_key = _get_torch_cpu_key(model_id)
    append_line_to_file(
        get_cpu_accuracy_file(),
        f"{cache_key}: {evaluate_result.torch_accuracy:.3g}",
    )


def sim_accuracy_on_dataset(
    model: BaseModel,
    dataset_name: str,
    model_id: str,
    precision: Precision,
) -> None:
    """
    Computes accuracy for the given model on quantsim.
    Async testing must be enabled to run this method.

    Parameters
    ----------
    model
        Model instance to run inference on.
    dataset_name
        Name of the dataset to use for evaluation.
    model_id
        Model ID.
    precision
        Model precision.

    """
    if precision == Precision.float:
        return
    scorecard_path = ScorecardProfilePath.ONNX
    cache_path_patch = _get_dataset_cache_patch(
        dataset_name, scorecard_path, model.__class__
    )
    quantize_jobs = fetch_async_test_jobs(
        hub.JobType.QUANTIZE,
        model_id,
        precision,
        None,
        cs_universal,
        raise_if_not_successful=True,
    )
    assert quantize_jobs is not None
    if len(quantize_jobs) != 1:
        print(
            f"Expected 1 quantize job for precision {precision} but got {len(quantize_jobs)}."
        )

    num_samples = get_num_eval_samples(dataset_name)
    with cache_path_patch:
        evaluate_result = evaluate_on_dataset(
            evaluator_func=model.get_evaluator,
            quantized_model=quantize_jobs[None].get_target_model(),
            dataset_name=dataset_name,
            use_cache=True,
            num_samples=num_samples,
            samples_per_job=num_samples,
        )
        cache_key = _get_sim_cpu_key(model_id, precision)
        append_line_to_file(
            get_cpu_accuracy_file(),
            f"{cache_key}: {evaluate_result.sim_accuracy:.3g}",
        )
