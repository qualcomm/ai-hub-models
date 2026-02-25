# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast, overload

import qai_hub as hub
from pydantic import Field
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.datasets.common import DatasetMetadata
from qai_hub_models.evaluators.base_evaluators import MetricMetadata
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.envvars import (
    DEFAULT_AGGREGATED_CSV_NAME,
    ArtifactsDirEnvvar,
    EnableAsyncTestingEnvvar,
)
from qai_hub_models.scorecard.execution_helpers import get_async_job_cache_name
from qai_hub_models.scorecard.results.scorecard_job import ScorecardJob
from qai_hub_models.scorecard.results.yaml import (
    COMPILE_YAML_BASE,
    DEFAULT_TOOL_VERSIONS_YAML_FILE_NAME,
    get_scorecard_job_yaml,
)
from qai_hub_models.utils.asset_loaders import load_yaml, qaihm_temp_dir
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.file_hash import file_hashes_are_identical
from qai_hub_models.utils.onnx.torch_wrapper import extract_onnx_zip

# If a model has many outputs, how many of them to store PSNR for
MAX_PSNR_VALUES = 10


class CachedScorecardJobError(ValueError):
    """Raised when a scorecard job is missing or failed."""


def callable_side_effect(side_effects: Iterator) -> Callable:
    """
    Return a function that:
        * Gets the next value in side_effects.
            * If the value is not callable, returns it directly.
            * If the value is callable, calls() it (using passthrough function arguments)
                and returns the function's result.

        Example,
        def my_func(input: str) -> str:
            return str + '_hello_world'
        f = callable_side_effect('1', '3', my_func)

        f("hello_world") # returns: '1'
        f("testing 123") # returns: '3'
        f("beep") # returns 'beep_hello_world'
        f("boop") # raises error (out of values to iterate over)
    """

    def f(*args: Any, **kwargs: Any) -> object:
        result = next(side_effects)
        if callable(result):
            return result(*args, **kwargs)
        return result

    return f


def append_line_to_file(path: os.PathLike, line: str) -> None:
    with open(path, "a") as f:
        f.write(line + "\n")


def is_hub_testing_async() -> bool:
    return EnableAsyncTestingEnvvar.get()


def get_artifacts_dir_opt() -> Path:
    return ArtifactsDirEnvvar.get()


def get_artifact_filepath(
    filename: str, artifacts_dir: os.PathLike | str | None = None, create: bool = True
) -> Path:
    artifacts_dir = Path(artifacts_dir or get_artifacts_dir_opt())
    os.makedirs(artifacts_dir, exist_ok=True)
    path = artifacts_dir / filename
    if create:
        path.touch()
    return path


def get_dataset_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("dataset-ids.yaml", artifacts_dir)


def get_compile_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("compile-jobs.yaml", artifacts_dir)


def get_compile_jobs_are_identical_cache_file(
    artifacts_dir: os.PathLike | str | None = None,
) -> Path:
    return get_artifact_filepath("compile-jobs-are-identical-cache.yaml", artifacts_dir)


def get_profile_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("profile-jobs.yaml", artifacts_dir)


def get_inference_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("inference-jobs.yaml", artifacts_dir)


def get_quantize_job_ids_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("quantize-jobs.yaml", artifacts_dir)


def get_cpu_accuracy_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("cpu-accuracy.yaml", artifacts_dir)


def get_tool_versions_file(
    artifacts_dir: os.PathLike | str | None = None, create: bool = True
) -> Path:
    return get_artifact_filepath(
        DEFAULT_TOOL_VERSIONS_YAML_FILE_NAME, artifacts_dir, create
    )


def get_environment_file(
    artifacts_dir: os.PathLike | str | None = None, create: bool = True
) -> Path:
    return get_artifact_filepath("environment.env", artifacts_dir, create)


def get_release_assets_file(artifacts_dir: os.PathLike | str | None = None) -> Path:
    return get_artifact_filepath("release-assets.yaml", artifacts_dir)


def get_accuracy_metadata_columns() -> list[str]:
    return [
        "dataset_name",
        "dataset_link",
        "split_description",
        "metric_name",
        "metric_unit",
        "metric_description",
        "metric_min",
        "metric_max",
        "metric_threshold",
        "num_samples",
    ]


def get_accuracy_numerics_columns() -> list[str]:
    cols = ["Torch Accuracy", "Sim Accuracy", "Device Accuracy"]
    cols.extend(f"PSNR_{i}" for i in range(MAX_PSNR_VALUES))
    return cols


def get_accuracy_columns() -> list[str]:
    cols = ["model_id", "precision", "runtime"]
    cols.extend(get_accuracy_numerics_columns())
    cols.extend(["date", "branch", "chipset"])
    cols.extend(get_accuracy_metadata_columns())
    return cols


def get_accuracy_file() -> Path:
    filepath = get_artifact_filepath("accuracy.csv")
    if filepath.stat().st_size == 0:
        append_line_to_file(filepath, ",".join(get_accuracy_columns()))
    return filepath


def get_async_test_job_cache_path(job_type: hub.JobType) -> Path:
    """
    Loads the appropriate Scorecard job cache for the type of the given job.

    Parameters;
        job_type: hub.JobType
            Job Type
    """
    if job_type == hub.JobType.COMPILE:
        return get_compile_job_ids_file()
    if job_type == hub.JobType.PROFILE:
        return get_profile_job_ids_file()
    if job_type == hub.JobType.INFERENCE:
        return get_inference_job_ids_file()
    if job_type == hub.JobType.QUANTIZE:
        return get_quantize_job_ids_file()
    raise NotImplementedError(
        f"No file for storing test jobs of type {job_type.display_name}"
    )


def str_with_async_test_metadata(
    val: str,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    device: ScorecardDevice | None,
    component: str | None = None,
) -> str:
    """
    Generate a string (generally used for printing) that includes the scorecard run metadata with the value.
    Prints : {model_name::model_component} | {path} | {device} | val

    Parameters;
        model_id: str
            Model ID

        precision: Precision
            Model precision

        path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None
            Scorecard path

        device: ScorecardDevice | None
            Scorecard device

        component: str | None = None
            Name of model component (if applicable)
    """
    model_name = f"{model_id}::{component}" if component else model_id
    return f"{model_name} | {precision} | {f'{path.name} | ' if path else ''}{f'{device.name} | ' if device else ''}{val}"


def assert_success_or_cache_job(
    model_id: str,
    job: hub.Job | None,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | TargetRuntime | None,
    device: ScorecardDevice = cs_universal,
    component: str | None = None,
) -> None:
    if (
        job is None
        and path is None
        and precision in [Precision.mixed, Precision.mixed_with_float]
    ):
        return  # For models where precision varies per-component, some components may not have jobs (e.g., float components in quantization)

    assert job is not None
    if is_hub_testing_async():
        cache_path = get_async_test_job_cache_path(job._job_type)
        cache = get_scorecard_job_yaml(job._job_type)
        cache.set_job_id(job.job_id, path, model_id, device, precision, component)
        cache.to_file(cache_path, append=True)
    else:
        job_status = job.wait()
        assert job_status.success, str_with_async_test_metadata(
            f"{job._job_type.display_name.title()} job ({job.job_id}) failed: {job_status.message}",
            model_id,
            precision,
            path,
            device,
            component,
        )


@overload
def fetch_async_test_job(
    job_type: Literal[hub.JobType.COMPILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> hub.CompileJob | None: ...


@overload
def fetch_async_test_job(
    job_type: Literal[hub.JobType.PROFILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> hub.ProfileJob | None: ...


@overload
def fetch_async_test_job(
    job_type: Literal[hub.JobType.INFERENCE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> hub.InferenceJob | None: ...


@overload
def fetch_async_test_job(
    job_type: Literal[hub.JobType.QUANTIZE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component: str | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> hub.QuantizeJob | None: ...


@overload
def fetch_async_test_job(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component: str | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> hub.Job | None: ...


def fetch_async_test_job(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component: str | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> hub.Job | None:
    """
    Get the successful async test job that corresponds to the given parameters.

    Parameters
    ----------
    job_type
        Type of job to fetch.
    model_id
        Model ID.
    precision
        Model precision.
    path
        Scorecard path.
    device
        Scorecard device.
    component
        Name of model component. Default is None.
    cache_path
        Path to the cache file. If None, uses the default cache path. Default is None.
    raise_if_not_successful
        Raise a ValueError if any job is not successful. Default is False.

    Returns
    -------
    cached_job : hub.Job | None
        A successful Hub job, or None if this job type was not found in the cache.

    Raises
    ------
    ValueError
        If the job is cached but failed or is still running.
    """
    scorecard_job: ScorecardJob = get_scorecard_job_yaml(
        job_type, cache_path or get_async_test_job_cache_path(job_type)
    ).get_job(
        path,
        model_id,
        device,
        precision,
        component,
        wait_for_job=True,
    )

    if not scorecard_job.job_id:
        # No job ID, this wasn't found in the cache.
        return None
    if raise_if_not_successful and not scorecard_job.success:
        # If the job has an ID but it's marked as "skipped", then it timed out.
        error_str = (
            "still running" if scorecard_job.skipped else scorecard_job.job_status
        )
        raise CachedScorecardJobError(
            str_with_async_test_metadata(
                f"{scorecard_job.job._job_type.display_name.title()} job {error_str}: {scorecard_job.job.url}",
                model_id,
                precision,
                path,
                device,
                component,
            )
        )

    return scorecard_job.job


@overload
def fetch_async_test_jobs(
    job_type: Literal[hub.JobType.COMPILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> Mapping[str | None, hub.CompileJob] | None: ...


@overload
def fetch_async_test_jobs(
    job_type: Literal[hub.JobType.PROFILE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> Mapping[str | None, hub.ProfileJob] | None: ...


@overload
def fetch_async_test_jobs(
    job_type: Literal[hub.JobType.INFERENCE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> Mapping[str | None, hub.InferenceJob] | None: ...


@overload
def fetch_async_test_jobs(
    job_type: Literal[hub.JobType.QUANTIZE],
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> Mapping[str | None, hub.QuantizeJob] | None: ...


@overload
def fetch_async_test_jobs(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> Mapping[str | None, hub.Job] | None: ...


def fetch_async_test_jobs(
    job_type: hub.JobType,
    model_id: str,
    precision: Precision,
    path: ScorecardCompilePath | ScorecardProfilePath | None,
    device: ScorecardDevice,
    component_names: list[str] | None = None,
    cache_path: str | Path | None = None,
    raise_if_not_successful: bool = False,
) -> Mapping[str | None, hub.Job] | None:
    """
    Get the async test jobs that correspond to the given parameters.

    Parameters
    ----------
    job_type
        Type of hub job to fetch.
    model_id
        Model ID.
    precision
        Model precision.
    path
        Scorecard path.
    device
        Scorecard device.
    component_names
        Name of all model components (if applicable), or None if there are no components.
        Default is None.
    cache_path
        Path to the cache file. If None, uses the default cache path. Default is None.
    raise_if_not_successful
        Raise a ValueError if any job is not successful. Default is False.

    Returns
    -------
    component_jobs : Mapping[str | None, hub.Job] | None
        For models WITHOUT components, returns a dict: { None: Job }.
        For models WITH components, returns a dict: { 'component_1_name': Job, ... }.
        Returns None if one or more components do not have a cached job of the given type.

        If the precision type is "variable", and the job type is QUANTIZE, some components may not have a cached job.
        In this case, only components with cached jobs will be returned. Components without cached jobs are assumed
        to be floating point components that do not require quantization, and will be excluded from the returned dict.

    Raises
    ------
    ValueError
        If raise_if_not_successful is True and any cached job failed or is still running.
    """
    component_jobs: dict[str | None, hub.Job | None] = {}
    for component in component_names or [None]:  # type: ignore[list-item]
        component_jobs[component] = fetch_async_test_job(
            job_type,
            model_id,
            precision,
            path,  # type: ignore[arg-type]
            device,
            component,
            cache_path,
            raise_if_not_successful,
        )

    if (
        precision in [Precision.mixed, Precision.mixed_with_float]
        and hub.JobType.QUANTIZE
    ):
        # Variable precision quantize jobs may be missing for some components
        # (if a component is floating point).
        component_jobs = {
            component: job
            for component, job in component_jobs.items()
            if job is not None
        }

    has_jobs = all(component_jobs.values())
    return component_jobs if has_jobs else None  # type: ignore[return-value]


def cache_dataset(model_id: str, dataset_name: str, dataset: hub.Dataset) -> None:
    append_line_to_file(
        get_dataset_ids_file(),
        f"{model_id}_{dataset_name}: {dataset.dataset_id}",
    )


def get_cached_dataset(model_id: str, dataset_name: str) -> hub.Dataset | None:
    dataset_ids = load_yaml(get_dataset_ids_file())
    key = f"{model_id}_{dataset_name}"
    if key not in dataset_ids:
        return None
    return hub.get_dataset(dataset_ids[key])


def get_cached_dataset_entries(
    model_id: str, dataset_name: str
) -> DatasetEntries | None:
    if x := get_cached_dataset(model_id, dataset_name):
        return cast(DatasetEntries, x.download())
    return None


def get_job_date(artifacts_dir: os.PathLike | str | None = None) -> str:
    date_file = get_artifact_filepath("date.txt", artifacts_dir)
    if date_file.stat().st_size == 0:
        curr_date = datetime.today().strftime("%Y-%m-%d")
        with open(date_file, "w") as f:
            f.write(curr_date)
        return curr_date
    with open(date_file) as f:
        return f.read()


def write_accuracy(
    model_name: str,
    chipset: str,
    precision: Precision,
    path: ScorecardProfilePath,
    psnr_values: list[str],
    torch_accuracy: float | None = None,
    device_accuracy: float | None = None,
    sim_accuracy: float | None = None,
    dataset_name: str | None = None,
    dataset_metadata: DatasetMetadata | None = None,
    metric_metadata: MetricMetadata | None = None,
    num_samples: int | None = None,
) -> None:
    line = f"{model_name},{precision!s},{path.value},"
    line += f"{torch_accuracy:.3g}," if torch_accuracy is not None else ","
    line += f"{sim_accuracy:.3g}," if sim_accuracy is not None else ","
    line += f"{device_accuracy:.3g}," if device_accuracy is not None else ","
    if len(psnr_values) >= MAX_PSNR_VALUES:
        line += ",".join(psnr_values[:10])
    else:
        # If the psnr list is empty, we only want 9 commas after
        line += ",".join(psnr_values) + "," * min(MAX_PSNR_VALUES - len(psnr_values), 9)
    line += f",{get_job_date()},main,{chipset}"

    line += ","
    if dataset_name is not None:
        line += dataset_name

    if dataset_metadata is not None:
        line += f",{dataset_metadata.link},{dataset_metadata.split_description}"
    else:
        line += ",,"

    if metric_metadata is not None:
        line += f",{metric_metadata.name},{metric_metadata.unit},{metric_metadata.description},{metric_metadata.range[0]},{metric_metadata.range[1]},{metric_metadata.float_vs_device_threshold}"
    else:
        line += ",,,"

    line += ","
    if num_samples is not None:
        line += str(num_samples)
    append_line_to_file(get_accuracy_file(), line)


class CompileJobsAreIdenticalCache(BaseQAIHMConfig):
    """
    When running scorecard, users have the option to enable a caching feature. This feature:
        * Checks if compile assets from the previous scorecard and the current scorecard are the same.
        * If the compiled assets are the same, profile / inference jobs from the previous scorecard are re-used.

    One challenge in the implementation the above caching feature is that we need to download the output of both
    compile jobs in each test and MD5 each file to determine if they're the same. The "naive" implementation runs this
    on every individual profile & inference test. This is unecessary for assets that aren't device-specific.

    This caches the sameness of current compile jobs and previous compile jobs, so we only need to compute sameness once.

    -----

    SUCCESSFUL COMPILE JOBS
        If:
            * both compile jobs from the previous scorecard and the current scorecard are succesful
            * the produced asset file is the same (md5 hash)
        The jobs are considered "identical".

    FAILING and MISSING COMPILE JOBS
        If:
            * a compile job is missing in EITHER OR BOTH the previous / current scorecard
            * a compile job failed in EITHER OR BOTH the previous / current scorecard,
        The jobs are considered NOT identical. This is the case even if both scorecard jobs fail with the same reason.

    COMPONENT MODELS
        Component models are considered "identical" only if all components' compile jobs are identical (per the above guidelines).
    """

    compile_jobs_are_identical: dict[str, bool] = Field(default_factory=dict)

    def is_identical(
        self,
        model_id: str,
        precision: Precision,
        path: ScorecardCompilePath | ScorecardProfilePath,
        device: ScorecardDevice,
        component_names: list[str] | None = None,
    ) -> bool:
        """
        Returns true if:
            * All compile jobs for the given parameters passed in previous and current scorecards,
            and
            * All compiled assets for the given parameters are identical between the previous and current scorecards.

        If the sameness of relevant compile jobs is not cached, sameness will be computed and added to the "CompileJobsAreIdentical" cache.
        """
        try:
            each_component_is_identical = all(
                self.compile_jobs_are_identical[
                    self.get_cache_key(
                        model_id, precision, path, device, component_name
                    )
                ]
                for component_name in component_names or [None]  # type: ignore[list-item]
            )
        except KeyError:
            # This set of parameters is missing from the cache, we need to add them.
            previous_compile_jobs = fetch_async_test_jobs(
                hub.JobType.COMPILE,
                model_id,
                precision,
                path,
                device,
                component_names,
                COMPILE_YAML_BASE,
            )

            current_compile_jobs = fetch_async_test_jobs(
                hub.JobType.COMPILE,
                model_id,
                precision,
                path,
                device,
                component_names,
            )

            if not current_compile_jobs or not previous_compile_jobs:
                each_component_is_identical = False
                # If any jobs fail for a model, we treat all jobs as non-identical
                for component in component_names or [None]:  # type: ignore[list-item]
                    self.compile_jobs_are_identical[
                        self.get_cache_key(model_id, precision, path, device, component)
                    ] = False
            else:
                each_component_is_identical = True
                for component in component_names or [None]:  # type: ignore[list-item]
                    is_identical = self.compile_jobs_are_same(
                        current_compile_jobs[component],
                        previous_compile_jobs[component],
                    )
                    self.compile_jobs_are_identical[
                        self.get_cache_key(model_id, precision, path, device, component)
                    ] = is_identical
                    each_component_is_identical = (
                        each_component_is_identical and is_identical
                    )

        return each_component_is_identical

    @staticmethod
    def get_cache_key(
        model_id: str,
        precision: Precision,
        path: ScorecardCompilePath | ScorecardProfilePath,
        device: ScorecardDevice,
        component: str | None = None,
    ) -> str:
        if isinstance(path, ScorecardProfilePath):
            path = path.compile_path
        if path.is_universal:
            device = cs_universal
        return get_async_job_cache_name(path, model_id, device, precision, component)

    @classmethod
    def compile_jobs_are_same(
        cls,
        current_compile_job: hub.CompileJob,
        previous_compile_job: hub.CompileJob,
    ) -> bool:
        """
        Compare the MD5 hashes of the compiled models for two jobs.

        Parameters
        ----------
        current_compile_job
            The current compile job.
        previous_compile_job
            The previous compile job.

        Returns
        -------
        jobs_are_same : bool
            True if the MD5 hashes of the compiled models for the two jobs are the same, False otherwise.
        """
        if previous_compile_job.get_status().failure:
            return False

        if "qnn_lib_aarch64_android" in current_compile_job.options:
            # shared libraries do have stable md5 hashes between hub jobs
            return False

        if current_compile_job.get_status().failure:
            return False

        if ToolVersions.from_job(previous_compile_job) != ToolVersions.from_job(
            current_compile_job
        ):
            return False

        # The temporary directory and all its contents will be automatically cleaned up when the 'with' block is exited
        with qaihm_temp_dir() as tmp_dir:
            current_model_file = cast(
                str,
                current_compile_job.download_target_model(
                    os.path.join(tmp_dir, "current_model")
                ),
            )
            previous_model_file = cast(
                str,
                previous_compile_job.download_target_model(
                    os.path.join(tmp_dir, "previous_model")
                ),
            )

            # ONNX zip files from hub don't have stable hashes,
            # but the model.onnx and model.data _do_ have stable hashes.
            # Unzip if necessary, then compare each model file.
            current_model_files: list[os.PathLike | str] = []
            previous_model_files: list[os.PathLike | str] = []
            for model_file, model_files in [
                (current_model_file, current_model_files),
                (previous_model_file, previous_model_files),
            ]:
                if model_file.endswith(".onnx.zip"):
                    model_files.extend(
                        [x for x in extract_onnx_zip(model_file) if os.path.exists(x)]
                    )
                else:
                    model_files.append(model_file)

            # Compare the MD5 hashes of the compiled models
            return all(
                file_hashes_are_identical(current_model, previous_model)
                for current_model, previous_model in zip(
                    current_model_files, previous_model_files, strict=False
                )
            )


def get_accuracy_csv_path(manual_path: str | None = None) -> Path | None:
    if manual_path:
        return Path(manual_path)
    artifacts_dir = get_artifacts_dir_opt()
    if artifacts_dir is None:
        return None
    for scorecard_file in artifacts_dir.iterdir():
        if scorecard_file.name.startswith("accuracy") and scorecard_file.name.endswith(
            ".csv"
        ):
            return scorecard_file
    return None


def get_scorecard_csv_path(manual_path: str | None = None) -> Path:
    if manual_path:
        return Path(manual_path)
    artifacts_dir = get_artifacts_dir_opt()
    assert artifacts_dir is not None
    for scorecard_file in artifacts_dir.iterdir():
        if scorecard_file.name.startswith(
            "scorecard-summary"
        ) and scorecard_file.name.endswith(".csv"):
            return scorecard_file
    raise ValueError("No perf data found.")


def get_aggreggated_results_csv_path(manual_path: str | None = None) -> Path:
    if manual_path:
        return Path(manual_path)
    artifacts_dir = get_artifacts_dir_opt()
    if artifacts_dir is not None:
        for subpath in artifacts_dir.iterdir():
            if subpath.name.startswith("Scorecard CSV"):
                return subpath / DEFAULT_AGGREGATED_CSV_NAME
        return artifacts_dir / DEFAULT_AGGREGATED_CSV_NAME
    return Path(f"build/{DEFAULT_AGGREGATED_CSV_NAME}")
