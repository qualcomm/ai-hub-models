# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import datetime
from functools import cached_property
from typing import Any, Generic, TypeVar, cast

import qai_hub as hub
from qai_hub.public_rest_api import DatasetEntries
from qai_hub_models_cli.proto import model_perf_pb2
from qai_hub_models_cli.proto.shared import range_pb2

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard.execution_helpers import wait_for_prerequisite_job

JobTypeVar = TypeVar(
    "JobTypeVar",
    hub.CompileJob,
    hub.LinkJob,
    hub.InferenceJob,
    hub.QuantizeJob,
)
ScorecardJobTypeVar = TypeVar("ScorecardJobTypeVar", bound="ScorecardJob")


def _primary_compute_unit(
    layer_counts: model_perf_pb2.ModelPerf.PerfMetrics.LayerCounts,
) -> str:
    """The compute unit running the most layers ("null" if there are none)."""
    if layer_counts.npu == 0 and layer_counts.gpu == 0 and layer_counts.cpu == 0:
        return "null"
    most = max(layer_counts.cpu, layer_counts.gpu, layer_counts.npu)
    if most == layer_counts.npu:
        return "NPU"
    if most == layer_counts.gpu:
        return "GPU"
    return "CPU"


"""
The purpose of this class is to fetch & cache all information related to a workbench job.

This allows us to fetch all job information at once, to avoid expensive sequential
workbench API calls when doing analysis on the results.
"""


class ScorecardJob(Generic[JobTypeVar]):
    job_type_class: type[JobTypeVar]
    job_type: hub.JobType

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def wait(self, max_wait_seconds: int | None = None) -> hub.JobStatus:
        s = wait_for_prerequisite_job(self.job, max_wait_seconds)
        self.cache_results()
        return s

    def cache_results(self) -> None:
        # Verify job is the correct type
        if not isinstance(self.job, self.job_type_class):
            raise TypeError(
                f"Job {self.job.job_id}({self.job.name}) is {type(self.job)}. Expected {self.job_type_class.__name__}"
            )

    @cached_property
    def job(self) -> JobTypeVar:
        """
        Get the AI Hub Workbench Job.
        Waits for completion if necessary.
        """
        if not self.job_id:
            raise ValueError("No Job ID")
        return cast(JobTypeVar, hub.get_job(self.job_id))

    @cached_property
    def running(self) -> bool:
        return self._job_status.running

    @cached_property
    def failed(self) -> bool:
        return self._job_status.failure

    @cached_property
    def success(self) -> bool:
        return self._job_status.success

    @cached_property
    def status_message(self) -> str | None:
        return self._job_status.message

    @cached_property
    def _job_status(self) -> hub.JobStatus:
        """Get the job status of the profile job."""
        if self.job_id:
            return self.job.get_status()
        raise ValueError("Can't get status without a job ID.")

    @cached_property
    def job_status(self) -> str:
        """Get the job status of the profile job."""
        if self._job_status.success:
            return "Passed"
        if self._job_status.failure:
            return "Failed"
        return "Skipped"

    @cached_property
    def hub_device(self) -> hub.Device:
        if isinstance(self.job, (hub.CompileJob, hub.LinkJob, hub.InferenceJob)):
            return self.job.device
        raise NotImplementedError("Device is not applicable for this job type")

    @cached_property
    def tool_versions(self) -> ToolVersions:
        return ToolVersions.from_job(self.job, parse_version_tags=True)

    @cached_property
    def date(self) -> datetime.datetime | None:
        if self.job is None:
            return None
        return self.job.date


class QuantizeScorecardJob(ScorecardJob[hub.QuantizeJob]):
    job_type_class = hub.QuantizeJob
    job_type = hub.JobType.QUANTIZE


class CompileScorecardJob(ScorecardJob[hub.CompileJob]):
    job_type_class = hub.CompileJob
    job_type = hub.JobType.COMPILE


class LinkScorecardJob(ScorecardJob[hub.LinkJob]):
    job_type_class = hub.LinkJob
    job_type = hub.JobType.LINK


class InferenceScorecardJob(ScorecardJob[hub.InferenceJob]):
    job_type_class = hub.InferenceJob
    job_type = hub.JobType.INFERENCE

    @property
    def input_dataset(self) -> DatasetEntries:
        """Input dataset."""
        if self.job.inputs is None:
            raise ValueError("Job has no input dataset.")
        return cast(DatasetEntries, self.job.inputs.download())

    @property
    def output_dataset(self) -> DatasetEntries:
        """Output dataset."""
        if not self.success:
            raise ValueError("Can't get output dataset if job did not succeed.")
        return cast(DatasetEntries, self.job.download_output_data())

    @cached_property
    def profile_results(self) -> dict[str, Any]:
        """Profile results from the job."""
        if self.success:
            profile = self.job.download_profile()
            assert isinstance(profile, dict)
            return profile
        raise ValueError("Can't get profile results if job did not succeed.")

    @cached_property
    def inference_time_milliseconds(self) -> float:
        """Get the inference time from the job's profile results."""
        return float(
            self.profile_results["execution_summary"]["estimated_inference_time"] / 1000
        )

    @cached_property
    def first_load_time_milliseconds(self) -> float:
        """Get the first load time from the job's profile results."""
        return float(
            self.profile_results["execution_summary"]["first_load_time"] / 1000
        )

    @cached_property
    def warm_load_time_milliseconds(self) -> float:
        """Get the warm load time from the job's profile results."""
        return float(self.profile_results["execution_summary"]["warm_load_time"] / 1000)

    @cached_property
    def layer_counts(self) -> model_perf_pb2.ModelPerf.PerfMetrics.LayerCounts:
        """Count layers per compute unit."""

        def _count_unit(unit: str) -> int:
            return sum(
                1
                for detail in self.profile_results["execution_detail"]
                if detail["compute_unit"] == unit
            )

        npu, gpu, cpu = _count_unit("NPU"), _count_unit("GPU"), _count_unit("CPU")
        return model_perf_pb2.ModelPerf.PerfMetrics.LayerCounts(
            total=npu + gpu + cpu, npu=npu, gpu=gpu, cpu=cpu
        )

    @cached_property
    def estimated_peak_memory_range_mb(self) -> range_pb2.IntRange:
        """Get the estimated peak memory range."""
        low, high = self.profile_results["execution_summary"][
            "inference_memory_peak_range"
        ]
        bytes_per_mb = 1 << 20
        return range_pb2.IntRange(
            min=round(low / bytes_per_mb), max=round(high / bytes_per_mb)
        )

    @cached_property
    def performance_metrics(self) -> model_perf_pb2.ModelPerf.PerformanceDetails:
        details = model_perf_pb2.ModelPerf.PerformanceDetails(
            tool_versions=self.tool_versions.to_proto(),
        )
        if self.job_id is not None or self.job_status is not None:
            details.profile_job.CopyFrom(
                model_perf_pb2.ModelPerf.ProfileJob(
                    id=self.job_id or "", status=self.job_status or ""
                )
            )
        if self.success:
            layer_counts = self.layer_counts
            details.metrics.CopyFrom(
                model_perf_pb2.ModelPerf.PerfMetrics(
                    inference_time_milliseconds=self.inference_time_milliseconds,
                    estimated_peak_memory_range_mb=self.estimated_peak_memory_range_mb,
                    primary_compute_unit=_primary_compute_unit(layer_counts),
                    layer_counts=layer_counts,
                )
            )
        return details
