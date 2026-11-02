# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, final

from qai_hub import JobType

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import (
    ScorecardDevice,
    ScorecardProfilePath,
)
from qai_hub_models.scorecard.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.scorecard.execution_helpers import (
    get_enabled_paths_for_testing,
    get_evaluation_parameterized_pytest_config,
    get_profile_parameterized_pytest_config,
)
from qai_hub_models.scorecard.internal_config_yaml import (
    QAIHMModelInternalConfig,
    model_perf_component_key,
)
from qai_hub_models.scorecard.params import (
    ScExportTestParams,
    ScJobParams,
    ScorecardPathT,
)
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.results.chipset_helpers import (
    get_supported_devices,
    sorted_chipsets,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    LinkScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
)
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    PreQDQCompileScorecardJobYaml,
    ProfileScorecardJobYaml,
    QuantizeScorecardJobYaml,
)
from qai_hub_models.scorecard.static.model_config import ScorecardModelConfig
from qai_hub_models.scorecard.static.model_exec import (
    get_static_model_test_parameterizations,
)
from qai_hub_models.utils.export.result import ComponentGroup

# Maximum acceptable inference time (milliseconds).
# Above this inference time, a model will not be published.
MAX_ACCEPTABLE_INFERENCE_TIME_MS = 4000


@final
@dataclass
class ScorecardJobSummary(Generic[ScorecardPathT]):
    """
    Stores job in scorecard along with its prerequisite jobs.
    When all jobs are set, a single instance of this class is equivalent to a row in the final csv produced in the scorecard results.

    Each row of the table corresponds to a (model + component + graph name + runtime + precision + device).
    Each row has a unique profile / inference job, but prerequisite jobs may be shared between rows.
    """

    params: ScJobParams[ScorecardPathT]
    pre_qdq_onnx_compile_job: CompileScorecardJob | None = None
    quantize_job: QuantizeScorecardJob | None = None
    compile_job: CompileScorecardJob | None = None
    link_job: LinkScorecardJob | None = None
    profile_job: ProfileScorecardJob | None = None
    inference_job: InferenceScorecardJob | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.params.path, ScorecardProfilePath):
            raise TypeError(
                "Only profile scorecard paths are supported for run summaries."
            )
        if self.params.device is None:
            raise ValueError("A device must be defined for a run summary.")

    def __hash__(self) -> int:
        return hash(self.params.device_job_id)

    @staticmethod
    def from_params(
        params: ScJobParams,
        pre_qdq_jobs: PreQDQCompileScorecardJobYaml | None,
        quantize_jobs: QuantizeScorecardJobYaml | None,
        compile_jobs: CompileScorecardJobYaml | None,
        link_jobs: LinkScorecardJobYaml | None,
        profile_jobs: ProfileScorecardJobYaml | None,
        inference_jobs: InferenceScorecardJobYaml | None,
    ) -> ScorecardJobSummary:
        """
        Using the given job parameters, load all relevant cached jobs.
        If any YAML is None, the corresponding job type is not loaded (resolved to None).
        """
        return ScorecardJobSummary(
            params=params,
            pre_qdq_onnx_compile_job=pre_qdq_jobs.get_job(params)
            if (pre_qdq_jobs and params.has_quantize_job)
            else None,
            quantize_job=quantize_jobs.get_job(params)
            if (quantize_jobs and params.has_quantize_job)
            else None,
            compile_job=compile_jobs.get_job(params)
            if (compile_jobs and params.has_compile_job)
            else None,
            link_job=link_jobs.get_job(params)
            if (link_jobs and params.has_link_job)
            else None,
            profile_job=profile_jobs.get_job(params)
            if (profile_jobs and params.has_device_job)
            else None,
            inference_job=inference_jobs.get_job(params)
            if (inference_jobs and params.has_device_job)
            else None,
        )

    def add_to_perf(
        self: ScorecardJobSummary[ScorecardProfilePath],
        perf: QAIHMModelPerf,
        include_failures: bool = False,
    ) -> None:
        """Add the profile job for this export test to the given QAIHMModelPerf object."""
        params = self.params
        assert params.precision is not None
        assert params.device is not None

        if not include_failures and (
            not self.profile_job or not self.profile_job.success
        ):
            return

        if params.precision not in perf.precisions:
            perf.precisions[params.precision] = QAIHMModelPerf.PrecisionDetails()

        precision_details = perf.precisions[params.precision]
        component_id = model_perf_component_key(params.model_id, params.component)
        if component_id not in precision_details.components:
            # This field is set only when the parent precision is "mixed", since it is not otherwise
            # possible to decipher what precision was used for each component.
            component_precision: Precision | None = None
            if (
                self.params.precision in {Precision.mixed, Precision.mixed_with_float}
                and self.quantize_job is not None
            ):
                component_precision = Precision.from_quantize_job(self.quantize_job.job)
            precision_details.components[component_id] = (
                QAIHMModelPerf.ComponentDetails(precision=component_precision)
            )

        perf_metrics = precision_details.components[component_id].performance_metrics

        if params.device not in perf_metrics:
            perf_metrics[params.device] = {}
        perf_metrics[params.device][params.path] = (
            self.profile_job.performance_metrics
            if self.profile_job
            else QAIHMModelPerf.PerformanceDetails(job_id=None, job_status="Failed")
        )


@final
@dataclass
class ScorecardExportTestSummary:
    """
    Stores all jobs that run as a part of a single export test.
    A "single export test" is equal to a user running the export script (one model + runtime + precision + device).
    """

    params: ScExportTestParams[ScorecardProfilePath]
    job_summaries: list[ScorecardJobSummary[ScorecardProfilePath]]

    @staticmethod
    def from_params(
        params: ScExportTestParams[ScorecardProfilePath],
        pre_qdq_jobs: PreQDQCompileScorecardJobYaml | None,
        quantize_jobs: QuantizeScorecardJobYaml | None,
        compile_jobs: CompileScorecardJobYaml | None,
        link_jobs: LinkScorecardJobYaml | None,
        profile_jobs: ProfileScorecardJobYaml | None,
        inference_jobs: InferenceScorecardJobYaml | None,
    ) -> ScorecardExportTestSummary:
        """
        Using the given job parameters, load all relevant cached jobs.
        If any YAML is None, the corresponding job type is not loaded (resolved to None).
        """
        return ScorecardExportTestSummary(
            params,
            [
                ScorecardJobSummary.from_params(
                    pp,
                    pre_qdq_jobs,
                    quantize_jobs,
                    compile_jobs,
                    link_jobs,
                    profile_jobs,
                    inference_jobs,
                )
                for pp in params.all_device_job_params
            ],
        )

    def get_failure_reason(self) -> str | None:
        """Returns the reason this export test failed, or None if it succeeded."""
        too_slow_jobs: list[str] = []
        failing_jobs: list[str] = []
        has_missing_profile_job: bool = False
        for summary in self.job_summaries:
            if summary.profile_job:
                if not summary.profile_job.success:
                    failing_jobs.append(summary.profile_job.job_id)
                elif (
                    summary.profile_job.inference_time_milliseconds
                    > MAX_ACCEPTABLE_INFERENCE_TIME_MS
                ):
                    too_slow_jobs.append(summary.profile_job.job_id)
                continue
            if self.params.expects_device_job(summary.params.component):
                has_missing_profile_job = True

            if summary.link_job:
                if not summary.link_job.success:
                    failing_jobs.append(summary.link_job.job_id)
                continue

            if summary.compile_job:
                if not summary.compile_job.success:
                    failing_jobs.append(summary.compile_job.job_id)
                continue

            if summary.quantize_job:
                if not summary.quantize_job.success:
                    failing_jobs.append(summary.quantize_job.job_id)
                continue

            if summary.pre_qdq_onnx_compile_job:
                if not summary.pre_qdq_onnx_compile_job.success:
                    failing_jobs.append(summary.pre_qdq_onnx_compile_job.job_id)
                continue

        if failing_jobs:
            return "Failing jobs: " + " ".join(failing_jobs)
        if has_missing_profile_job:
            return "Some expected profile jobs were missing."
        if too_slow_jobs:
            return f"Profiling jobs slower than {MAX_ACCEPTABLE_INFERENCE_TIME_MS}ms: {' '.join(too_slow_jobs)}"
        return None

    @property
    def profiled_job_summaries(self) -> list[ScorecardJobSummary[ScorecardProfilePath]]:
        """The job summaries whose component is expected to have a profile job."""
        return [
            x
            for x in self.job_summaries
            if self.params.expects_device_job(x.params.component)
        ]

    @property
    def has_profile_failure(self) -> bool:
        """Returns true if any profile job failed or any expected profile job is missing in the job cache for this export test."""
        return any(
            x.profile_job is None or not x.profile_job.success
            for x in self.profiled_job_summaries
        )

    def add_to_perf(self, perf: QAIHMModelPerf, include_failures: bool = False) -> None:
        """Add all profile jobs from this export test to the given QAIHMModelPerf object."""
        assert self.params.device
        if not include_failures and self.has_profile_failure:
            return

        summaries = self.profiled_job_summaries
        if not summaries:
            return

        for summary in summaries:
            summary.add_to_perf(perf, include_failures)

        new_chipsets: set[str] = set()
        if self.params.device.available_in_hub:
            new_chipsets.update(
                self.params.device.supported_chipsets_for_paths([self.params.path])
            )
        elif device_details := DevicesAndChipsetsYaml.load().devices.get(
            self.params.device.reference_device_name
        ):
            new_chipsets.add(device_details.chipset)

        supported_chipsets = set(perf.supported_chipsets)
        if new_chipsets - supported_chipsets:
            chips = supported_chipsets.union(new_chipsets)
            perf.supported_chipsets = sorted_chipsets(chips)
            perf.supported_devices = get_supported_devices(chips)


@dataclass
class ModelTestConfig:
    """Stores what tests would have run for that model. We use that information to collect test results."""

    model_id: str
    component_names: list[str] | None
    graph_names: list[str] | None
    component_graph_names: ComponentGroup[list[str]] | None

    # Components the generated test_profile actually profiles. Narrower than
    # component_names for hybrid LLMs, whose backbone parts are measured end-to-end;
    # counting those as expected profile jobs would mark every export test failed.
    profile_component_names: list[str] | None

    profile_tests: list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]
    inference_tests: list[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]
    enabled_paths: dict[Precision, list[ScorecardProfilePath]]

    @staticmethod
    def from_recipe_model(
        manifest: QAIHMModelManifest,
        component_names: list[str] | None = None,
        graph_names: list[str] | None = None,
        component_graph_names: ComponentGroup[list[str]] | None = None,
    ) -> ModelTestConfig:
        """Load the test configuration for the given PyTorch recipe model."""
        assert manifest.id is not None
        model_id = manifest.id

        # Get enabled test paths for this model
        model_supported_paths = manifest.get_supported_paths_for_testing(
            only_include_passing=False
        )
        model_passing_paths = manifest.get_supported_paths_for_testing(
            only_include_passing=True
        )
        enabled_paths = get_enabled_paths_for_testing(
            model_id,
            model_supported_paths,
            model_passing_paths,
            ScorecardProfilePath,
            manifest.can_use_quantize_job,
        )

        scorecard_config = QAIHMModelInternalConfig.from_model(model_id)

        # is_llm must match what the generated test_profile ran with, or the scope
        # computed here covers devices the model never profiles on.
        is_llm = scorecard_config.is_llm
        profile_tests = get_profile_parameterized_pytest_config(
            model_id,
            model_supported_paths,
            model_passing_paths,
            manifest.can_use_quantize_job,
            is_llm=is_llm,
        )
        inference_tests = get_evaluation_parameterized_pytest_config(
            model_id,
            ScorecardDevice.get(manifest.default_device),
            model_supported_paths,
            model_passing_paths,
            manifest.can_use_quantize_job,
        )

        standalone_components = scorecard_config.standalone_components
        profile_component_names = component_names
        if standalone_components and component_names is not None:
            profile_component_names = [
                c for c in component_names if c in standalone_components
            ]

        return ModelTestConfig(
            model_id=model_id,
            component_names=component_names,
            graph_names=graph_names,
            component_graph_names=component_graph_names,
            profile_component_names=profile_component_names,
            profile_tests=profile_tests,
            inference_tests=inference_tests,
            enabled_paths=enabled_paths,
        )

    @staticmethod
    def from_static_model(
        model_info: ScorecardModelConfig,
    ) -> ModelTestConfig:
        """Load the test configuration for the given static model."""
        return ModelTestConfig(
            model_id=model_info.id,
            component_names=None,
            component_graph_names=None,
            graph_names=None,
            profile_component_names=None,
            profile_tests=get_static_model_test_parameterizations(
                model_info.id,
                JobType.PROFILE,
                ScorecardProfilePath,
                model_info.precision,
                model_info.devices,
                model_info.enabled_profile_runtimes,
            ),
            inference_tests=get_static_model_test_parameterizations(
                model_info.id,
                JobType.INFERENCE,
                ScorecardProfilePath,
                model_info.precision,
                [model_info.eval_device],
                model_info.enabled_profile_runtimes,
            ),
            enabled_paths={},
        )

    def get_all_export_params(self) -> list[ScExportTestParams]:
        """Get the list of export tests that would run given this test configuration."""
        all_paramaterizations = {*self.inference_tests, *self.profile_tests}
        return [
            ScExportTestParams(
                self.model_id,
                pp[1],
                pp[0],
                pp[2],
                self.component_names,
                self.graph_names,
                self.component_graph_names,
                profile_component_names=self.profile_component_names,
            )
            for pp in all_paramaterizations
        ]

    def get_all_export_test_summaries(
        self,
        pre_qdq_jobs: PreQDQCompileScorecardJobYaml | None,
        quantize_jobs: QuantizeScorecardJobYaml | None,
        compile_jobs: CompileScorecardJobYaml | None,
        link_jobs: LinkScorecardJobYaml | None,
        profile_jobs: ProfileScorecardJobYaml | None,
        inference_jobs: InferenceScorecardJobYaml | None,
    ) -> list[ScorecardExportTestSummary]:
        """Get the set of export tests that would run given this test configuration, and load all cached jobs that apply to those tests."""
        return [
            ScorecardExportTestSummary.from_params(
                pp,
                pre_qdq_jobs,
                quantize_jobs,
                compile_jobs,
                link_jobs,
                profile_jobs,
                inference_jobs,
            )
            for pp in self.get_all_export_params()
        ]
