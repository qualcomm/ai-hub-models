# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import ruamel.yaml
from pydantic import Field
from qai_hub import JobType

from qai_hub_models import Precision
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard.artifacts import ScorecardArtifact, ScorecardYamlFile
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.errors import CachedScorecardJobError
from qai_hub_models.scorecard.params import (
    ScExportTestParams,
    ScJobParams,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.release_assets_yaml import (
    QAIHMModelReleaseAssets,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    JobTypeVar,
    LinkScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
    ScorecardJobTypeVar,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig, sorted_mappings
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)

ScorecardJobYamlTypeVar = TypeVar("ScorecardJobYamlTypeVar", bound="ScorecardJobYaml")


@dataclass(frozen=True)
class ToolVersionChange:
    path: str
    tool: str
    previous: str
    current: str


# Schema for sdk versions dumped to Hugging Face / Scorecard Intermediates in YAML format.
class ToolVersionsByPathYaml(BaseQAIHMConfig):
    tool_versions: dict[ScorecardProfilePath, ToolVersions] = Field(
        default_factory=dict
    )

    @staticmethod
    def from_profile_paths(
        paths: list[ScorecardProfilePath] | None = None,
    ) -> ToolVersionsByPathYaml:
        """
        Get a tool versions YAML object, with all paths in the list populated with tool versions.
        This will fetch versions for AI Hub Workbench deployment used by scorecard (set by envvars).

        If paths is None, populates all enabled scorecard profile paths.
        """
        out = ToolVersionsByPathYaml()
        for path in paths or ScorecardProfilePath:
            out.tool_versions[path] = path.tool_versions
        return out

    @staticmethod
    def from_dir(
        dirpath: str | os.PathLike, filename: str = "tool-versions.yaml"
    ) -> ToolVersionsByPathYaml:
        return ToolVersionsByPathYaml.from_yaml(
            Path(dirpath) / filename,
            create_empty_if_no_file=True,
        )

    def to_dir(
        self,
        dirpath: str | os.PathLike,
        filename: str = "tool-versions.yaml",
    ) -> bool:
        return self.to_yaml(Path(dirpath) / filename, write_if_empty=False)

    def diff(self, previous: ToolVersionsByPathYaml) -> list[ToolVersionChange]:
        """
        Compute per-path, per-tool version changes between this (current) and a previous yaml.

        Returns one ToolVersionChange per differing field. Missing values are rendered as "N/A".
        """
        rows: list[ToolVersionChange] = []
        all_paths = sorted(
            set(self.tool_versions) | set(previous.tool_versions),
            key=lambda p: p.name,
        )
        for path in all_paths:
            curr = self.tool_versions.get(path, ToolVersions())
            prev = previous.tool_versions.get(path, ToolVersions())
            for field_name in ToolVersions.model_fields:
                curr_val = getattr(curr, field_name)
                prev_val = getattr(prev, field_name)
                if curr_val == prev_val:
                    continue
                rows.append(
                    ToolVersionChange(
                        path=path.name,
                        tool=field_name,
                        previous=_format_tool_version(prev_val),
                        current=_format_tool_version(curr_val),
                    )
                )
        return rows


def _format_tool_version(value: object) -> str:
    if value is None:
        return "N/A"
    if hasattr(value, "full_version_with_flavor"):
        return value.full_version_with_flavor
    return str(value)


class ScorecardJobYaml(ScorecardYamlFile[str], Generic[ScorecardJobTypeVar]):
    SCORECARD_JOB_TYPE: type[ScorecardJobTypeVar]

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        path: str | os.PathLike | None = None,
    ) -> None:
        super().__init__(mapping, path)
        # ScorecardJob classes are expensive to create
        # (workbench API calls), so we cache them.
        self.job_cache: dict[str, ScorecardJobTypeVar] = {}

    def to_file(self, path: str | Path | None = None, append: bool = False) -> None:
        path = path or self.path
        assert path is not None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if len(self.mapping) > 0:
            with open(path, "a" if append else "w") as yaml_file:
                ruamel.yaml.YAML().dump(sorted_mappings(self.mapping), yaml_file)
        elif not append:
            Path(path).touch()

    def get_job_key(self, params: ScJobParams) -> str:
        return params.job_id(self.SCORECARD_JOB_TYPE.job_type)

    def set_job_id(self, job_id: str, params: ScJobParams) -> None:
        """
        Set the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

        Parameters
        ----------
        job_id
            Job ID to associate with the other parameters in the YAML
        params
            Job identification parameters
        """
        self.mapping[self.get_job_key(params)] = job_id

    def get_job_id(self, params: ScJobParams) -> str | None:
        """Return the raw job ID string for *params*, or None if not present."""
        return self.mapping.get(self.get_job_key(params))

    def update(self, other: ScorecardJobYaml) -> None:
        """Merge the other YAML into this YAML, overwriting any existing jobs with the same job name"""
        if type(other) is not type(self):
            raise ValueError(
                f"Cannot merge scorecard YAMLS of types {type(other)} and {type(self)}"
            )
        self.mapping.update(other.mapping)

    def drop_in_scope(self, params: list[ScExportTestParams]) -> None:
        """Delete only the job keys the given export-test params would write."""
        for p in params:
            if not self._params_apply(p):
                continue
            for job_params in self._get_all_job_params(p):
                self.mapping.pop(self.get_job_key(job_params), None)

    def _params_apply(self, params: ScExportTestParams) -> bool:
        """Whether *params* identifies any jobs stored in this yaml."""
        return params.applies_to_job_type(self.SCORECARD_JOB_TYPE.job_type)

    def get_job(
        self,
        params: ScJobParams,
        wait_for_job: bool = True,
        wait_for_job_max_seconds: int | None = None,
        raise_if_not_successful: bool = False,
    ) -> ScorecardJobTypeVar | None:
        """
        Get the scorecard job from the YAML associated with these parameters.

        Parameters
        ----------
        params
            Job identification parameters
        wait_for_job
            If false, running jobs are treated like they were "skipped"
        wait_for_job_max_seconds
            Allow the job this many seconds after creation to complete
        raise_if_not_successful
            If true, raise an error if the job is not successful

        Returns
        -------
        job : ScorecardJobTypeVar | None
            The scorecard job matching these parameters, or None if the job does not exist.
        """
        # Get the job.
        key = self.get_job_key(params)
        job = self.job_cache.get(key)
        if job is None:
            if job_id := self.mapping.get(key):
                job = self.SCORECARD_JOB_TYPE(job_id)
                # ScorecardJob classes are expensive to create
                # (workbench API calls), so we cache them.
                self.job_cache[key] = job
            else:
                return None

        # Wait for the job to finish and cache the results.
        if wait_for_job:
            job.wait(wait_for_job_max_seconds)
        else:
            job.cache_results()

        # Verify the job succeeded.
        if raise_if_not_successful and not job.success:
            if job.running:
                error_str = f"still running after max allowed job duration of {wait_for_job_max_seconds / 60 if wait_for_job_max_seconds else '__'} minutes"
            else:
                error_str = job.job_status

            raise CachedScorecardJobError(
                params.str_with_description(
                    f"Prerequisite {job.job._job_type.display_name.title()} job {error_str}: {job.job.url}"
                )
            )

        return job

    def _get_all_job_params(self, params: ScExportTestParams) -> list[ScJobParams]:
        return params.all_job_params(self.SCORECARD_JOB_TYPE.job_type)

    def get_all_jobs(
        self,
        params: ScExportTestParams,
        wait_for_job: bool = True,
        wait_for_job_max_seconds: int | None = None,
        raise_if_not_successful: bool = False,
        raise_if_jobs_are_missing: bool = False,
    ) -> dict[ScJobParams, ScorecardJobTypeVar | None]:
        """Get all cached jobs that should exist for the given test paramaters. If a job should exist but is not in the cache, it is returned as None."""
        all_jobs = {
            pp: self.get_job(
                pp, wait_for_job, wait_for_job_max_seconds, raise_if_not_successful
            )
            for pp in self._get_all_job_params(params)
        }

        if raise_if_jobs_are_missing and sum(
            x is not None for x in all_jobs.values()
        ) != len(all_jobs):
            raise CachedScorecardJobError(
                params.str_with_description(
                    f"Could not find all cached {self.SCORECARD_JOB_TYPE.job_type.name} jobs."
                )
            )

        return all_jobs

    def update_from_export_output(
        self,
        export_output: None
        | JobTypeVar
        | MultiGraphGroup[JobTypeVar]
        | ComponentGroup[JobTypeVar]
        | MultiGraphComponentGroup[JobTypeVar],
        test_params: ScExportTestParams,
    ) -> None:
        """From the output of an export pipeline step, populate this cache."""
        if export_output is None:
            raise ValueError("Export output is missing.")

        if isinstance(export_output, MultiGraphComponentGroup):
            for (
                component,
                graph_name,
            ), job in export_output.items():
                self.set_job_id(
                    job.job_id,
                    ScJobParams(
                        test_params.model_id,
                        test_params.path,
                        test_params.precision,
                        test_params.device,
                        component,
                        graph_name=graph_name,
                    ),
                )
        elif isinstance(export_output, ComponentGroup):
            for component, job in export_output.items():
                self.set_job_id(
                    job.job_id,
                    ScJobParams(
                        test_params.model_id,
                        test_params.path,
                        test_params.precision,
                        test_params.device,
                        component,
                        graph_name=None,
                    ),
                )
        elif isinstance(export_output, MultiGraphGroup):
            for graph_name, job in export_output.items():
                self.set_job_id(
                    job.job_id,
                    ScJobParams(
                        test_params.model_id,
                        test_params.path,
                        test_params.precision,
                        test_params.device,
                        component=None,
                        graph_name=graph_name,
                    ),
                )
        else:
            self.set_job_id(
                export_output.job_id,
                ScJobParams(
                    test_params.model_id,
                    test_params.path,
                    test_params.precision,
                    test_params.device,
                    component=None,
                    graph_name=None,
                ),
            )

    def get_export_output(
        self,
        test_params: ScExportTestParams,
        wait_for_job: bool = True,
        wait_for_job_max_seconds: int | None = None,
        raise_if_not_successful: bool = True,
        raise_if_jobs_are_missing: bool = True,
    ) -> (
        JobTypeVar
        | MultiGraphGroup[JobTypeVar]
        | ComponentGroup[JobTypeVar]
        | MultiGraphComponentGroup[JobTypeVar]
        | None
    ):
        """Load the output of an export pipeline step that would have generated this cache."""
        all_jobs = self.get_all_jobs(
            test_params,
            wait_for_job,
            wait_for_job_max_seconds,
            raise_if_not_successful,
            raise_if_jobs_are_missing,
        )
        has_components = test_params.component_names is not None
        has_graph_names = self.SCORECARD_JOB_TYPE.job_type not in {
            JobType.QUANTIZE,
            JobType.LINK,
        } and (
            test_params.graph_names is not None
            or test_params.component_graph_names is not None
        )

        if has_graph_names:
            if has_components:
                mgcg_components: dict[tuple[str, str], JobTypeVar] = {}
                for job_params, sc_job in all_jobs.items():
                    assert job_params.component is not None
                    assert job_params.graph_name is not None
                    if sc_job is None:
                        continue
                    mgcg_components[(job_params.component, job_params.graph_name)] = (
                        sc_job.job
                    )
                return MultiGraphComponentGroup(mgcg_components)
            out_gn: dict[str, JobTypeVar] = {}
            for job_params, sc_job in all_jobs.items():
                assert (
                    job_params.component is None and job_params.graph_name is not None
                )
                if sc_job is not None:
                    out_gn[job_params.graph_name] = sc_job.job
            return MultiGraphGroup(out_gn)

        if has_components:
            out_comp: dict[str, JobTypeVar] = {}
            for job_params, sc_job in all_jobs.items():
                assert (
                    job_params.component is not None and job_params.graph_name is None
                )
                if sc_job is not None:
                    out_comp[job_params.component] = sc_job.job
            return ComponentGroup(out_comp)

        if len(all_jobs) == 0:
            return None
        assert len(all_jobs) == 1
        sc_job = next(iter(all_jobs.values()))
        assert sc_job is not None
        return sc_job.job


class PreQDQCompileScorecardJobYaml(ScorecardJobYaml[CompileScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.COMPILE_YAML
    SCORECARD_JOB_TYPE = CompileScorecardJob

    def get_job_key(self, params: ScJobParams) -> str:
        return params.pre_quantize_compile_job_id

    def _get_all_job_params(self, params: ScExportTestParams) -> list[ScJobParams]:
        return params.all_pre_qdq_compile_job_params

    def _params_apply(self, params: ScExportTestParams) -> bool:
        # Pre-QDQ compile jobs only exist along the quantization path.
        return params.precision is None or params.precision != Precision.float


class QuantizeScorecardJobYaml(ScorecardJobYaml[QuantizeScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.QUANTIZE_YAML
    SCORECARD_JOB_TYPE = QuantizeScorecardJob


class CompileScorecardJobYaml(ScorecardJobYaml[CompileScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.COMPILE_YAML
    SCORECARD_JOB_TYPE = CompileScorecardJob


class LinkScorecardJobYaml(ScorecardJobYaml[LinkScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.LINK_YAML
    SCORECARD_JOB_TYPE = LinkScorecardJob


class ProfileScorecardJobYaml(ScorecardJobYaml[ProfileScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.PROFILE_YAML
    SCORECARD_JOB_TYPE = ProfileScorecardJob


class InferenceScorecardJobYaml(ScorecardJobYaml[InferenceScorecardJob]):
    ARTIFACT_TYPE = ScorecardArtifact.INFERENCE_YAML
    SCORECARD_JOB_TYPE = InferenceScorecardJob


class ComponentNamesYaml(ScorecardYamlFile[list[str]]):
    """Maps model_id -> list of component names."""

    ARTIFACT_TYPE = ScorecardArtifact.COMPONENT_NAMES

    def set(self, model_id: str, component_names: list[str]) -> None:
        self.mapping[model_id] = component_names

    def get(self, model_id: str) -> list[str] | None:
        return self.mapping.get(model_id)


class GraphNamesYaml(ScorecardYamlFile[list[str]]):
    """Maps model_id_component_name -> list of graph names."""

    ARTIFACT_TYPE = ScorecardArtifact.GRAPH_NAMES

    @staticmethod
    def _key(model_id: str, component_name: str | None = None) -> str:
        return f"{model_id}_{component_name}" if component_name else model_id

    def set(self, model_id: str, component_name: str, graph_names: list[str]) -> None:
        self.mapping[self._key(model_id, component_name)] = graph_names

    def get(self, model_id: str, component_name: str | None = None) -> list[str] | None:
        return self.mapping.get(self._key(model_id, component_name))


def get_model_component_and_graph_names(
    model_id: str,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
) -> tuple[list[str] | None, list[str] | None, ComponentGroup[list[str]] | None]:
    """
    Extract component names, graph names, and component-graph-names mapping for a model.

    Parameters
    ----------
    model_id
        Model identifier.
    component_names_yaml
        YAML containing component names for each model.
    graph_names_yaml
        YAML containing graph names for each model component.

    Returns
    -------
    component_names : list[str] | None
        List of component names, or None if this model has no components.
    graph_names : list[str] | None
        List of graph names for a single-component model, or None.
    component_graph_names : ComponentGroup[list[str]] | None
        Component-grouped graph names for multi-component models, or None.
    """
    component_names = component_names_yaml.get(model_id)

    graph_names: list[str] | None = None
    component_graph_names: ComponentGroup[list[str]] | None = None
    if component_names is not None:
        cgn: dict[str, list[str]] = {}
        for comp_name in component_names:
            gn = graph_names_yaml.get(model_id, comp_name)
            if gn is not None:
                cgn[comp_name] = gn
        if cgn:
            component_graph_names = ComponentGroup(cgn)
    else:
        graph_names = graph_names_yaml.get(model_id, model_id)

    return component_names, graph_names, component_graph_names


class ScorecardAssetYaml(BaseQAIHMConfig):
    models: dict[str, QAIHMModelReleaseAssets] = Field(default_factory=dict)

    def add_asset(
        self,
        details: QAIHMModelReleaseAssets.AssetDetails,
        model_id: str,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
    ) -> None:
        if model_id not in self.models:
            self.models[model_id] = QAIHMModelReleaseAssets()
        self.models[model_id].add_asset(
            details,
            precision,
            device.chipset if path.runtime.is_aot_compiled else None,
            path,
        )

    def get_asset(
        self,
        model_id: str,
        precision: Precision,
        device: ScorecardDevice,
        path: ScorecardProfilePath,
    ) -> QAIHMModelReleaseAssets.AssetDetails | None:
        if model_id not in self.models:
            return None
        return self.models[model_id].get_asset(
            precision,
            device.chipset if path.runtime.is_aot_compiled else None,
            path,
        )
