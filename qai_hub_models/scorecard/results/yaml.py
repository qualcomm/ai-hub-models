# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, Literal, TypeVar, overload

import qai_hub as hub
import ruamel.yaml
from pydantic import Field
from typing_extensions import Self

from qai_hub_models.configs.release_assets_yaml import (
    QAIHMModelReleaseAssets,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision
from qai_hub_models.scorecard.device import ScorecardDevice, cs_universal
from qai_hub_models.scorecard.execution_helpers import (
    for_each_scorecard_path_and_device,
    get_async_job_cache_name,
)
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.performance_summary import (
    ModelCompileSummary,
    ModelInferenceSummary,
    ModelPerfSummary,
    ModelQuantizeSummary,
    ModelSummaryTypeVar,
)
from qai_hub_models.scorecard.results.scorecard_job import (
    CompileScorecardJob,
    InferenceScorecardJob,
    ProfileScorecardJob,
    QuantizeScorecardJob,
    ScorecardJobTypeVar,
    ScorecardPathOrNoneTypeVar,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT

DEFAULT_TOOL_VERSIONS_YAML_FILE_NAME = "tool-versions.yaml"
INTERMEDIATES_DIR = QAIHM_PACKAGE_ROOT / "scorecard" / "intermediates"
ENVIRONMENT_ENV_BASE = INTERMEDIATES_DIR / "environment.env"
QUANTIZE_YAML_BASE = INTERMEDIATES_DIR / "quantize-jobs.yaml"
COMPILE_YAML_BASE = INTERMEDIATES_DIR / "compile-jobs.yaml"
PROFILE_YAML_BASE = INTERMEDIATES_DIR / "profile-jobs.yaml"
INFERENCE_YAML_BASE = INTERMEDIATES_DIR / "inference-jobs.yaml"
TOOL_VERSIONS_BASE = INTERMEDIATES_DIR / DEFAULT_TOOL_VERSIONS_YAML_FILE_NAME
ACCURACY_CSV_BASE = INTERMEDIATES_DIR / "accuracy.csv"
RELEASE_CSV_BASE = INTERMEDIATES_DIR / "release-assets.yaml"
DATASETS_BASE = INTERMEDIATES_DIR / "dataset-ids.yaml"
ScorecardJobYamlTypeVar = TypeVar("ScorecardJobYamlTypeVar", bound="ScorecardJobYaml")


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
        paths = paths or ScorecardProfilePath.all_paths(enabled=True)
        out = ToolVersionsByPathYaml()
        for path in paths:
            out.tool_versions[path] = path.tool_versions
        return out

    @staticmethod
    def from_dir(
        dirpath: str | os.PathLike, filename: str = DEFAULT_TOOL_VERSIONS_YAML_FILE_NAME
    ) -> ToolVersionsByPathYaml:
        return ToolVersionsByPathYaml.from_yaml(
            Path(dirpath) / filename,
            create_empty_if_no_file=True,
        )

    def to_dir(
        self,
        dirpath: str | os.PathLike,
        filename: str = DEFAULT_TOOL_VERSIONS_YAML_FILE_NAME,
    ) -> bool:
        return self.to_yaml(Path(dirpath) / filename, write_if_empty=False)


class ScorecardJobYaml(
    Generic[ScorecardJobTypeVar, ScorecardPathOrNoneTypeVar, ModelSummaryTypeVar]
):
    scorecard_job_type: type[ScorecardJobTypeVar]
    scorecard_path_type: type[ScorecardPathOrNoneTypeVar]
    scorecard_model_summary_type: type[ModelSummaryTypeVar]

    def __init__(self, job_id_mapping: dict[str, str] | None = None) -> None:
        self.job_id_mapping = job_id_mapping or {}

    @classmethod
    def from_file(cls, config_path: str | Path) -> Self:
        """Read yaml files."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"File not found with job ids at {config_path}")

        yaml = ruamel.yaml.YAML()
        with open(config_path) as file:
            return cls(yaml.load(file))

    def to_file(self, path: str | Path, append: bool = False) -> None:
        if len(self.job_id_mapping) > 0:
            with open(path, "a" if append else "w") as yaml_file:
                ruamel.yaml.YAML().dump(self.job_id_mapping, yaml_file)
        elif not append:
            # If the dict is empty, ruamel dumps "{}" (which is not YAML) and breaks the file
            Path(path).touch()

    def clear_jobs(self, model_id: str | None = None) -> None:
        if not model_id:
            self.job_id_mapping.clear()
        else:
            # find jobs to delete
            # catch "model", ignore "model_quantized"
            keys_to_delete = [
                key
                for key in self.job_id_mapping
                if (
                    key.startswith(f"{model_id}_")
                    and not key.startswith(f"{model_id}_quantized_")
                )
            ]

            # Delete keys
            for key in keys_to_delete:
                del self.job_id_mapping[key]

    def get_job_id(
        self,
        path: ScorecardPathOrNoneTypeVar,
        model_id: str,
        device: ScorecardDevice,
        precision: Precision = Precision.float,
        component: str | None = None,
    ) -> str | None:
        """
        Get the ID of this job in the YAML that stores asyncronously-ran scorecard jobs.
        Returns None if job does not exist.

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
        job_id : str | None
            The job ID if found, None otherwise.
        """
        return self.job_id_mapping.get(
            get_async_job_cache_name(
                path,
                model_id,
                device.mirror_device or device,
                precision,
                component,
            )
        )

    def set_job_id(
        self,
        job_id: str,
        path: ScorecardPathOrNoneTypeVar,
        model_id: str,
        device: ScorecardDevice,
        precision: Precision = Precision.float,
        component: str | None = None,
    ) -> None:
        """
        Set the key for this job in the YAML that stores asyncronously-ran scorecard jobs.

        Parameters
        ----------
        job_id
            Job ID to associate with the other parameters in the YAML
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
        """
        self.job_id_mapping[
            get_async_job_cache_name(path, model_id, device, precision, component)
        ] = job_id

    def update(self, other: ScorecardJobYaml) -> None:
        """Merge the other YAML into this YAML, overwriting any existing jobs with the same job name"""
        if type(other) is not type(self):
            raise ValueError(
                f"Cannot merge scorecard YAMLS of types {type(other)} and {type(self)}"
            )
        self.job_id_mapping.update(other.job_id_mapping)

    def get_job(
        self,
        path: ScorecardPathOrNoneTypeVar,
        model_id: str,
        device: ScorecardDevice,
        precision: Precision = Precision.float,
        component: str | None = None,
        wait_for_job: bool = True,
        wait_for_max_job_duration: int | None = None,
    ) -> ScorecardJobTypeVar:
        """
        Get the scorecard job from the YAML associated with these parameters.

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
        wait_for_job
            If false, running jobs are treated like they were "skipped"
        wait_for_max_job_duration
            Allow the job this many seconds after creation to complete

        Returns
        -------
        job : ScorecardJobTypeVar
            The scorecard job matching these parameters.
        """
        job_id = self.get_job_id(
            path,
            model_id,
            device,
            precision,
            component,
        )
        return self.scorecard_job_type(
            component or model_id,
            precision,
            job_id,
            device,
            wait_for_job,
            wait_for_max_job_duration,
            path,  # type: ignore[arg-type]
        )

    def get_all_jobs(
        self,
        model_id: str,
        precisions: list[Precision] | None = None,
        devices: list[ScorecardDevice] | None = None,
        components: Iterable[str] | None = None,
    ) -> list[ScorecardJobTypeVar]:
        """Get all jobs in this YAML related to the provided model."""
        if precisions is None:
            precisions = [Precision.float]
        model_runs: list[ScorecardJobTypeVar] = []
        for component in components or [None]:  # type: ignore[list-item]

            def create_job(
                precision: Precision,
                path: ScorecardPathOrNoneTypeVar,
                device: ScorecardDevice,
            ) -> None:
                model_runs.append(
                    self.get_job(path, model_id, device, precision, component or None)  # noqa: B023
                )

            for_each_scorecard_path_and_device(
                self.__class__.scorecard_path_type,
                create_job,
                precisions,
                include_mirror_devices=True,
                include_devices=devices,
            )

        return model_runs

    def summary_from_model(
        self,
        model_id: str,
        precisions: list[Precision] | None = None,
        devices: list[ScorecardDevice] | None = None,
        components: Iterable[str] | None = None,
    ) -> ModelSummaryTypeVar:
        """Creates a summary of all jobs related to the given model."""
        if precisions is None:
            precisions = [Precision.float]
        runs = self.get_all_jobs(model_id, precisions, devices, components)
        return self.scorecard_model_summary_type.from_runs(model_id, runs, components)  # type: ignore[arg-type,return-value]


class QuantizeScorecardJobYaml(
    ScorecardJobYaml[QuantizeScorecardJob, None, ModelQuantizeSummary]
):
    scorecard_job_type = QuantizeScorecardJob
    scorecard_path_type = type(None)
    scorecard_model_summary_type = ModelQuantizeSummary

    def get_job_id(
        self,
        path: ScorecardPathOrNoneTypeVar,
        model_id: str,
        device: ScorecardDevice,
        precision: Precision = Precision.float,
        component: str | None = None,
    ) -> str | None:
        return self.job_id_mapping.get(
            get_async_job_cache_name(None, model_id, cs_universal, precision, component)
        )

    def set_job_id(
        self,
        job_id: str,
        path: ScorecardPathOrNoneTypeVar,
        model_id: str,
        device: ScorecardDevice,
        precision: Precision = Precision.float,
        component: str | None = None,
    ) -> None:
        self.job_id_mapping[
            get_async_job_cache_name(None, model_id, cs_universal, precision, component)
        ] = job_id

    def get_all_jobs(
        self,
        model_id: str,
        precisions: list[Precision] | None = None,
        devices: list[ScorecardDevice] | None = None,
        components: Iterable[str] | None = None,
    ) -> list[QuantizeScorecardJob]:
        if precisions is None:
            precisions = [Precision.float]
        model_runs: list[QuantizeScorecardJob] = [
            self.get_job(None, model_id, cs_universal, precision, component or None)
            for precision in precisions
            for component in components or [None]  # type: ignore[list-item]
        ]

        return model_runs


class CompileScorecardJobYaml(
    ScorecardJobYaml[CompileScorecardJob, ScorecardCompilePath, ModelCompileSummary]
):
    scorecard_job_type = CompileScorecardJob
    scorecard_path_type = ScorecardCompilePath
    scorecard_model_summary_type = ModelCompileSummary

    def get_all_jobs(
        self,
        model_id: str,
        precisions: list[Precision] | None = None,
        devices: list[ScorecardDevice] | None = None,
        components: Iterable[str] | None = None,
    ) -> list[CompileScorecardJob]:
        if precisions is None:
            precisions = [Precision.float]
        if devices:
            # Always include the universal device.
            devices_set = set(devices)
            devices_set.add(cs_universal)
            devices = list(devices_set)

        return super().get_all_jobs(model_id, precisions, devices, components)

    def get_job_id(
        self,
        path: ScorecardCompilePath | ScorecardProfilePath,
        model_id: str,
        device: ScorecardDevice,
        precision: Precision = Precision.float,
        component: str | None = None,
    ) -> str | None:
        """
        Get the ID of this job in the YAML that stores asyncronously-ran scorecard jobs.
        Returns None if job does not exist.

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
        job_id : str | None
            The job ID if found, None otherwise.
        """
        if isinstance(path, ScorecardProfilePath):
            # Get the compile job used with this profile path.
            path = path.compile_path

        if x := super().get_job_id(path, model_id, device, precision, component):
            return x

        # For compilation, fallback to the "universal" device if no path is found.
        if path and path.is_universal:
            return self.job_id_mapping.get(
                get_async_job_cache_name(
                    path,
                    model_id,
                    cs_universal,
                    precision,
                    component,
                )
            )

        return None


class ProfileScorecardJobYaml(
    ScorecardJobYaml[ProfileScorecardJob, ScorecardProfilePath, ModelPerfSummary]
):
    scorecard_job_type = ProfileScorecardJob
    scorecard_path_type = ScorecardProfilePath
    scorecard_model_summary_type = ModelPerfSummary


class InferenceScorecardJobYaml(
    ScorecardJobYaml[InferenceScorecardJob, ScorecardProfilePath, ModelInferenceSummary]
):
    scorecard_job_type = InferenceScorecardJob
    scorecard_path_type = ScorecardProfilePath
    scorecard_model_summary_type = ModelInferenceSummary


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.COMPILE], path: str | Path | None = None
) -> CompileScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.PROFILE], path: str | Path | None = None
) -> ProfileScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.INFERENCE], path: str | Path | None = None
) -> InferenceScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml(
    job_type: Literal[hub.JobType.QUANTIZE], path: str | Path | None = None
) -> QuantizeScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml(
    job_type: hub.JobType, path: str | Path | None = None
) -> ScorecardJobYaml: ...


def get_scorecard_job_yaml(
    job_type: hub.JobType, path: str | Path | None = None
) -> ScorecardJobYaml:
    """Loads the appropriate Scorecard job cache for the type of the given job."""
    if job_type == hub.JobType.COMPILE:
        return (
            CompileScorecardJobYaml()
            if not path
            else CompileScorecardJobYaml.from_file(path)
        )
    if job_type == hub.JobType.PROFILE:
        return (
            ProfileScorecardJobYaml()
            if not path
            else ProfileScorecardJobYaml.from_file(path)
        )
    if job_type == hub.JobType.INFERENCE:
        return (
            InferenceScorecardJobYaml()
            if not path
            else InferenceScorecardJobYaml.from_file(path)
        )
    if job_type == hub.JobType.QUANTIZE:
        return (
            QuantizeScorecardJobYaml()
            if not path
            else QuantizeScorecardJobYaml.from_file(path)
        )
    raise NotImplementedError(
        f"No file for storing test jobs of type {job_type.display_name}"
    )


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.CompileJob, path: str | Path | None = None
) -> CompileScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.ProfileJob, path: str | Path | None = None
) -> ProfileScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.InferenceJob, path: str | Path | None = None
) -> InferenceScorecardJobYaml: ...


@overload
def get_scorecard_job_yaml_from_job(
    job: hub.QuantizeJob, path: str | Path | None = None
) -> QuantizeScorecardJobYaml: ...


def get_scorecard_job_yaml_from_job(
    job: hub.Job, path: str | Path | None = None
) -> ScorecardJobYaml:
    return get_scorecard_job_yaml(job._job_type, path)


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
            precision, device.chipset if path.runtime.is_aot_compiled else None, path
        )
