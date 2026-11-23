# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar, cast, final

from qai_hub import JobType

from qai_hub_models import Precision
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
    ScorecardInferencePath,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.results.scorecard_job import JobTypeVar  # noqa: F401
from qai_hub_models.utils.export.result import ComponentGroup

ScorecardPathT = TypeVar(
    "ScorecardPathT", ScorecardInferencePath, ScorecardCompilePath, None
)


def _job_id(
    model_id: str,
    precision: Precision | None,
    path: ScorecardInferencePath | ScorecardCompilePath | None,
    device: ScorecardDevice | None,
    component: str | None,
    graph_name: str | None,
) -> str:
    return (
        f"{model_id}"
        + (
            ("_" + str(precision))
            if (precision is not None and precision != Precision.float)
            else ""
        )
        + (("_" + path.name) if path else "")
        + (("-" + device.name) if device else "")
        + (("_" + component) if component else "")
        + (("_" + graph_name) if graph_name else "")
    )


def _str_with_description(
    val: str,
    model_id: str,
    precision: Precision | None,
    path: ScorecardInferencePath | ScorecardCompilePath | None,
    device: ScorecardDevice | None,
    component: str | None,
    graph_name: str | None,
) -> str:
    model_name = f"{model_id}::{component}" if component else model_id
    model_name = f"{model_name}::{graph_name}" if graph_name else model_name
    precision_opt = f"{precision} | " if precision is not None else ""
    path_opt = f"{path.name} | " if path is not None else ""
    device_opt = f"{device.name} | " if device is not None else ""
    return f"{model_name} | {precision_opt}{path_opt}{device_opt}: {val}"


@final
@dataclass
class ScJobParams(Generic[ScorecardPathT]):
    """The parameters necessary to identify a job in the scorecard."""

    model_id: str

    # Optional because sc path is not necessary to identify quantize or pre-quantize-compile jobs.
    path: ScorecardPathT

    # Optional because precision is not necessary to identify pre-quantize-compile jobs.
    precision: Precision | None = None

    # Optional because device is not necessary to identify quantize jobs and some compile jobs.
    device: ScorecardDevice | None = None

    # Optional because not all models have components.
    component: str | None = None

    # Optional because not all models have graph names.
    graph_name: str | None = None

    @property
    def has_pre_qdq_compile_job(self) -> bool:
        """Whether these params apply to a pre-qdq-compile-job."""
        return self.precision is None or self.precision != Precision.float

    @cached_property
    def pre_quantize_compile_job_id(self) -> str:
        """A unique string ID of a pre-quantize-compile-to-onnx job with these parameters."""
        if not self.has_pre_qdq_compile_job:
            raise ValueError(
                "No pre qdq compile job exists for a floating point target precision."
            )
        return _job_id(
            self.model_id,
            self.precision,
            ScorecardCompilePath.ONNX_FOR_QUANTIZATION,
            None,
            self.component,
            None,
        )

    @property
    def has_quantize_job(self) -> bool:
        """Whether these params apply to a quantize job."""
        return self.precision is not None and self.precision != Precision.float

    @cached_property
    def quantize_job_id(self) -> str:
        """A unique string ID of a quantize job with these parameters."""
        if not self.has_quantize_job:
            raise ValueError(
                "No quantize job exists for a floating point target precision."
            )
        return _job_id(self.model_id, self.precision, None, None, self.component, None)

    @property
    def has_compile_job(self) -> bool:
        """Whether these params apply to a compile job."""
        return self.path is not None and self.precision is not None

    @cached_property
    def compile_job_id(self) -> str:
        """A unique string ID of a compile job with these parameters."""
        if not self.has_compile_job:
            raise ValueError(
                self.str_with_description("Test params do not apply to compile jobs.")
            )
        assert self.path is not None
        return _job_id(
            self.model_id,
            self.precision,
            self.path,
            self.device if self.path.runtime.is_aot_compiled else None,
            self.component,
            self.graph_name,
        )

    @property
    def has_link_job(self) -> bool:
        """Whether these params apply to a link job."""
        if self.device is None or self.device == cs_universal:
            return False
        if self.path is None:
            return False
        return self.path.runtime.uses_hub_link

    @cached_property
    def link_job_id(self) -> str:
        """A unique string ID of a link job with these parameters."""
        if not self.has_link_job:
            raise ValueError(
                self.str_with_description("Test params do not apply to link jobs.")
            )
        return _job_id(
            self.model_id, self.precision, self.path, self.device, self.component, None
        )

    @property
    def has_inference_job(self) -> bool:
        """Whether these params apply to an inference job."""
        if not isinstance(self.path, ScorecardInferencePath) or self.precision is None:
            return False
        return self.device is not None and self.device != cs_universal

    @cached_property
    def inference_job_id(self) -> str:
        """A unique string ID of an inference job with these parameters."""
        if not self.has_inference_job:
            raise ValueError(
                self.str_with_description("Test params do not apply to inference jobs.")
            )
        return _job_id(
            self.model_id,
            self.precision,
            self.path,
            self.device,
            self.component,
            self.graph_name,
        )

    def job_id(self, job_type: JobType) -> str:
        """A unique string ID of a job of the given type with these parameters."""
        if job_type == JobType.QUANTIZE:
            return self.quantize_job_id
        if job_type == JobType.COMPILE:
            return self.compile_job_id
        if job_type == JobType.LINK:
            return self.link_job_id
        if job_type == JobType.INFERENCE:
            return self.inference_job_id
        raise NotImplementedError()

    def __hash__(self) -> int:
        return hash(
            (
                self.model_id,
                self.precision,
                self.path,
                self.device,
                self.component,
                self.graph_name,
            )
        )

    def str_with_description(self, val: str) -> str:
        """Print a string with a job-identifier prefix beforehand."""
        return _str_with_description(
            val,
            self.model_id,
            self.precision,
            self.path,
            self.device,
            self.component,
            self.graph_name,
        )


@final
@dataclass
class ScExportTestParams(Generic[ScorecardPathT]):
    """
    The necessary parameters to identify all jobs that run as a part of a single export test.
    A "single export test" is equal to a user running the export script (one model + runtime + precision + device).
    """

    model_id: str
    path: ScorecardPathT
    precision: Precision | None = None
    device: ScorecardDevice | None = None
    component_names: list[str] | None = None
    graph_names: list[str] | None = None
    component_graph_names: ComponentGroup[list[str]] | None = None

    # Components expected to have profile / inference jobs. None means all of them.
    # Hybrid LLMs narrow this: their backbone parts compile and link but are measured
    # end-to-end, so they never get a profile job of their own.
    profile_component_names: list[str] | None = None

    def str_with_description(self, val: str) -> str:
        return _str_with_description(
            val, self.model_id, self.precision, self.path, self.device, None, None
        )

    def expects_inference_job(self, component: str | None) -> bool:
        """Whether a profile / inference job is expected for this component."""
        if self.profile_component_names is None or component is None:
            return True
        return component in self.profile_component_names

    @cached_property
    def component_gn_pairs(self) -> list[tuple[str | None, str | None]]:
        """
        A list of all component + graph name pairings for this model.

        Returns
        -------
        list[tuple[str | None, str | None]]
            Tuple of [Component Name, Graph Name]
            If there are no components, the component name will be None.
            If there are no graph names, one (compoonent name, graph name) entry will appear, with the graph name as None.
        """
        if self.component_names is not None:
            if self.component_graph_names is not None:
                return [
                    (component, graph_name)
                    for component in self.component_names
                    for graph_name in self.component_graph_names.get(component, [None])
                ]
            return [(component, None) for component in self.component_names]
        if self.graph_names:
            return [(None, graph_name) for graph_name in self.graph_names]

        return [(None, None)]

    @property
    def all_pre_qdq_compile_job_params(self) -> list[ScJobParams[ScorecardCompilePath]]:
        """A list of all expected pre-QDQ compile jobs for this export test."""
        components: Sequence[str | None] = self.component_names or cast(
            list[str | None], [None]
        )
        return [
            ScJobParams(
                self.model_id,
                precision=None,
                path=ScorecardCompilePath.ONNX_FOR_QUANTIZATION,
                component=component,
            )
            for component in components
        ]

    @property
    def all_quantize_job_params(self) -> list[ScJobParams[None]]:
        """A list of all expected quantize jobs for this export test."""
        if self.precision == Precision.float or self.precision is None:
            return []
        if self.graph_names or self.component_graph_names:
            raise ValueError(
                "Models with multiple graphs are not supported for auto-quantization."
            )
        components: Sequence[str | None] = self.component_names or cast(
            list[str | None], [None]
        )
        return [
            ScJobParams(
                self.model_id,
                path=None,
                precision=self.precision,
                component=component,
            )
            for component in components
        ]

    @property
    def all_compile_job_params(self) -> list[ScJobParams[ScorecardCompilePath]]:
        """A list of all expected compile jobs for this export test."""
        assert self.path is not None and self.precision is not None
        return [
            ScJobParams(
                self.model_id,
                path=self.path.compile_path
                if isinstance(self.path, ScorecardInferencePath)
                else self.path,
                precision=self.precision,
                device=self.device,
                component=component,
                graph_name=graph_name,
            )
            for (component, graph_name) in self.component_gn_pairs
        ]

    @property
    def all_link_job_params(self) -> list[ScJobParams[ScorecardCompilePath]]:
        """A list of all expected link jobs for this export test."""
        assert (
            self.path is not None
            and self.precision is not None
            and self.path.runtime.uses_hub_link
            and self.device is not None
        )
        components: Sequence[str | None] = self.component_names or cast(
            list[str | None], [None]
        )
        return [
            ScJobParams(
                self.model_id,
                path=self.path.compile_path
                if isinstance(self.path, ScorecardInferencePath)
                else self.path,
                precision=self.precision,
                device=self.device,
                component=component,
            )
            for component in components
        ]

    @property
    def all_inference_job_params(self) -> list[ScJobParams[ScorecardInferencePath]]:
        """A list of all expected inference jobs for this export test."""
        assert (
            isinstance(self.path, ScorecardInferencePath)
            and self.device is not None
            and self.precision is not None
        )
        return [
            ScJobParams(
                self.model_id,
                path=self.path,
                precision=self.precision,
                device=self.device,
                component=component,
                graph_name=graph_name,
            )
            for (component, graph_name) in self.component_gn_pairs
        ]

    def all_job_params(self, job_type: JobType) -> list[ScJobParams]:
        """A list of all expected jobs of a given type for this export test."""
        if job_type == JobType.QUANTIZE:
            return self.all_quantize_job_params
        if job_type == JobType.COMPILE:
            return self.all_compile_job_params
        if job_type == JobType.LINK:
            return self.all_link_job_params
        if job_type == JobType.INFERENCE:
            return self.all_inference_job_params
        raise NotImplementedError()

    def applies_to_job_type(self, job_type: JobType) -> bool:
        """Whether these params identify any jobs of *job_type*."""
        if job_type == JobType.QUANTIZE:
            return (
                self.precision is not None
                and self.precision != Precision.float
                and not self.graph_names
                and not self.component_graph_names
            )
        if job_type == JobType.COMPILE:
            return self.path is not None and self.precision is not None
        if job_type == JobType.LINK:
            return (
                self.path is not None
                and self.precision is not None
                and self.path.runtime.uses_hub_link
                and self.device is not None
            )
        if job_type == JobType.INFERENCE:
            return (
                isinstance(self.path, ScorecardInferencePath)
                and self.device is not None
                and self.precision is not None
            )
        raise NotImplementedError()
