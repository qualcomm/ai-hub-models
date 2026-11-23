# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Lets the scorecard stages treat every recipe the same way. Single models,
collection models (one set of jobs per component), multi-graph models (one set
per graph) and multi-graph collections (one per component and graph, e.g.
LLMs) each have their own export pipeline, result type and intermediates type.
The stages pass the recipe's own ``*ExportIntermediates`` around; the functions
here read and update them, so the stages don't need to check which pipeline
type they have.
"""

from __future__ import annotations

import dataclasses
import functools
from enum import Enum
from typing import TypeVar

import qai_hub as hub

from qai_hub_models.utils.base_collection_model import CollectionModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.context import resolve_model_cls
from qai_hub_models.utils.export.result import (
    CollectionExportIntermediates,
    CollectionExportResult,
    ComponentGroup,
    ExportIntermediates,
    ExportResult,
    MultiGraphCollectionExportIntermediates,
    MultiGraphCollectionExportResult,
    MultiGraphComponentGroup,
    MultiGraphExportIntermediates,
    MultiGraphExportResult,
    MultiGraphGroup,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

T = TypeVar("T")

Intermediates = (
    ExportIntermediates
    | CollectionExportIntermediates
    | MultiGraphExportIntermediates
    | MultiGraphCollectionExportIntermediates
)
Result = (
    ExportResult
    | CollectionExportResult
    | MultiGraphExportResult
    | MultiGraphCollectionExportResult
)

# Each is one value (a single model, or the single link job of a multi-graph
# model) or a group of them (collection component / multi-graph graph /
# multi-graph collection (component, graph)).
SourceModels = (
    hub.Model
    | ComponentGroup[hub.Model]
    | MultiGraphGroup[hub.Model]
    | MultiGraphComponentGroup[hub.Model]
)
CalibrationData = hub.Dataset | ComponentGroup[hub.Dataset]
QuantizeJobs = (
    hub.QuantizeJob
    | ComponentGroup[hub.QuantizeJob]
    | MultiGraphGroup[hub.QuantizeJob]
    | MultiGraphComponentGroup[hub.QuantizeJob]
)
CompileJobs = (
    hub.CompileJob
    | ComponentGroup[hub.CompileJob]
    | MultiGraphGroup[hub.CompileJob]
    | MultiGraphComponentGroup[hub.CompileJob]
)
LinkJobs = (
    hub.LinkJob
    | ComponentGroup[hub.LinkJob]
    | MultiGraphGroup[hub.LinkJob]
    | MultiGraphComponentGroup[hub.LinkJob]
)
InferenceJobs = (
    hub.InferenceJob
    | ComponentGroup[hub.InferenceJob]
    | MultiGraphGroup[hub.InferenceJob]
    | MultiGraphComponentGroup[hub.InferenceJob]
)


class PipelineType(Enum):
    """
    Which export pipeline a recipe uses. It follows from the model's class, but
    the stages only have a model id, and loading the class is slow, so this is
    looked up once per model (see ``pipeline_type``).
    """

    SINGLE = "single"
    COLLECTION = "collection"
    MULTI_GRAPH = "multi_graph"
    MULTI_GRAPH_COLLECTION = "multi_graph_collection"


@functools.cache
def pipeline_type(model_id: str) -> PipelineType:
    model_cls = resolve_model_cls(QAIHM_MODELS_ROOT / model_id)
    if issubclass(model_cls, MultiGraphCollectionModel):
        return PipelineType.MULTI_GRAPH_COLLECTION
    if issubclass(model_cls, CollectionModel):
        return PipelineType.COLLECTION
    if issubclass(model_cls, MultiGraphWorkbenchModel):
        return PipelineType.MULTI_GRAPH
    return PipelineType.SINGLE


def empty_intermediates(shape: PipelineType) -> Intermediates:
    if shape == PipelineType.COLLECTION:
        return CollectionExportIntermediates()
    if shape == PipelineType.MULTI_GRAPH:
        return MultiGraphExportIntermediates()
    if shape == PipelineType.MULTI_GRAPH_COLLECTION:
        return MultiGraphCollectionExportIntermediates()
    return ExportIntermediates()


def source_of(intermediates: Intermediates) -> SourceModels | None:
    if isinstance(intermediates, ExportIntermediates):
        return intermediates.source_model
    return intermediates.source_models


_NEVER_QUANTIZED = (
    MultiGraphExportIntermediates,
    MultiGraphCollectionExportIntermediates,
)


def calibration_data_of(intermediates: Intermediates) -> CalibrationData | None:
    """Multi-graph models and collections are never quantized, so have none."""
    if isinstance(intermediates, _NEVER_QUANTIZED):
        return None
    return intermediates.calibration_data


def quantize_of(intermediates: Intermediates) -> QuantizeJobs | None:
    """Multi-graph models and collections are never quantized."""
    if isinstance(intermediates, _NEVER_QUANTIZED):
        return None
    if isinstance(intermediates, ExportIntermediates):
        return intermediates.quantize_job
    return intermediates.quantize_jobs


def compile_of(intermediates: Intermediates) -> CompileJobs | None:
    if isinstance(intermediates, ExportIntermediates):
        return intermediates.compile_job
    return intermediates.compile_jobs


def link_of(intermediates: Intermediates) -> LinkJobs | None:
    if isinstance(intermediates, ExportIntermediates | MultiGraphExportIntermediates):
        return intermediates.link_job
    return intermediates.link_jobs  # collection / multi-graph collection


def inference_of(intermediates: Intermediates) -> InferenceJobs | None:
    if isinstance(intermediates, ExportIntermediates):
        return intermediates.inference_job
    return intermediates.inference_jobs


def _one(value: object, cls: type[T], default: T | None) -> T | None:
    """`value` as a `cls`, or `default` if it is None."""
    if value is None:
        return default
    assert isinstance(value, cls), f"Expected {cls.__name__}, got {type(value)}"
    return value


def _components(
    value: object, cls: type[T], default: ComponentGroup[T] | None
) -> ComponentGroup[T] | None:
    """`value` as a ComponentGroup of `cls`, or `default` if it is None."""
    if value is None:
        return default
    assert isinstance(value, ComponentGroup), (
        f"Expected ComponentGroup, got {type(value)}"
    )
    assert all(isinstance(v, cls) for v in value.values())
    return value


def _graphs(
    value: object, cls: type[T], default: MultiGraphGroup[T] | None
) -> MultiGraphGroup[T] | None:
    """`value` as a MultiGraphGroup of `cls`, or `default` if it is None."""
    if value is None:
        return default
    assert isinstance(value, MultiGraphGroup), (
        f"Expected MultiGraphGroup, got {type(value)}"
    )
    assert all(isinstance(v, cls) for v in value.values())
    return value


def _graph_components(
    value: object, cls: type[T], default: MultiGraphComponentGroup[T] | None
) -> MultiGraphComponentGroup[T] | None:
    """`value` as a MultiGraphComponentGroup of `cls`, or `default` if it is None."""
    if value is None:
        return default
    assert isinstance(value, MultiGraphComponentGroup), (
        f"Expected MultiGraphComponentGroup, got {type(value)}"
    )
    assert all(isinstance(v, cls) for v in value.values())
    return value


def typed_jobs(
    value: object, job_cls: type[T]
) -> T | ComponentGroup[T] | MultiGraphGroup[T] | MultiGraphComponentGroup[T] | None:
    """`value`, e.g. read from a job-ID YAML, checked to hold only `job_cls` jobs."""
    if value is None:
        return None
    if isinstance(value, MultiGraphComponentGroup):
        return _graph_components(value, job_cls, None)
    if isinstance(value, ComponentGroup):
        return _components(value, job_cls, None)
    if isinstance(value, MultiGraphGroup):
        return _graphs(value, job_cls, None)
    return _one(value, job_cls, None)


def update_intermediates(
    intermediates: Intermediates,
    *,
    source: SourceModels | None = None,
    calibration_data: CalibrationData | None = None,
    quantize_jobs: QuantizeJobs | None = None,
    compile_jobs: CompileJobs | None = None,
    link_jobs: LinkJobs | None = None,
    inference_jobs: InferenceJobs | None = None,
) -> Intermediates:
    """
    `intermediates` with whichever assets are given filled in; None leaves that
    asset as is. Asserts each has the form the shape uses (one, or a group).
    """
    if isinstance(intermediates, ExportIntermediates):
        return dataclasses.replace(
            intermediates,
            source_model=_one(source, hub.Model, intermediates.source_model),
            calibration_data=_one(
                calibration_data, hub.Dataset, intermediates.calibration_data
            ),
            quantize_job=_one(
                quantize_jobs, hub.QuantizeJob, intermediates.quantize_job
            ),
            compile_job=_one(compile_jobs, hub.CompileJob, intermediates.compile_job),
            link_job=_one(link_jobs, hub.LinkJob, intermediates.link_job),
            inference_job=_one(
                inference_jobs, hub.InferenceJob, intermediates.inference_job
            ),
        )
    if isinstance(intermediates, CollectionExportIntermediates):
        return dataclasses.replace(
            intermediates,
            source_models=_components(source, hub.Model, intermediates.source_models),
            calibration_data=_components(
                calibration_data, hub.Dataset, intermediates.calibration_data
            ),
            quantize_jobs=_components(
                quantize_jobs, hub.QuantizeJob, intermediates.quantize_jobs
            ),
            compile_jobs=_components(
                compile_jobs, hub.CompileJob, intermediates.compile_jobs
            ),
            link_jobs=_components(link_jobs, hub.LinkJob, intermediates.link_jobs),
            inference_jobs=_components(
                inference_jobs, hub.InferenceJob, intermediates.inference_jobs
            ),
        )
    assert quantize_jobs is None, "Multi-graph models are not quantized"
    assert calibration_data is None, "Multi-graph models are not quantized"
    if isinstance(intermediates, MultiGraphCollectionExportIntermediates):
        return dataclasses.replace(
            intermediates,
            source_models=_graph_components(
                source, hub.Model, intermediates.source_models
            ),
            compile_jobs=_graph_components(
                compile_jobs, hub.CompileJob, intermediates.compile_jobs
            ),
            link_jobs=_components(link_jobs, hub.LinkJob, intermediates.link_jobs),
            inference_jobs=_graph_components(
                inference_jobs, hub.InferenceJob, intermediates.inference_jobs
            ),
        )
    return dataclasses.replace(
        intermediates,
        source_models=_graphs(source, hub.Model, intermediates.source_models),
        compile_jobs=_graphs(compile_jobs, hub.CompileJob, intermediates.compile_jobs),
        link_job=_one(link_jobs, hub.LinkJob, intermediates.link_job),
        inference_jobs=_graphs(
            inference_jobs, hub.InferenceJob, intermediates.inference_jobs
        ),
    )


def _input_model(job: hub.Job) -> hub.Model:
    """The model `job` consumes. A link job must have exactly one."""
    if isinstance(job, hub.LinkJob):
        assert len(job.models) == 1, f"Link job {job.job_id} has several inputs"
        return job.models[0]
    assert isinstance(job, hub.CompileJob | hub.QuantizeJob | hub.InferenceJob)
    return job.model


def get_source_model(job: hub.Job) -> hub.Model:
    """
    The uploaded model behind `job`, found by following producers back from its
    input model. Each hop is a round trip, so start from the earliest job known.
    """
    model = _input_model(job)
    producer = model.get_producer()
    if producer is None:
        return model
    return get_source_model(producer)


def update_from_result(intermediates: Intermediates, result: Result) -> Intermediates:
    """
    `intermediates` updated with the jobs `result` carries. The source models
    (and calibration data) are derived from those jobs if there are none yet.
    """
    source: SourceModels | None = None
    calibration_data: CalibrationData | None = None
    quantize_jobs: QuantizeJobs | None
    compile_jobs: CompileJobs | None
    link_jobs: LinkJobs | None
    inference_jobs: InferenceJobs | None
    if isinstance(result, ExportResult):
        quantize_jobs, compile_jobs = result.quantize_job, result.compile_job
        link_jobs, inference_jobs = result.link_job, result.inference_job
        if (first_job := result.quantize_job or result.compile_job) is not None:
            source = get_source_model(first_job)
        if quantize_jobs is not None:
            calibration_data = quantize_jobs.calibration_dataset
    elif isinstance(result, CollectionExportResult):
        quantize_jobs, compile_jobs = result.quantize_jobs, result.compile_jobs
        link_jobs, inference_jobs = result.link_jobs, result.inference_jobs
        if result.quantize_jobs is not None:
            calibration_data = ComponentGroup(
                {c: j.calibration_dataset for c, j in result.quantize_jobs.items()}
            )
        if result.compile_jobs is not None:
            quantize_by_component: ComponentGroup[hub.QuantizeJob] = (
                result.quantize_jobs or ComponentGroup()
            )
            source = ComponentGroup(
                {
                    c: get_source_model(quantize_by_component.get(c) or job)
                    for c, job in result.compile_jobs.items()
                }
            )
    elif isinstance(result, MultiGraphCollectionExportResult):
        quantize_jobs, compile_jobs = None, result.compile_jobs
        link_jobs, inference_jobs = result.link_jobs, result.inference_jobs
        if result.compile_jobs is not None:
            source = MultiGraphComponentGroup(
                {k: get_source_model(job) for k, job in result.compile_jobs.items()}
            )
    else:
        quantize_jobs, compile_jobs = None, result.compile_jobs
        link_jobs, inference_jobs = result.link_job, result.inference_jobs
        if result.compile_jobs is not None:
            source = MultiGraphGroup(
                {g: get_source_model(job) for g, job in result.compile_jobs.items()}
            )
    return update_intermediates(
        intermediates,
        source=None if source_of(intermediates) is not None else source,
        calibration_data=calibration_data,
        quantize_jobs=quantize_jobs,
        compile_jobs=compile_jobs,
        link_jobs=link_jobs,
        inference_jobs=inference_jobs,
    )
