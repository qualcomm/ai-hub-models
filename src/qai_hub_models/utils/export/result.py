# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import qai_hub as hub

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions

ValT = TypeVar("ValT")

"""
Generic containers that group values by graph name and/or component name.
Used throughout the export pipeline to hold jobs, models, or other per-graph/per-component values.
"""


class MultiGraphGroup(dict[str, ValT]):
    """Groups a value per graph for a model with multiple input specs."""


class ComponentGroup(dict[str, ValT]):
    """Groups a value per component for a collection model."""


class MultiGraphComponentGroup(dict[tuple[str, str], ValT]):
    """Groups a value per (component, graph_name) for a collection model with multi-graph components."""

    def by_component(self, component_name: str) -> dict[str, ValT]:
        """Return {graph_name: value} for entries matching the given component."""
        return {gn: v for (comp, gn), v in self.items() if comp == component_name}


"""
Results for the export script as a whole.
"""


@dataclass
class ExportResult:
    """The result of an export script for a standard model."""

    compile_job: hub.CompileJob | None = None
    quantize_job: hub.QuantizeJob | None = None
    inference_job: hub.InferenceJob | None = None
    link_job: hub.LinkJob | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None
    inference_metrics: pd.DataFrame | None = None


@dataclass
class MultiGraphExportResult:
    """The result of an export script for a model with multiple input specs."""

    quantize_job: hub.QuantizeJob | None = None
    compile_jobs: MultiGraphGroup[hub.CompileJob] | None = None
    link_job: hub.LinkJob | None = None
    inference_jobs: MultiGraphGroup[hub.InferenceJob] | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None
    inference_metrics: MultiGraphGroup[pd.DataFrame] | None = None


@dataclass
class CollectionExportResult:
    """The result of an export script for a collection model."""

    quantize_jobs: ComponentGroup[hub.QuantizeJob] | None = None
    compile_jobs: ComponentGroup[hub.CompileJob] | None = None
    link_jobs: ComponentGroup[hub.LinkJob] | None = None
    inference_jobs: ComponentGroup[hub.InferenceJob] | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None
    inference_metrics: ComponentGroup[pd.DataFrame] | None = None


@dataclass
class MultiGraphCollectionExportResult:
    """The result of an export script for a collection model with one more components that have multiple input specs."""

    quantize_jobs: ComponentGroup[hub.QuantizeJob] | None = None
    compile_jobs: MultiGraphComponentGroup[hub.CompileJob] | None = None
    link_jobs: ComponentGroup[hub.LinkJob] | None = None
    inference_jobs: MultiGraphComponentGroup[hub.InferenceJob] | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None
    inference_metrics: MultiGraphComponentGroup[pd.DataFrame] | None = None


@dataclass
class LegacyCollectionExportResult:
    """Legacy export result for hand-written LLM export scripts that bundle per-component results."""

    components: dict[str, ExportResult] = field(default_factory=dict)
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None


@dataclass
class ExportIntermediates:
    """
    Already-built assets to start from, in place of rebuilding them.

    Each stage takes a model or the job that made it, never both. A supplied
    field must cover every component/graph being built. Only the model fields
    have CLI flags (--quantized-model-id, --compiled-model-id).

    ``compiled_model`` is the compile job's output, before linking. For
    runtimes that link, the linked model comes from ``link_job``.
    """

    source_model: hub.Model | None = None
    calibration_data: hub.Dataset | None = None
    quantized_model: hub.Model | None = None
    compiled_model: hub.Model | None = None
    quantize_job: hub.QuantizeJob | None = None
    compile_job: hub.CompileJob | None = None
    link_job: hub.LinkJob | None = None
    inference_job: hub.InferenceJob | None = None


@dataclass
class MultiGraphExportIntermediates:
    """Already-built assets for a multi-graph (non-collection) model."""

    source_models: MultiGraphGroup[hub.Model] | None = None
    compiled_models: MultiGraphGroup[hub.Model] | None = None
    compile_jobs: MultiGraphGroup[hub.CompileJob] | None = None
    link_job: hub.LinkJob | None = None
    inference_jobs: MultiGraphGroup[hub.InferenceJob] | None = None


@dataclass
class CollectionExportIntermediates:
    """Already-built assets for a collection model, keyed per component."""

    source_models: ComponentGroup[hub.Model] | None = None
    calibration_data: ComponentGroup[hub.Dataset] | None = None
    quantized_models: ComponentGroup[hub.Model] | None = None
    compiled_models: ComponentGroup[hub.Model] | None = None
    quantize_jobs: ComponentGroup[hub.QuantizeJob] | None = None
    compile_jobs: ComponentGroup[hub.CompileJob] | None = None
    link_jobs: ComponentGroup[hub.LinkJob] | None = None
    inference_jobs: ComponentGroup[hub.InferenceJob] | None = None


@dataclass
class MultiGraphCollectionExportIntermediates:
    """Already-built assets for a multi-graph collection model (sharded LLMs)."""

    source_models: MultiGraphComponentGroup[hub.Model] | None = None
    compiled_models: MultiGraphComponentGroup[hub.Model] | None = None
    compile_jobs: MultiGraphComponentGroup[hub.CompileJob] | None = None
    link_jobs: ComponentGroup[hub.LinkJob] | None = None
    inference_jobs: MultiGraphComponentGroup[hub.InferenceJob] | None = None


ExportIntermediatesT = (
    ExportIntermediates
    | MultiGraphExportIntermediates
    | CollectionExportIntermediates
    | MultiGraphCollectionExportIntermediates
)

# A stage's asset is supplied as a model or as the job that made it, never both.
_MODEL_JOB_FIELDS = (
    (("quantized_model", "quantized_models"), ("quantize_job", "quantize_jobs")),
    (("compiled_model", "compiled_models"), ("compile_job", "compile_jobs")),
)
_COMPILE_FIELDS = ("compiled_model", "compiled_models", "compile_job", "compile_jobs")
_LINK_FIELDS = ("link_job", "link_jobs")
_INFERENCE_FIELDS = ("inference_job", "inference_jobs")


def _supplied(intermediates: ExportIntermediatesT, names: Iterable[str]) -> list[str]:
    return [n for n in names if getattr(intermediates, n, None) is not None]


def _component_of(key: Any) -> Any:
    return key[0] if isinstance(key, tuple) else key


def validate_export_intermediates(
    intermediates: ExportIntermediatesT,
    uses_hub_link: bool,
    keys: Iterable[Any] | None = None,
    quantize_keys: Iterable[Any] | None = None,
) -> None:
    """
    Raise ValueError if `intermediates` is contradictory or would make a
    pipeline rerun a stage whose output it was handed a later result of.

    * A stage's model and job are never both supplied.
    * A link job comes with the compile asset it linked.
    * An inference job comes with the compile asset and, for runtimes that
      link, the link job it ran on.
    * A supplied per-component or per-graph field covers all of `keys` (the
      components, or the (component, graph) pairs, being built). Link jobs
      are per component. Quantize fields cover `quantize_keys` instead.
    """
    for model_names, job_names in _MODEL_JOB_FIELDS:
        models = _supplied(intermediates, model_names)
        jobs = _supplied(intermediates, job_names)
        if models and jobs:
            raise ValueError(
                f"export_intermediates sets both a model and a job for one stage: {models + jobs}."
            )

    has_compile = bool(_supplied(intermediates, _COMPILE_FIELDS))
    has_link = bool(_supplied(intermediates, _LINK_FIELDS))
    if has_link and not has_compile:
        raise ValueError(
            "export_intermediates sets a link job without the compile asset it linked; "
            "the compile stage would rerun."
        )
    if _supplied(intermediates, _INFERENCE_FIELDS) and (
        not has_compile or (uses_hub_link and not has_link)
    ):
        raise ValueError(
            "export_intermediates sets an inference job without the compile asset"
            + (" and link job" if uses_hub_link else "")
            + " it ran on; the earlier stages would rerun."
        )

    keys = list(keys or [])
    required = {
        "compile_jobs": keys,
        "compiled_models": keys,
        "inference_jobs": keys,
        "link_jobs": list(dict.fromkeys(_component_of(k) for k in keys)),
        "quantize_jobs": list(quantize_keys or []),
        "quantized_models": list(quantize_keys or []),
    }
    for field_name, field_keys in required.items():
        supplied = getattr(intermediates, field_name, None)
        if supplied is not None and (
            missing := [k for k in field_keys if k not in supplied]
        ):
            raise ValueError(
                f"export_intermediates.{field_name} is missing entries for {missing}; "
                "partial coverage is not supported."
            )
