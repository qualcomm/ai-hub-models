# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

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
    profile_job: hub.ProfileJob | None = None
    inference_job: hub.InferenceJob | None = None
    link_job: hub.LinkJob | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None


@dataclass
class MultiGraphExportResult:
    """The result of an export script for a model with multiple input specs."""

    quantize_job: hub.QuantizeJob | None = None
    compile_jobs: MultiGraphGroup[hub.CompileJob] | None = None
    link_job: hub.LinkJob | None = None
    profile_jobs: MultiGraphGroup[hub.ProfileJob] | None = None
    inference_jobs: MultiGraphGroup[hub.InferenceJob] | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None


@dataclass
class CollectionExportResult:
    """The result of an export script for a collection model."""

    quantize_jobs: ComponentGroup[hub.QuantizeJob] | None = None
    compile_jobs: ComponentGroup[hub.CompileJob] | None = None
    link_jobs: ComponentGroup[hub.LinkJob] | None = None
    profile_jobs: ComponentGroup[hub.ProfileJob] | None = None
    inference_jobs: ComponentGroup[hub.InferenceJob] | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None


@dataclass
class MultiGraphCollectionExportResult:
    """The result of an export script for a collection model with one more components that have multiple input specs."""

    quantize_jobs: ComponentGroup[hub.QuantizeJob] | None = None
    compile_jobs: MultiGraphComponentGroup[hub.CompileJob] | None = None
    link_jobs: ComponentGroup[hub.LinkJob] | None = None
    profile_jobs: MultiGraphComponentGroup[hub.ProfileJob] | None = None
    inference_jobs: MultiGraphComponentGroup[hub.InferenceJob] | None = None
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None


@dataclass
class LegacyCollectionExportResult:
    """Legacy export result for hand-written LLM export scripts that bundle per-component results."""

    components: dict[str, ExportResult] = field(default_factory=dict)
    download_path: Path | None = None
    tool_versions: ToolVersions | None = None
    metadata: ModelMetadata | None = None
