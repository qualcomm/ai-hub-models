# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar

from qai_hub import Device
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.protocols import FromPretrainedProtocol
from qai_hub_models.utils.base_model import (
    WorkbenchModel,
    _model_cls_name,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
)
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, get_channel_last
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.qai_hub_helpers import (
    build_compile_options,
    build_inference_options,
    build_link_options,
    build_quantize_options,
    expand_to_batch_size,
    make_sample_inputs,
)
from qai_hub_models.utils.transpose_channel import (
    transpose_channel_first_to_last,
)


class MultiGraphCollectionModel(ABC, FromPretrainedProtocol):
    """
    Abstract interface for collection models where components may have multiple graphs.

    Each component is identified by name and contains one or more independently-compiled
    graphs. Per-component methods accept both a component_name and a graph_name, while
    "all-component" getters aggregate results into MultiGraphComponentGroup dicts keyed
    by (component_name, graph_name).
    """

    # -- Subclasses must implement these --
    @property
    @abstractmethod
    def component_names(self) -> list[str]:
        """Ordered list of component names in this collection."""
        ...

    @abstractmethod
    def get_component_graph_names(self, component_name: str) -> list[str]:
        """Return graph names for a component."""
        ...

    @abstractmethod
    def get_component_graph_input_spec(
        self, component_name: str, graph_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        """Return the input spec for a specific graph of a component."""
        ...

    @abstractmethod
    def get_component_graph_output_spec(
        self, component_name: str, graph_name: str
    ) -> OutputSpec:
        """Return the output spec for a specific graph of a component."""
        ...

    @abstractmethod
    def serialize_component_graph(
        self,
        component_name: str,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Serialize a component graph to disk in a format suitable for AI Hub compilation."""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self: ...

    # -- Subclasses may override these --
    @property
    def name(self) -> str:
        """Model name."""
        return _model_cls_name(self)

    def get_component_has_shared_source_model(self, component_name: str) -> bool:
        """Return whether all graphs in a component share a single source model file."""
        return False

    def get_component_unsupported_reason(
        self,
        component_name: str,
        graph_name: str,
        target_runtime: TargetRuntime,
        device: Device,
    ) -> str | None:
        """Return a human-readable reason if the component graph is unsupported, or None."""
        return None

    def get_component_graph_litemp_percentage(
        self, component_name: str, graph_name: str, precision: Precision
    ) -> float | None:
        """Return the Lite-MP percentage for mixed-precision quantization, or None."""
        return None

    def get_component_graph_mixed_precision(
        self, component_name: str, graph_name: str, precision: Precision
    ) -> Precision:
        """Return the per-graph precision when using mixed precision mode."""
        assert precision in (Precision.mixed, Precision.mixed_with_float)
        raise NotImplementedError(
            "Mixed precision is not supported by this collection model."
        )

    def get_component_graph_hub_quantize_options(
        self,
        component_name: str,
        graph_name: str,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> str:
        """Return AI Hub quantize option string for a component graph."""
        litemp_percentage = (
            self.get_component_graph_litemp_percentage(
                component_name, graph_name, precision
            )
            if precision.override_type is not None
            else None
        )
        return build_quantize_options(
            precision, litemp_percentage, other_quantize_options
        )

    def get_component_graph_hub_compile_options(
        self,
        component_name: str,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        """Return AI Hub compile option string for a component graph."""
        return build_compile_options(
            target_runtime,
            precision,
            self.get_component_graph_input_spec(component_name, graph_name),
            self.get_component_graph_output_spec(component_name, graph_name),
            graph_name,
        )

    def get_component_hub_link_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        """Return AI Hub link option string for a component."""
        return build_link_options(
            target_runtime,
            other_link_options,
        )

    def get_component_graph_hub_inference_options(
        self,
        component_name: str,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> str:
        """Return AI Hub inference option string for a component graph."""
        return build_inference_options(
            target_runtime, graph_name, other_inference_options
        )

    def get_component_graph_sample_inputs(
        self,
        component_name: str,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        """Generate sample inputs for a component graph, optionally in channel-last format."""
        input_spec = input_spec or self.get_component_graph_input_spec(
            component_name, graph_name
        )
        inputs = expand_to_batch_size(make_sample_inputs(input_spec), input_spec)
        if use_channel_last_format and (channel_last := get_channel_last(input_spec)):
            return transpose_channel_first_to_last(channel_last, inputs)
        return inputs

    def remap_metadata_input_spec(
        self,
        component_name: str,
        graph_input_spec: InputSpec,
        compiled_input_names: set[str],
    ) -> InputSpec:
        """Allow model-specific remapping before metadata input merge.

        By default, return the graph input spec unchanged.
        """
        return graph_input_spec

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        """
        Write supplementary files required by the model during inference.
        These files will be packaged alongside the model when deployed.

        Parameters
        ----------
        output_dir
            Directory where the supplementary files should be written.
        metadata
            The metadata for the compiled models.
            metadata.supplementary_files will be populated with the files written.
        """
        return

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        """Returns list of dataset classes on which this model can be evaluated."""
        return []

    def get_calibration_dataset_cls(self) -> type[BaseDataset] | None:
        """Dataset class used for calibration when quantizing the model."""
        return None

    # -- All-component getters --

    @property
    def all_component_graph_names(self) -> ComponentGroup[list[str]]:
        """Return a mapping of component name to its list of graph names."""
        return ComponentGroup(
            {
                component_name: self.get_component_graph_names(component_name)
                for component_name in self.component_names
            }
        )

    @property
    def all_flattened_component_graph_names(self) -> list[tuple[str, str]]:
        """Return a flat list of (component_name, graph_name) tuples across all components."""
        return [
            (component_name, graph_name)
            for component_name in self.component_names
            for graph_name in self.get_component_graph_names(component_name)
        ]

    def get_input_spec(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> MultiGraphComponentGroup[InputSpec]:
        """Return input specs for all component graphs as a MultiGraphComponentGroup."""
        return MultiGraphComponentGroup[InputSpec](
            {
                (component_name, graph_name): self.get_component_graph_input_spec(
                    component_name, graph_name, *args, **kwargs
                )
                for component_name, graph_name in self.all_flattened_component_graph_names
            }
        )

    def get_unsupported_reason(
        self,
        target_runtime: TargetRuntime,
        device: Device,
    ) -> str | None:
        """Return the first unsupported reason across all components, or None."""
        return None

    def get_hub_quantize_options(
        self,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> MultiGraphComponentGroup[str]:
        """Return AI Hub quantize options for all component graphs."""
        return MultiGraphComponentGroup[str](
            {
                (
                    component_name,
                    graph_name,
                ): self.get_component_graph_hub_quantize_options(
                    component_name, graph_name, precision, other_quantize_options
                )
                for component_name, graph_name in self.all_flattened_component_graph_names
            }
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> MultiGraphComponentGroup[str]:
        """Return AI Hub compile options for all component graphs."""
        return MultiGraphComponentGroup[str](
            {
                (
                    component_name,
                    graph_name,
                ): self.get_component_graph_hub_compile_options(
                    component_name,
                    graph_name,
                    target_runtime,
                    precision,
                    other_compile_options,
                    device,
                )
                for component_name, graph_name in self.all_flattened_component_graph_names
            }
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> ComponentGroup[str]:
        """Return AI Hub link options for all components."""
        return ComponentGroup[str](
            {
                component_name: self.get_component_hub_link_options(
                    component_name,
                    target_runtime,
                    other_link_options,
                )
                for component_name in self.component_names
            }
        )

    def get_hub_inference_options(
        self,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> MultiGraphComponentGroup[str]:
        """Return AI Hub inference options for all component graphs."""
        return MultiGraphComponentGroup[str](
            {
                (
                    component_name,
                    graph_name,
                ): self.get_component_graph_hub_inference_options(
                    component_name,
                    graph_name,
                    target_runtime,
                    other_inference_options,
                )
                for component_name, graph_name in self.all_flattened_component_graph_names
            }
        )

    def sample_inputs(
        self,
        input_spec: MultiGraphComponentGroup[InputSpec] | None = None,
        use_channel_last_format: bool = True,
    ) -> MultiGraphComponentGroup[SampleInputsType]:
        """Return sample inputs for all component graphs."""
        return MultiGraphComponentGroup[SampleInputsType](
            {
                (
                    component_name,
                    graph_name,
                ): self.get_component_graph_sample_inputs(
                    component_name,
                    graph_name,
                    input_spec.get((component_name, graph_name))
                    if input_spec
                    else None,
                    use_channel_last_format,
                )
                for component_name, graph_name in self.all_flattened_component_graph_names
            }
        )


WorkbenchModelT = TypeVar("WorkbenchModelT", bound=WorkbenchModel)
MultiGraphWorkbenchModelT = TypeVar(
    "MultiGraphWorkbenchModelT", bound=MultiGraphWorkbenchModel
)


class MultiGraphWorkbenchModelCollection(
    MultiGraphCollectionModel, Generic[MultiGraphWorkbenchModelT, WorkbenchModelT]
):
    """
    Concrete MultiGraphCollectionModel backed by a dict of WorkbenchModel or MultiGraphWorkbenchModel instances.

    Delegates per-component-graph methods to the appropriate model, dispatching based on
    whether the component is a MultiGraphWorkbenchModel (has named graphs) or a plain
    WorkbenchModel (single graph, graph_name=context_graph_name).
    """

    def __init__(
        self, components: dict[str, WorkbenchModelT | MultiGraphWorkbenchModelT]
    ) -> None:
        self.components = components
        for name, component in components.items():
            setattr(self, name, component)

    @property
    def component_names(self) -> list[str]:
        return list(self.components)

    def get_component_graph_names(self, component_name: str) -> list[str]:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.graph_names
        return [component.context_graph_name]

    def get_component_graph_input_spec(
        self, component_name: str, graph_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_input_spec(
                graph_name, **filter_kwargs(component.get_graph_input_spec, kwargs)
            )
        return component.get_input_spec(
            **filter_kwargs(component.get_input_spec, kwargs)
        )

    def get_component_graph_output_spec(
        self, component_name: str, graph_name: str
    ) -> OutputSpec:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_output_spec(graph_name)
        return component.get_output_spec()

    def serialize_component_graph(
        self,
        component_name: str,
        graph_name: str | None,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            assert graph_name is not None
            return component.serialize_graph(graph_name, output_dir, input_spec)
        return component.serialize(output_dir, input_spec)

    def get_component_has_shared_source_model(self, component_name: str) -> bool:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.shared_source_model
        return True

    def get_component_graph_litemp_percentage(
        self, component_name: str, graph_name: str, precision: Precision
    ) -> float | None:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_hub_litemp_percentage(graph_name, precision)
        return component.get_hub_litemp_percentage(precision)

    def get_component_graph_hub_quantize_options(
        self,
        component_name: str,
        graph_name: str,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> str:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_hub_quantize_options(
                graph_name, precision, other_quantize_options
            )
        return component.get_hub_quantize_options(precision, other_quantize_options)

    def get_component_graph_hub_compile_options(
        self,
        component_name: str,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_hub_compile_options(
                graph_name, target_runtime, precision, other_compile_options, device
            )
        return component.get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, graph_name
        )

    def get_component_hub_link_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        return self.components[component_name].get_hub_link_options(
            target_runtime, other_link_options
        )

    def get_component_graph_hub_inference_options(
        self,
        component_name: str,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> str:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_hub_inference_options(
                graph_name, target_runtime, other_inference_options
            )
        return component.get_hub_inference_options(
            target_runtime, other_inference_options
        )

    def get_component_graph_sample_inputs(
        self,
        component_name: str,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        component = self.components[component_name]
        if isinstance(component, MultiGraphWorkbenchModel):
            return component.get_graph_sample_inputs(
                graph_name, input_spec, use_channel_last_format
            )
        return component.sample_inputs(input_spec, use_channel_last_format)

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        for component in self.components.values():
            component.write_supplementary_files(output_dir, metadata)
