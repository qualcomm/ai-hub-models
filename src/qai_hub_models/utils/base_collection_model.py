# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from qai_hub.client import Device

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common import BaseDataset
from qai_hub_models.protocols import FromPretrainedProtocol
from qai_hub_models.utils._cached_from_pretrained import CachedFromPretrainedMeta
from qai_hub_models.utils.base_model import (
    PrecompiledWorkbenchModel,
    WorkbenchModel,
    _model_cls_name,
)
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    get_channel_last,
)
from qai_hub_models.utils.kwarg_helpers import (
    filter_kwargs,
)
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


class CollectionModel(ABC, FromPretrainedProtocol):
    """
    Abstract interface for models composed of multiple independently-compiled components.

    Each component has its own input/output spec, compile options, and source model.
    Subclasses define the component list and per-component behavior; the "all-component"
    getters aggregate results into ComponentGroup dicts keyed by component name.
    """

    @property
    def name(self) -> str:
        """Model name."""
        return _model_cls_name(self)

    @property
    @abstractmethod
    def component_names(self) -> list[str]:
        """Ordered list of component names in this collection."""
        ...

    @abstractmethod
    def get_component_input_spec(
        self, component_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        """Return the input spec for a single component."""
        ...

    @abstractmethod
    def get_component_output_spec(self, component_name: str) -> OutputSpec:
        """Return the output spec (with semantic metadata) for a single component."""
        ...

    @abstractmethod
    def serialize_component(
        self,
        component_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Serialize a component to disk in a format suitable for AI Hub compilation."""
        ...

    def get_component_context_graph_name(self, component_name: str) -> str:
        return f"{self.name}_{component_name}"

    def get_component_unsupported_reason(
        self, component_name: str, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        """Return a human-readable reason if the component is unsupported, or None."""
        return None

    def get_component_litemp_percentage(
        self, component_name: str, precision: Precision
    ) -> float | None:
        """Return the Lite-MP percentage for mixed-precision quantization, or None."""
        return None

    def get_component_mixed_precision(
        self, component_name: str, precision: Precision
    ) -> Precision:
        """Return the per-component precision when using mixed precision mode."""
        assert precision in (Precision.mixed, Precision.mixed_with_float)
        raise NotImplementedError(
            "Mixed precision is not supported by this collection model."
        )

    def get_component_hub_quantize_options(
        self, component_name: str, precision: Precision, other_options: str = ""
    ) -> str:
        """Return AI Hub quantize option string for a component."""
        litemp_percentage = (
            self.get_component_litemp_percentage(component_name, precision)
            if precision.override_type is not None
            else None
        )
        return build_quantize_options(precision, litemp_percentage, other_options)

    def get_component_hub_compile_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        """Return AI Hub compile option string for a component."""
        if context_graph_name is None and target_runtime.is_aot_compiled:
            context_graph_name = self.get_component_context_graph_name(component_name)
        return build_compile_options(
            target_runtime,
            precision,
            self.get_component_input_spec(component_name),
            self.get_component_output_spec(component_name),
            context_graph_name,
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

    def get_component_hub_inference_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        """Return AI Hub inference option string for a component."""
        return build_inference_options(
            target_runtime, context_graph_name, other_inference_options
        )

    def get_component_sample_inputs(
        self,
        component_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        """Generate sample inputs for a component, optionally in channel-last format."""
        input_spec = input_spec or self.get_component_input_spec(component_name)
        inputs = expand_to_batch_size(make_sample_inputs(input_spec), input_spec)
        if use_channel_last_format and (channel_last := get_channel_last(input_spec)):
            return transpose_channel_first_to_last(channel_last, inputs)
        return inputs

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

    def get_input_spec(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ComponentGroup[InputSpec]:
        """Return input specs for all components as a ComponentGroup."""
        return ComponentGroup(
            {
                name: self.get_component_input_spec(name, *args, **kwargs)
                for name in self.component_names
            }
        )

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        """Return the first unsupported reason across all components, or None."""
        for name in self.component_names:
            if reason := self.get_component_unsupported_reason(
                name, target_runtime, device
            ):
                return f"Component {name}: {reason}"
        return None

    def get_hub_quantize_options(
        self,
        precision: Precision,
        other_options: str = "",
    ) -> ComponentGroup[str]:
        """Return AI Hub quantize options for all components."""
        return ComponentGroup(
            {
                name: self.get_component_hub_quantize_options(
                    name,
                    precision,
                    other_options,
                )
                for name in self.component_names
            }
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> ComponentGroup[str]:
        """Return AI Hub compile options for all components."""
        return ComponentGroup(
            {
                name: self.get_component_hub_compile_options(
                    name,
                    target_runtime,
                    precision,
                    other_compile_options,
                    device,
                )
                for name in self.component_names
            }
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> ComponentGroup[str]:
        """Return AI Hub link options for all components."""
        return ComponentGroup(
            {
                name: self.get_component_hub_link_options(
                    name,
                    target_runtime,
                    other_link_options,
                )
                for name in self.component_names
            }
        )

    def get_hub_inference_options(
        self,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> ComponentGroup[str]:
        """Return AI Hub inference options for all components."""
        return ComponentGroup(
            {
                name: self.get_component_hub_inference_options(
                    name, target_runtime, other_inference_options
                )
                for name in self.component_names
            }
        )

    def get_mixed_precisions(
        self,
        precision: Precision,
    ) -> dict[str, Precision]:
        """Return a mapping of component name to precision for mixed-precision mode."""
        return {
            n: self.get_component_mixed_precision(n, precision)
            for n in self.component_names
        }

    def sample_inputs(
        self,
        input_specs: ComponentGroup[InputSpec] | None = None,
        use_channel_last_format: bool = True,
    ) -> ComponentGroup[SampleInputsType]:
        """Return sample inputs for all components."""
        specs = input_specs or self.get_input_spec()
        return ComponentGroup(
            {
                name: self.get_component_sample_inputs(
                    name, specs.get(name), use_channel_last_format
                )
                for name in self.component_names
            }
        )


WorkbenchModelT = TypeVar("WorkbenchModelT", bound=WorkbenchModel)


class WorkbenchModelCollection(
    CollectionModel, Generic[WorkbenchModelT], metaclass=CachedFromPretrainedMeta
):
    """
    Concrete CollectionModel backed by a dict of WorkbenchModel instances.

    Delegates all per-component methods directly to the corresponding WorkbenchModel,
    making it easy to wrap independently-defined models into a single collection
    without subclassing CollectionModel from scratch.
    """

    def __init__(self, components: dict[str, WorkbenchModelT]) -> None:
        self.components = components
        for name, component in components.items():
            setattr(self, name, component)

    @property
    def component_names(self) -> list[str]:
        return list(self.components)

    def get_component_context_graph_name(self, component_name: str) -> str:
        component = self.components[component_name]
        if component.context_graph_name not in (
            component.__class__.__name__,
            self.name,
        ):
            return component.context_graph_name
        # We'd like to always return component.context_graph_name, but this is legacy behavior.
        # This is what the export script used before model classes determined their own graph name.Expand commentComment on line R609Resolved
        return f"{self.name}_{component_name.lower()}"

    def get_component_input_spec(
        self, component_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        component = self.components[component_name]
        return component.get_input_spec(
            **filter_kwargs(component.get_input_spec, kwargs)
        )

    def get_component_output_spec(self, component_name: str) -> OutputSpec:
        return self.components[component_name].get_output_spec()

    def serialize_component(
        self,
        component_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        component = self.components[component_name]
        component_dir = Path(output_dir) / component_name
        component_dir.mkdir(parents=True, exist_ok=True)
        return component.serialize(component_dir, input_spec)

    def get_component_unsupported_reason(
        self, component_name: str, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        return self.components[component_name].get_unsupported_reason(
            target_runtime, device
        )

    def get_component_litemp_percentage(
        self, component_name: str, precision: Precision
    ) -> float | None:
        return self.components[component_name].get_hub_litemp_percentage(precision)

    def get_component_mixed_precision(
        self, component_name: str, precision: Precision
    ) -> Precision:
        return self.components[component_name].component_precision()

    def get_component_hub_quantize_options(
        self, component_name: str, precision: Precision, other_options: str = ""
    ) -> str:
        return self.components[component_name].get_hub_quantize_options(
            precision, other_options
        )

    def get_component_hub_compile_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if not context_graph_name and target_runtime.is_aot_compiled:
            context_graph_name = self.get_component_context_graph_name(component_name)
        return self.components[component_name].get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
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

    def get_component_hub_inference_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        return self.components[component_name].get_hub_inference_options(
            target_runtime, other_inference_options, context_graph_name
        )

    def get_component_sample_inputs(
        self,
        component_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        return self.components[component_name].sample_inputs(
            input_spec, use_channel_last_format
        )

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        for component in self.components.values():
            component.write_supplementary_files(output_dir, metadata)


class PrecompiledCollectionModel(CollectionModel):
    """
    Abstract interface for precompiled collection models.

    Each component is a PrecompiledWorkbenchModel whose compiled assets are
    available on disk. Subclasses implement component_names and
    from_pretrained(), and provide access to each component's target model path.
    """

    @abstractmethod
    def get_component_target_model_path(self, component_name: str) -> str:
        """Return the path to the compiled model asset for a component."""
        ...


PrecompiledModelT = TypeVar("PrecompiledModelT", bound=PrecompiledWorkbenchModel)


class PrecompiledWorkbenchModelCollection(
    PrecompiledCollectionModel, Generic[PrecompiledModelT]
):
    """
    Concrete PrecompiledCollectionModel backed by a dict of PrecompiledWorkbenchModel instances.

    Delegates all per-component methods directly to the corresponding PrecompiledWorkbenchModel,
    following the same pattern as WorkbenchModelCollection for pretrained models.
    """

    def __init__(self, components: dict[str, PrecompiledModelT]) -> None:
        self.components = components

    @property
    def component_names(self) -> list[str]:
        return list(self.components)

    def get_component_target_model_path(self, component_name: str) -> str:
        return self.components[component_name].get_target_model_path()

    def get_component_input_spec(
        self, component_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec:
        component = self.components[component_name]
        return component.get_input_spec(
            **filter_kwargs(component.get_input_spec, kwargs)
        )

    def get_component_output_spec(self, component_name: str) -> OutputSpec:
        return self.components[component_name].get_output_spec()

    def serialize_component(
        self,
        component_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        return Path(
            cast(
                str,
                self.components[component_name].serialize(Path(output_dir), input_spec),
            )
        )

    def get_component_unsupported_reason(
        self, component_name: str, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        return self.components[component_name].get_unsupported_reason(
            target_runtime, device
        )

    def get_component_hub_compile_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        raise NotImplementedError()

    def get_component_hub_inference_options(
        self,
        component_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        return self.components[component_name].get_hub_inference_options(
            target_runtime, other_inference_options, context_graph_name
        )

    def get_component_sample_inputs(
        self,
        component_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        return self.components[component_name].sample_inputs(
            input_spec, use_channel_last_format
        )

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        for component in self.components.values():
            component.write_supplementary_files(output_dir, metadata)
