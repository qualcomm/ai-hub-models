# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.protocols import FromPretrainedProtocol
from qai_hub_models.utils.base_model import _model_cls_name
from qai_hub_models.utils.export.result import MultiGraphGroup
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    get_channel_last,
    make_torch_inputs,
)
from qai_hub_models.utils.qai_hub_helpers import (
    build_compile_options,
    build_inference_options,
    build_link_options,
    build_quantize_options,
)
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last

__all__ = [
    "MultiGraphWorkbenchModel",
]


class MultiGraphWorkbenchModel(ABC, FromPretrainedProtocol):
    """
    A model composed of multiple independently-compiled graphs that are linked into one executable binary.

    The QAIRT SDK can support a single model file that contains multiple graphs.
    The graphs will share weights on disk when possible.

    Subclasses must implement:
    - `get_graph_names()` — the source of truth for which graphs exist
    - `get_graph_input_spec(graph_name)` — input spec for a single graph
    - `get_graph_output_spec(graph_name)` — output spec for a single graph
    - `convert_to_hub_source_model(...)` — serialize to disk

    The `get_*` methods auto-build dicts from the per-graph `get_graph_*` getters.
    """

    # Whether to pass --output_names to AI Hub compile jobs.
    pass_output_names_to_compile: bool = True

    # -- Subclasses must implement these --
    @property
    @abstractmethod
    def graph_names(self) -> list[str]: ...

    @abstractmethod
    def get_graph_output_spec(self, graph_name: str) -> OutputSpec: ...

    @abstractmethod
    def get_graph_input_spec(
        self, graph_name: str, *args: Any, **kwargs: Any
    ) -> InputSpec: ...

    @abstractmethod
    def serialize_graph(
        self,
        graph_name: str,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path: ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self: ...

    # -- Subclasses may override these --
    @property
    def name(self) -> str:
        return _model_cls_name(self)

    @property
    def shared_source_model(self) -> bool:
        return False

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        if not target_runtime.is_aot_compiled:
            return "Multi-graph models only support AOT-compiled target runtimes."
        return None

    def component_precision(self) -> Precision:
        raise NotImplementedError()

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        return None

    def get_graph_sample_inputs(
        self,
        graph_name: str,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> SampleInputsType:
        spec = input_spec or self.get_graph_input_spec(graph_name)
        inputs_dict: SampleInputsType = {}
        inputs_list = make_torch_inputs(spec)
        for i, input_name in enumerate(spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        if use_channel_last_format and (cl := get_channel_last(spec)):
            inputs_dict = transpose_channel_first_to_last(cl, inputs_dict)
        return inputs_dict

    # -- Per-graph hub option getters (subclasses may override) --

    def get_graph_hub_litemp_percentage(
        self, graph_name: str, precision: Precision
    ) -> float | None:
        return None

    def get_graph_hub_quantize_options(
        self,
        graph_name: str,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> str:
        litemp_percentage = (
            self.get_graph_hub_litemp_percentage(graph_name, precision)
            if precision.override_type is not None
            else None
        )
        return build_quantize_options(
            precision, litemp_percentage, other_quantize_options
        )

    def get_graph_hub_compile_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> str:
        assert target_runtime.is_aot_compiled, (
            "Multi-graph models only support AOT-compiled target runtimes."
        )
        return build_compile_options(
            target_runtime,
            precision,
            self.get_graph_input_spec(graph_name),
            self.get_graph_output_spec(graph_name),
            graph_name,
            other_compile_options,
            self.pass_output_names_to_compile,
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        assert target_runtime.is_aot_compiled, (
            "Multi-graph models only support AOT-compiled target runtimes."
        )
        return build_link_options(target_runtime, other_link_options)

    def get_graph_hub_inference_options(
        self,
        graph_name: str,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> str:
        assert target_runtime.is_aot_compiled, (
            "Multi-graph models only support AOT-compiled target runtimes."
        )
        return build_inference_options(
            target_runtime, graph_name, other_inference_options
        )

    # -- Auto-built from per-graph getters --

    def get_input_spec(self, *args: Any, **kwargs: Any) -> MultiGraphGroup[InputSpec]:
        return MultiGraphGroup(
            {
                name: self.get_graph_input_spec(name, *args, **kwargs)
                for name in self.graph_names
            }
        )

    def get_output_spec(self) -> MultiGraphGroup[OutputSpec]:
        return MultiGraphGroup(
            {name: self.get_graph_output_spec(name) for name in self.graph_names}
        )

    # -- Auto-built hub option dicts --

    def get_hub_quantize_options(
        self,
        precision: Precision,
        other_quantize_options: str = "",
    ) -> MultiGraphGroup[str]:
        return MultiGraphGroup(
            {
                name: self.get_graph_hub_quantize_options(
                    name, precision, other_quantize_options
                )
                for name in self.graph_names
            }
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
    ) -> MultiGraphGroup[str]:
        return MultiGraphGroup(
            {
                name: self.get_graph_hub_compile_options(
                    name, target_runtime, precision, other_compile_options, device
                )
                for name in self.graph_names
            }
        )

    def get_hub_inference_options(
        self,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
    ) -> MultiGraphGroup[str]:
        return MultiGraphGroup(
            {
                name: self.get_graph_hub_inference_options(
                    name, target_runtime, other_inference_options
                )
                for name in self.graph_names
            }
        )

    def sample_inputs(
        self,
        input_spec: MultiGraphGroup[InputSpec] | None = None,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> MultiGraphGroup[SampleInputsType]:
        input_specs = input_spec or self.get_input_spec(**kwargs)
        return MultiGraphGroup(
            {
                graph_name: self.get_graph_sample_inputs(
                    graph_name, spec, use_channel_last_format
                )
                for graph_name, spec in input_specs.items()
            }
        )
