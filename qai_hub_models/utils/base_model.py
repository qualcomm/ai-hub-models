# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import inspect
import os
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator
from qai_hub_models.models.common import (
    Precision,
    QAIRTVersion,
    SampleInputsType,
    SourceModelFormat,
    TargetRuntime,
)
from qai_hub_models.models.protocols import (
    ExecutableModelProtocol,
    FromPrecompiledProtocol,
    FromPretrainedProtocol,
    HubModelProtocol,
    PretrainedHubModelProtocol,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.input_spec import (
    InputSpec,
    broadcast_data_to_multi_batch,
    get_batch_size,
    make_torch_inputs,
)
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT
from qai_hub_models.utils.transpose_channel import transpose_channel_first_to_last

CollectionModelT = TypeVar("CollectionModelT", bound="CollectionModel")


class CollectionModel:
    """
    Model that glues together several BaseModels or BasePrecompiledModel.

    See test_base_model.py for usage examples.
    """

    component_classes: list[type[BaseModel]] = []
    component_class_names: list[str] = []

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Make copies of the list for each subclass so they don't share state
        cls.component_classes = list(cls.component_classes)
        cls.component_class_names = list(cls.component_class_names)

    def __init__(self, *args: BaseModel | BasePrecompiledModel) -> None:
        component_names = type(self).component_class_names
        components: dict[
            str, BaseModel | BasePrecompiledModel
        ] = {}  # class name -> instantiated object

        # Process positional arguments.
        if len(args) != len(component_names):
            raise ValueError(
                f"CollectionModel has {len(component_names)} ordered arguments, "
                "each of which should correspond with a single component."
            )
        for i, (name, arg) in enumerate(zip(component_names, args, strict=False)):
            expected_class = type(self).component_classes[i]
            if not isinstance(arg, (expected_class, ExecutableModelProtocol)):
                raise TypeError(
                    f"Expected component '{name}' to be an instance "
                    f"of {name} or ExecutableModelProtocol, got {type(arg).__name__}"
                )
            assert isinstance(arg, (BaseModel, BasePrecompiledModel))
            components[name] = arg

        # Check that all required components are provided.
        missing = [name for name in component_names if name not in components]
        if missing:
            raise ValueError(f"Missing components for: {missing}")

        self.components = components

    @classmethod
    def add_component(
        cls,
        component_class: type,
        component_name: str | None = None,
        subfolder_hf: str | None = None,
    ) -> Callable[[type[CollectionModelT]], type[CollectionModelT]]:
        """
        Decorator to add a component (a subclass of BaseModel or
        BasePrecompiledModel) to a CollectionModel.  The component is
        inserted at the beginning of the class-level dictionary `component_classes`,
        so that the outer decorator appears first.

        See test_base_model.py for usage examples.

        Parameters
        ----------
        component_class
            Component class to add to the CollectionModel.
        component_name
            Name the component. By default uses component_class.__name__.
        subfolder_hf
            By default the same as component_name. Specify this
            only when Huggingface uses a different subfolder name than the desired
            component_name. For example, in ControlNet the ControlNet model is not in
            any subfolder on HF, so subfolder_hf = "" even though we want
            to name our component "controlnet".

        Returns
        -------
        callable : Callable[[type[CollectionModelT]], type[CollectionModelT]]
            Decorator function that registers the component on the CollectionModel subclass.
        """

        def decorator(subclass: type[CollectionModelT]) -> type[CollectionModelT]:
            name = component_name or component_class.__name__
            if name in subclass.component_class_names:
                raise ValueError(f"Component with name {name} already registered")
            # prepend list
            subclass.component_classes.insert(0, component_class)
            subclass.component_class_names.insert(0, name)

            # Component class from_pretrained would look for default_subfolder
            # under checkpoint dir if checkpoint is local, or
            # default_subfolder_hf if checkpoint is on HF. Typically
            # default_subfolder == default_subfolder_hf
            # Allow @add_component(Klass, "") to enforce having no subfolder.
            # This is needed for controlnet where the controlnet is from a
            # different repo without subfolders
            subfolder = subfolder_hf if subfolder_hf is not None else name
            component_class.default_subfolder = component_name  # type: ignore[attr-defined]
            component_class.default_subfolder_hf = subfolder  # type: ignore[attr-defined]
            return subclass

        return decorator

    @staticmethod
    def reset_components() -> Callable[[type[CollectionModel]], type[CollectionModel]]:
        """
        Decorator to erase all components set on a CollectionModel.
        Useful when subclassing a CollectionModel and overriding component classes.

        See test_base_model.py for usage examples.
        """

        def decorator(subclass: type[CollectionModel]) -> type[CollectionModel]:
            subclass.component_classes = []
            subclass.component_class_names = []
            return subclass

        return decorator

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Returns list of strings with names of all datasets on which
        this model can be evaluated.

        All names must be registered in qai_hub_models/datasets/__init__.py
        """
        return []

    def sample_inputs(
        self,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
    ) -> dict[str, SampleInputsType]:
        return {
            component_name: component.sample_inputs(
                input_spec=input_spec, use_channel_last_format=use_channel_last_format
            )
            for component_name, component in self.components.items()
        }

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
    ) -> dict[str, str]:
        return {
            component_name: component.get_hub_profile_options(
                target_runtime=target_runtime,
                other_profile_options=other_profile_options,
            )
            for component_name, component in self.components.items()
        }

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
            metadata.precision, metadata.runtime, metadata.tool_versions, and metadata.model_files should be pre-populated by the caller.

        Returns
        -------
        None
            metadata.supplementary_files will be populated with the files written by this function.
        """
        return


class PretrainedCollectionModel(CollectionModel, FromPretrainedProtocol):
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CheckpointSpec = "DEFAULT",
        host_device: torch.device | str = torch.device("cpu"),
        **kwargs: Any,  # any extra you might want to forward
    ) -> Self:
        """
        Instantiate the collection by delegating to each component_cls.from_pretrained,
        but only passing it the arguments it actually declares.
        """
        # common kwargs we'd like to pass
        base_kwargs: dict[str, Any] = {
            "checkpoint": checkpoint,
            "host_device": host_device,
            **kwargs,
        }

        # helper to call any fn with only the supported subset of base_kwargs
        def _call_with_supported(fn: Callable, all_kwargs: dict[str, Any]) -> Any:
            sig = inspect.signature(fn)
            params = sig.parameters.values()

            # If fn accepts **kwargs, just pass everything through
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
                return fn(**all_kwargs)

            # Otherwise, filter to only the names fn explicitly declares
            supported = {
                name: value
                for name, value in all_kwargs.items()
                if name in sig.parameters
            }
            return fn(**supported)

        components = []
        for _name, component_cls in zip(
            cls.component_class_names, cls.component_classes, strict=False
        ):
            # call component_cls.from_pretrained but only with the args it accepts
            comp = _call_with_supported(component_cls.from_pretrained, base_kwargs)
            components.append(comp)

        # now build and return your collection model however CollectionModel expects
        return cls(*components)


class PrecompiledCollectionModel(CollectionModel, FromPrecompiledProtocol):
    @classmethod
    def from_precompiled(cls) -> Self:
        """
        Instantiate the CollectionModel by instantiating all registered components
        using their own from_precompiled() methods.
        """
        components = []
        for component_cls in cls.component_classes:
            if not (
                hasattr(component_cls, "from_precompiled")
                and callable(component_cls.from_precompiled)
            ):
                raise AttributeError(
                    f"Component '{component_cls.__name__}' does not have a callable from_precompiled method"
                )
            components.append(component_cls.from_precompiled())
        return cls(*components)


class HubModel(HubModelProtocol):
    """Base interface for AI Hub Workbench models."""

    def __init__(self) -> None:
        # If a child class implements _get_input_spec_for_instance(),
        # then calling `get_input_spec` on the instance will redirect to it.
        if self._get_input_spec_for_instance.__module__ != __name__:
            self.get_input_spec = self._get_input_spec_for_instance
        if self._get_output_names_for_instance.__module__ != __name__:
            self.get_output_names = self._get_output_names_for_instance
        if self._get_channel_last_inputs_for_instance.__module__ != __name__:
            self.get_channel_last_inputs = self._get_channel_last_inputs_for_instance
        if self._get_channel_last_outputs_for_instance.__module__ != __name__:
            self.get_channel_last_outputs = self._get_channel_last_outputs_for_instance

    def _get_input_spec_for_instance(self, *args: Any, **kwargs: Any) -> InputSpec:
        """
        Get the input specifications for an instance of this model.

        Typically this will pre-fill inputs of get_input_spec
        with values determined by instance members of the model class.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_input_spec with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_output_names_for_instance(self, *args: Any, **kwargs: Any) -> list[str]:
        """
        Get the output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_output_names with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_channel_last_inputs_for_instance(
        self, *args: Any, **kwargs: Any
    ) -> list[str]:
        """
        Get the channel last input names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_channel_last_inputs with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def _get_channel_last_outputs_for_instance(
        self, *args: Any, **kwargs: Any
    ) -> list[str]:
        """
        Get the channel last output names for an instance of this model.

        If this function is implemented by a child class, the initializer for BaseModel
        will automatically override get_channel_last_outputs with this function
        when the class is instantiated.
        """
        raise NotImplementedError

    def sample_inputs(
        self,
        input_spec: InputSpec | None = None,
        use_channel_last_format: bool = True,
        **kwargs: Any,
    ) -> SampleInputsType:
        """
        Returns a set of sample inputs for the model.

        For each input name in the model, a list of numpy arrays is provided.
        If the returned set is batch N, all input names must contain exactly N numpy arrays.

        Subclasses should NOT override this. They should instead override _sample_inputs_impl.

        This function will invoke _sample_inputs_impl and then apply any required channel
            format transposes.
        """
        sample_inputs = self._sample_inputs_impl(input_spec, **kwargs)
        if input_spec is not None:
            batch_size = get_batch_size(input_spec)
            if batch_size is not None and batch_size > 1:
                sample_inputs = broadcast_data_to_multi_batch(input_spec, sample_inputs)
        if use_channel_last_format and self.get_channel_last_inputs():
            return transpose_channel_first_to_last(
                self.get_channel_last_inputs(), sample_inputs
            )
        return sample_inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        See the `sample_inputs` doc for the expected format.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        inputs_dict = {}
        inputs_list = make_torch_inputs(input_spec)
        for i, input_name in enumerate(input_spec.keys()):
            inputs_dict[input_name] = [inputs_list[i].numpy()]
        return inputs_dict

    def get_hub_profile_options(
        self,
        target_runtime: TargetRuntime,
        other_profile_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench profile options recommended for the model."""
        if QAIRTVersion.HUB_FLAG not in other_profile_options:
            other_profile_options = (
                other_profile_options
                + f" {target_runtime.default_qairt_version.hub_option}"
            )

        if context_graph_name is not None:
            if not target_runtime.is_aot_compiled:
                raise ValueError(
                    "Cannot specify a context binary graph name if the target is not precompiled QAIRT."
                )
            other_profile_options += (
                f" --qnn_options context_enable_graphs={context_graph_name}"
            )

        return other_profile_options

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        """AI Hub Workbench link options recommended for the model."""
        if QAIRTVersion.HUB_FLAG not in other_link_options:
            other_link_options = (
                other_link_options
                + f" {target_runtime.default_qairt_version.hub_option}"
            )

        return other_link_options

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        """
        A list of input names that should be transposed to channel-last format
            for the on-device model in order to improve performance.
        """
        return []

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        """
        A list of output names that should be transposed to channel-last format
            for the on-device model in order to improve performance.
        """
        return []

    @staticmethod
    def component_precision() -> Precision:
        """
        If this is a component in a component model, the parent model may declare
        a "variable" precision, where different components use different precisions.

        Returns
        -------
        Precision
            The precision to which this model should be quantized when the parent component
            model uses "variable" precision.
        """
        raise NotImplementedError()


class BaseModel(
    torch.nn.Module,
    HubModel,
    PretrainedHubModelProtocol,
    ExecutableModelProtocol,
):
    """A pre-trained PyTorch model with helpers for submission to AI Hub Workbench."""

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        torch.nn.Module.__init__(self)  # Initialize Torch Module
        HubModel.__init__(self)  # Initialize Hub Model
        self.eval()
        if model is not None:
            self.model = model

    def __setattr__(self, name: str, value: Any) -> None:
        """
        When a new torch.nn.Module attribute is added, we want to set it to eval mode.
            If this model is being trained, calling `model.train()`
            will reverse all of these.
        """
        if isinstance(value, torch.nn.Module) and not self.training:
            value.eval()
        torch.nn.Module.__setattr__(self, name, value)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        If a model is in eval mode (which equates to self.training == False),
            we don't want to compute gradients when doing the forward pass.
        """
        context_fn = nullcontext if self.training else torch.no_grad
        with context_fn():
            return torch.nn.Module.__call__(self, *args, **kwargs)

    def convert_to_torchscript(
        self, input_spec: InputSpec | None = None, check_trace: bool = True
    ) -> Any:
        """
        Converts the torch module to a torchscript trace, which
        is the format expected by qai hub.

        This is a default implementation that may be overriden by a subclass.
        """
        if not input_spec:
            input_spec = self.get_input_spec()

        # Torchscript should never be trained, so disable gradients for all parameters.
        # Need to do this on a model copy, in case the original model is being trained.
        model_copy = deepcopy(self)
        for param in model_copy.parameters():
            param.requires_grad = False

        return torch.jit.trace(
            model_copy, make_torch_inputs(input_spec), check_trace=check_trace
        )

    def convert_to_hub_source_model(
        self,
        target_runtime: TargetRuntime,
        output_path: str | Path,
        input_spec: InputSpec | None = None,
        check_trace: bool = True,
        external_onnx_weights: bool = False,
        output_names: list[str] | None = None,
    ) -> str | None:
        """Convert to a AI Hub Workbench source model appropriate for the export method."""
        # Local import to prevent circular dependency
        from qai_hub_models.utils.inference import prepare_compile_zoo_model_to_hub

        return prepare_compile_zoo_model_to_hub(
            self,
            source_model_format=self.preferred_hub_source_model_format(target_runtime),
            target_runtime=target_runtime,
            output_path=output_path,
            input_spec=input_spec,
            check_trace=check_trace,
            external_onnx_weights=external_onnx_weights,
            output_names=output_names or self.get_output_names(),
        )

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench compile options recommended for the model."""
        compile_options = ""
        if "--target_runtime" not in other_compile_options:
            compile_options = target_runtime.aihub_target_runtime_flag
        if (
            QAIRTVersion.HUB_FLAG not in other_compile_options
            and target_runtime.qairt_version_changes_compilation
        ):
            compile_options = (
                compile_options + f" {target_runtime.default_qairt_version.hub_option}"
            )

        compile_options += f" --output_names {','.join(self.get_output_names())}"

        if target_runtime != TargetRuntime.ONNX:
            if self.get_channel_last_inputs():
                channel_last_inputs = ",".join(self.get_channel_last_inputs())
                compile_options += f" --force_channel_last_input {channel_last_inputs}"
            if self.get_channel_last_outputs():
                channel_last_outputs = ",".join(self.get_channel_last_outputs())
                compile_options += (
                    f" --force_channel_last_output {channel_last_outputs}"
                )

        if precision.activations_type is not None:
            compile_options += " --quantize_io"
            if target_runtime == TargetRuntime.TFLITE:
                # uint8 is the easiest I/O type for integration purposes,
                # especially for image applications. Images are always
                # uint8 RGB when coming from disk or a camera.
                #
                # Uint8 has not been thoroughly tested with other paths,
                # so it is enabled only for TF Lite today.
                compile_options += " --quantize_io_type uint8"

        if target_runtime.is_aot_compiled:
            # Without this flag, graph names are random.
            # Scorecard job caching relies on models keeping the same md5 hash if they haven't changed.
            # By always setting this flag, we remove a source of randomness in compiled QNN assets.
            compile_options += f" --qnn_options context_enable_graphs={context_graph_name or 'default_graph'}"

        if other_compile_options != "":
            return compile_options + " " + other_compile_options

        return compile_options

    def preferred_hub_source_model_format(
        self, target_runtime: TargetRuntime
    ) -> SourceModelFormat:
        """Source model format preferred for conversion on AI Hub Workbench."""
        return SourceModelFormat.TORCHSCRIPT

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        """
        Report the reason if any combination of runtime and device isn't
        supported.
        """
        return None

    @staticmethod
    def eval_datasets() -> list[str]:
        """
        Returns list of strings with names of all datasets on which
        this model can be evaluated.

        All names must be registered in qai_hub_models/datasets/__init__.py
        """
        return []

    def get_evaluator(self) -> BaseEvaluator:
        """Gets a class for evaluating output of this model."""
        raise NotImplementedError("No evaluator is supported for this model.")

    @staticmethod
    def calibration_dataset_name() -> str | None:
        """
        Name of the dataset to use for calibration when quantizing the model.

        Must be registered in qai_hub_models/datasets/__init__.py
        """
        return None

    @classmethod
    def get_labels_file_name(cls) -> str | None:
        """
        Returns the name of the labels file for this model.

        The labels file should exist in qai_hub_models/labels/ directory.

        Returns
        -------
        str | None
            Name of the labels file (e.g., "coco_labels.txt"), or None if no labels file.
        """
        return None

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        """
        Return the AI Hub Workbench quantize options for the given model precision.

        Generates CLI flags used during AI Hub Workbench quantization.

        - For `w8a8` precision, the default `range_scheme` is `mse_minimizer`.
        - For `w8a16` and mixed-precision profiles, the `range_scheme` is set to `min_max`.
        - For mixed-precision profiles, additional flags are included to specify the percentage and override quantization type (`int16` or `fp16`).
        """
        all_options = other_options or ""
        litemp_percentage = (
            self.get_hub_litemp_percentage(precision)
            if precision.override_type is not None
            else None
        )
        precision_options = precision.get_hub_quantize_options(litemp_percentage)
        if all_options and precision_options:
            all_options += " "
        all_options += precision_options
        return all_options

    @staticmethod
    def get_hub_litemp_percentage(precision: Precision) -> float:
        """
        Returns the Lite-MP percentage value for the specified mixed precision quantization.

        This method should be implemented for the models that support mixed precision quantization.

        NOTE: precision parameter is only included in the method signature to maintain compatibility with the base class.
        """
        raise NotImplementedError(
            f"Mixed precision {precision} is not supported for this model."
        )

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
            metadata.precision, metadata.runtime, metadata.tool_versions, and metadata.model_files should be pre-populated by the caller.

        Returns
        -------
        None
            metadata.supplementary_files will be populated with the files written by this function.
        """
        if labels_file_name := self.get_labels_file_name():
            out_path = Path(output_dir) / "labels.txt"
            labels_path = QAIHM_PACKAGE_ROOT / "labels" / labels_file_name
            shutil.copyfile(labels_path, out_path)
            metadata.supplementary_files["labels.txt"] = (
                "Mapping of model prediction indices -> string labels."
            )


class BasePrecompiledModel(HubModel, FromPrecompiledProtocol):
    """
    A pre-compiled hub model.
    Model PyTorch source is not available, but compiled assets are available.
    """

    def __init__(self, target_model_path: str) -> None:
        self.target_model_path = target_model_path

    def get_target_model_path(self) -> str:
        """Get the path to the compiled asset for this model on disk."""
        return self.target_model_path

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> None | str:
        """
        Report the reason if any combination of runtime and device isn't
        supported.
        """
        return None

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
            metadata.precision, metadata.runtime, metadata.tool_versions, and metadata.model_files should be pre-populated by the caller.

        Returns
        -------
        None
            metadata.supplementary_files will be populated with the files written by this function.
        """
        return
