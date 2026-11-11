# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.protocols import (
    EvaluatableModelProtocol,
    FromPretrainedProtocol,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    get_channel_last,
    make_torch_inputs,
)
from qai_hub_models.utils.kwarg_helpers import cli_friendly_class_name
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
from qai_hub_models.utils.version_helpers import get_tested_torch_version

__all__ = [
    "BaseModel",
    "PrecompiledWorkbenchModel",
    "PytorchWorkbenchModel",
    "WorkbenchModel",
]


def _model_cls_name(cls_instance: Any) -> str:
    """Model name."""
    # Return the cls_instance ID. Match exactly: qai_hub_models.models.<model_id>.<module>
    parts = type(cls_instance).__module__.split(".")
    if len(parts) == 4 and parts[:2] == ["qai_hub_models", "models"]:
        return parts[2]
    # Class defined outside qai_hub_models.models
    return cli_friendly_class_name(type(cls_instance).__name__)


class WorkbenchModel(ABC, FromPretrainedProtocol, EvaluatableModelProtocol):
    """Base interface for AI Hub Workbench models."""

    # -- Subclasses must implement these --
    @abstractmethod
    def get_input_spec(self, *args: Any, **kwargs: Any) -> InputSpec:
        """
        Returns a map from `{input_name -> (shape, dtype)}`
        specifying the shape and dtype for each input argument.
        """
        ...

    @abstractmethod
    def get_output_spec(self) -> OutputSpec:
        """
        Returns a map from `{output_name -> TensorSpec}` with semantic metadata
        for each output tensor (e.g. io_type, bbox format, description).

        Override in subclasses to provide output metadata for the model.
        """
        ...

    @abstractmethod
    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Convert to an AI Hub Workbench source model appropriate for the export method."""
        ...

    @classmethod
    @abstractmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Self: ...

    # -- Subclasses may override these --
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

    @property
    def name(self) -> str:
        """Model name / identifier."""
        return _model_cls_name(self)

    # Whether to pass --output_names to AI Hub compile jobs.
    pass_output_names_to_compile: bool = True

    @property
    def context_graph_name(self) -> str:
        """The default name used for the graph context when compiling to a QNN Context Binary. May be overriden in the parameters of get_compile_options."""
        return self.name

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        """Returns list of dataset classes on which this model can be evaluated."""
        return []

    def get_evaluator(self) -> BaseEvaluator:
        """Gets a class for evaluating output of this model."""
        raise NotImplementedError("No evaluator is supported for this model.")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Model inference with PyTorch Tensor I/O."""
        raise NotImplementedError("Python inference is not supported by this model.")

    def get_unsupported_reason(
        self, target_runtime: TargetRuntime, device: Device
    ) -> str | None:
        """Report the reason if any combination of runtime and device isn't supported."""
        return None

    def get_hub_litemp_percentage(self, precision: Precision) -> float | None:
        """
        Returns the Lite-MP percentage value for the specified mixed precision quantization.

        This method should be implemented for models that support mixed precision quantization.
        """
        return None

    def component_precision(self) -> Precision:
        """
        If this is a component in a collection model, the parent model may declare
        a "variable" precision, where different components use different precisions.

        Returns
        -------
        Precision
            The precision to which this model should be quantized when the parent
            collection model uses "variable" precision.
        """
        raise NotImplementedError("This model does not support mixed precision.")

    # -- Less likely, but subclasses may override these --

    def get_output_names(self) -> list[str]:
        """
        List of output names. If there are multiple outputs, the order of the names
        should match the order of tuple returned by the model.
        """
        outputs = self.get_output_spec().keys()
        assert outputs, "get_output_spec() is not defined!"
        return list(outputs)

    def get_calibration_dataset_cls(self) -> type[BaseDataset] | None:
        """The dataset class used for calibration when quantizing collection components."""
        return None

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str = ""
    ) -> str:
        """AI Hub Workbench quantize options recommended for the model."""
        litemp_percentage = (
            self.get_hub_litemp_percentage(precision)
            if precision.override_type is not None
            else None
        )
        return build_quantize_options(precision, litemp_percentage, other_options)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench compile options recommended for the model."""
        return build_compile_options(
            target_runtime,
            precision,
            self.get_input_spec(),
            self.get_output_spec(),
            context_graph_name or self.context_graph_name,
            other_compile_options,
            self.pass_output_names_to_compile,
        )

    def get_hub_link_options(
        self,
        target_runtime: TargetRuntime,
        other_link_options: str = "",
    ) -> str:
        """AI Hub Workbench link options recommended for the model."""
        return build_link_options(target_runtime, other_link_options)

    def get_hub_inference_options(
        self,
        target_runtime: TargetRuntime,
        other_inference_options: str = "",
        context_graph_name: str | None = None,
    ) -> str:
        """AI Hub Workbench inference options recommended for the model."""
        return build_inference_options(
            target_runtime, context_graph_name, other_inference_options
        )

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
        input_spec = input_spec or self.get_input_spec()
        inputs = self._sample_inputs_impl(input_spec, **kwargs)
        inputs = expand_to_batch_size(inputs, input_spec)
        if use_channel_last_format and (cl_inputs := get_channel_last(input_spec)):
            return transpose_channel_first_to_last(cl_inputs, inputs)
        return inputs

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """
        Default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.
        """
        if not input_spec:
            input_spec = self.get_input_spec()
        return make_sample_inputs(input_spec)


class PytorchWorkbenchModel(WorkbenchModel, torch.nn.Module):
    """A pre-trained PyTorch model with helpers for submission to AI Hub Workbench."""

    def __init__(
        self,
        model: torch.nn.Module | None = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.eval()
        self.model = cast(torch.nn.Module, model)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        When a new torch.nn.Module attribute is added, we want to set it to eval mode.
        If this model is being trained, calling `model.train()` will reverse all of these.
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

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        """Serialize this model to disk. The serialized model will be uploaded to AI Hub Workbench during export."""
        input_spec = input_spec or self.get_input_spec()
        output_path = Path(output_dir) / f"{self.name}.pt2"
        self.to("cpu").eval()
        try:
            with torch.no_grad():
                exported = torch.export.export(
                    self, tuple(make_torch_inputs(input_spec))
                )
        except Exception as e:
            raise RuntimeError(
                f"torch.export failed with torch=={torch.__version__}. Different "
                "torch versions can export different graphs; this failure may be "
                "due to using a torch version other than the one AI Hub Models is "
                f"tested with (torch=={get_tested_torch_version()}). Original error: {e}"
            ) from e
        torch.export.save(exported, output_path)
        return output_path


# BaseModel is a historical alias of PytorchWorkbenchModel.
# We will continue to refer to Pytorch models as BaseModel in most code
# as AI Hub Models does not accept contributions of non-pytorch-base models.
BaseModel = PytorchWorkbenchModel


class PrecompiledWorkbenchModel(WorkbenchModel):
    """
    A pre-compiled hub model.
    Model PyTorch source is not available, but compiled assets are available.
    """

    def __init__(self, target_model_path: str) -> None:
        self.target_model_path = target_model_path

    # -- Subclasses may override these --

    def get_target_model_path(self) -> str:
        return self.target_model_path

    def serialize(
        self,
        output_dir: str | os.PathLike,
        input_spec: InputSpec | None = None,
    ) -> Path:
        return Path(self.target_model_path)
