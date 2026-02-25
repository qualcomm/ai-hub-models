# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Metadata YAML schema for model I/O specifications.

This module defines the structure for metadata.yaml files that document
model input/output specifications and quantization parameters for a single
exported model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.base_config import BaseQAIHMConfig

if TYPE_CHECKING:
    import qai_hub as hub


class QuantizationParameters(BaseQAIHMConfig):
    """Quantization parameters for a tensor."""

    scale: float
    zero_point: int


class TensorSpec(BaseQAIHMConfig):
    """
    Specification for an input or output tensor.

    Extends the existing InputSpec format with additional metadata fields
    for documentation and quantization parameters.
    """

    shape: list[int]
    dtype: str
    description: str | None = None
    quantization_parameters: QuantizationParameters | None = None


class ModelFileMetadata(BaseQAIHMConfig):
    """
    Metadata for a single exported model file.

    This represents the I/O specifications for one exported model component,
    including input/output tensor shapes, dtypes, and quantization parameters.
    """

    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]

    @classmethod
    def from_hub_model(cls, hub_model: hub.Model) -> ModelFileMetadata:
        """
        Create ModelFileMetadata directly from hub.Model.

        Parameters
        ----------
        hub_model
            The hub.Model from compile_job.get_target_model()

        Returns
        -------
        metadata: ModelFileMetadata
            Created model file metadata
        """
        inputs = {}
        outputs = {}

        # Extract inputs from hub.Model.input_spec
        for tensor_specs in hub_model.input_spec.values():
            for tensor_spec in tensor_specs:
                quant_params = None
                if tensor_spec.scale is not None and tensor_spec.zero_point is not None:
                    quant_params = QuantizationParameters(
                        scale=tensor_spec.scale,
                        zero_point=int(tensor_spec.zero_point),
                    )
                assert tensor_spec.shape is not None
                inputs[tensor_spec.name] = TensorSpec(
                    shape=list(tensor_spec.shape),
                    dtype=tensor_spec.dtype,
                    quantization_parameters=quant_params,
                )

        # Extract outputs from hub.Model.output_spec
        for tensor_specs in hub_model.output_spec.values():
            for tensor_spec in tensor_specs:
                quant_params = None
                if tensor_spec.scale is not None and tensor_spec.zero_point is not None:
                    quant_params = QuantizationParameters(
                        scale=tensor_spec.scale,
                        zero_point=int(tensor_spec.zero_point),
                    )
                assert tensor_spec.shape is not None
                outputs[tensor_spec.name] = TensorSpec(
                    shape=list(tensor_spec.shape),
                    dtype=tensor_spec.dtype,
                    quantization_parameters=quant_params,
                )

        return cls(inputs=inputs, outputs=outputs)

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = False,
        delete_if_empty: bool = True,
        flow_lists: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Override to_yaml to default flow_lists=True for metadata files.

        This ensures lists (like tensor shapes) are formatted in flow style
        (e.g., [1, 3, 224, 224]) instead of block style for better readability.
        """
        return super().to_yaml(
            path=path,
            write_if_empty=write_if_empty,
            delete_if_empty=delete_if_empty,
            flow_lists=flow_lists,
            **kwargs,
        )


class ModelMetadata(BaseQAIHMConfig):
    """
    Metadata for a model that may have multiple model files.

    For single-file models (e.g., ResNet50), this will have one entry.
    For multi-file models (e.g., Stable Diffusion with text_encoder.bin, unet.bin, vae_decoder.bin),
    this will have multiple entries mapping model file names to their metadata.

    The keys in model_files are the actual file names (relative to the root of the export directory),
    making it clear what each metadata entry corresponds to in the exported files.

    This is the class that gets saved to disk as metadata.yaml.

    Attributes
    ----------
    runtime
        The target runtime for which the model was exported.
    precision
        The precision configuration for which the model was exported
    tool_versions
        ToolVersions object containing version information for tools used to compile/export the model.
        Includes fields like tflite, onnx, onnx_runtime, qairt, and ai_hub_models versions.
    model_files
        Dictionary mapping model file names to their metadata (I/O specs, quantization params).
    supplementary_files
        Optional dictionary mapping supplementary file names to their descriptions.
            key: file name (relative to root of export directory)
            value: description of the file's contents and purpose
        This can be populated by the model's write_supplementary_files method to document any
        additional files included in the export that are not model files (e.g., labels, config files).
    """

    runtime: TargetRuntime
    precision: Precision
    tool_versions: ToolVersions
    model_files: dict[str, ModelFileMetadata]
    supplementary_files: dict[str, str] = {}

    def to_yaml(
        self,
        path: str | Path,
        write_if_empty: bool = False,
        delete_if_empty: bool = True,
        flow_lists: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Override to_yaml to default flow_lists=True for metadata files.

        This ensures lists (like tensor shapes) are formatted in flow style
        (e.g., [1, 3, 224, 224]) instead of block style for better readability.
        """
        return super().to_yaml(
            path=path,
            write_if_empty=write_if_empty,
            delete_if_empty=delete_if_empty,
            flow_lists=flow_lists,
            **kwargs,
        )
