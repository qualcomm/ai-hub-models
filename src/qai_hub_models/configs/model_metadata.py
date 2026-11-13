# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Metadata schema for model I/O specifications.

This module defines the structure for metadata.json files that document
model input/output specifications and quantization parameters for a single
exported model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import qai_hub as hub
from qai_hub_models_cli.proto import model_metadata_pb2, platform_pb2
from qai_hub_models_cli.proto_helpers._common import (
    read_proto_json,
    read_proto_yaml,
    write_proto_json,
    write_proto_yaml,
)

import qai_hub_models._version as pkg_version
from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import MODEL_USE_CASE
from qai_hub_models.configs.chipset_yaml import (
    ChipsetYaml,
    WebsiteWorld,
    website_world_from_proto,
)
from qai_hub_models.configs.proto_helpers import (
    precision_from_proto,
    precision_to_proto,
    runtime_from_proto,
    runtime_to_proto,
    use_case_from_proto,
    use_case_to_proto,
)
from qai_hub_models.configs.tensor_spec import (
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.device import RegisteredDevice
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec


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
        inputs: dict[str, TensorSpec] = {}
        outputs: dict[str, TensorSpec] = {}

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
                    shape=tuple(tensor_spec.shape),
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
                    shape=tuple(tensor_spec.shape),
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
        exclude_defaults: bool = True,
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
            exclude_defaults=exclude_defaults,
            **kwargs,
        )

    def to_proto(self, filename: str) -> model_metadata_pb2.ModelFileMetadata:
        return model_metadata_pb2.ModelFileMetadata(
            filename=filename,
            inputs=[spec.to_proto(name) for name, spec in self.inputs.items()],
            outputs=[spec.to_proto(name) for name, spec in self.outputs.items()],
        )

    @classmethod
    def from_proto(
        cls, proto: model_metadata_pb2.ModelFileMetadata
    ) -> ModelFileMetadata:
        """Build from the proto. The file name (``proto.filename``) is left to the caller."""
        return cls(
            inputs={s.name: TensorSpec.from_proto(s) for s in proto.inputs},
            outputs={s.name: TensorSpec.from_proto(s) for s in proto.outputs},
        )


class GenieChatTemplate(BaseQAIHMConfig):
    """Chat template tokens and defaults for Genie SDK."""

    global_prefix: str = ""
    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = ""
    user_suffix: str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    vision_start: str = ""
    vision_end: str = ""
    default_system_prompt: str = ""
    think_start: str = ""
    think_end: str = ""
    think_header: str = ""

    def to_proto(self) -> model_metadata_pb2.GenieChatTemplate:
        return model_metadata_pb2.GenieChatTemplate(
            global_prefix=self.global_prefix,
            system_prefix=self.system_prefix,
            system_suffix=self.system_suffix,
            user_prefix=self.user_prefix,
            user_suffix=self.user_suffix,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            vision_start=self.vision_start,
            vision_end=self.vision_end,
            default_system_prompt=self.default_system_prompt,
            think_start=self.think_start,
            think_end=self.think_end,
            think_header=self.think_header,
        )

    @classmethod
    def from_proto(
        cls, proto: model_metadata_pb2.GenieChatTemplate
    ) -> GenieChatTemplate:
        return cls(
            global_prefix=proto.global_prefix,
            system_prefix=proto.system_prefix,
            system_suffix=proto.system_suffix,
            user_prefix=proto.user_prefix,
            user_suffix=proto.user_suffix,
            assistant_prefix=proto.assistant_prefix,
            assistant_suffix=proto.assistant_suffix,
            vision_start=proto.vision_start,
            vision_end=proto.vision_end,
            default_system_prompt=proto.default_system_prompt,
            think_start=proto.think_start,
            think_end=proto.think_end,
            think_header=proto.think_header,
        )


class GeniePipelineConnection(BaseQAIHMConfig):
    """A connection between two nodes in a Genie pipeline."""

    producer_node: str
    producer_node_io: str
    consumer_node: str
    consumer_node_io: str

    def to_proto(
        self,
    ) -> model_metadata_pb2.GenieMetadata.GeniePipeline.GeniePipelineConnection:
        return model_metadata_pb2.GenieMetadata.GeniePipeline.GeniePipelineConnection(
            producer_node=self.producer_node,
            producer_node_io=self.producer_node_io,
            consumer_node=self.consumer_node,
            consumer_node_io=self.consumer_node_io,
        )

    @classmethod
    def from_proto(
        cls,
        proto: model_metadata_pb2.GenieMetadata.GeniePipeline.GeniePipelineConnection,
    ) -> GeniePipelineConnection:
        return cls(
            producer_node=proto.producer_node,
            producer_node_io=proto.producer_node_io,
            consumer_node=proto.consumer_node,
            consumer_node_io=proto.consumer_node_io,
        )


class GenieSampleInput(BaseQAIHMConfig):
    """A sample input binding for a Genie pipeline node."""

    node: str
    node_io: str
    file: str

    def to_proto(self) -> model_metadata_pb2.GenieMetadata.GenieSampleInput:
        return model_metadata_pb2.GenieMetadata.GenieSampleInput(
            node=self.node,
            node_io=self.node_io,
            file=self.file,
        )

    @classmethod
    def from_proto(
        cls, proto: model_metadata_pb2.GenieMetadata.GenieSampleInput
    ) -> GenieSampleInput:
        return cls(node=proto.node, node_io=proto.node_io, file=proto.file)


class GeniePipeline(BaseQAIHMConfig):
    """Pipeline topology for Genie SDK."""

    nodes: dict[str, str]
    connections: list[GeniePipelineConnection]

    def to_proto(self) -> model_metadata_pb2.GenieMetadata.GeniePipeline:
        return model_metadata_pb2.GenieMetadata.GeniePipeline(
            nodes=self.nodes,
            connections=[c.to_proto() for c in self.connections],
        )

    @classmethod
    def from_proto(
        cls, proto: model_metadata_pb2.GenieMetadata.GeniePipeline
    ) -> GeniePipeline:
        return cls(
            nodes=dict(proto.nodes),
            connections=[
                GeniePipelineConnection.from_proto(c) for c in proto.connections
            ],
        )


class GenieVisionPreprocessing(BaseQAIHMConfig):
    """Vision encoder preprocessing parameters for Genie SDK."""

    image_width: int
    image_height: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    normalize_mean: list[float]
    normalize_std: list[float]

    def to_proto(self) -> model_metadata_pb2.GenieMetadata.GenieVisionPreprocessing:
        return model_metadata_pb2.GenieMetadata.GenieVisionPreprocessing(
            image_width=self.image_width,
            image_height=self.image_height,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            spatial_merge_size=self.spatial_merge_size,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        )

    @classmethod
    def from_proto(
        cls, proto: model_metadata_pb2.GenieMetadata.GenieVisionPreprocessing
    ) -> GenieVisionPreprocessing:
        return cls(
            image_width=proto.image_width,
            image_height=proto.image_height,
            patch_size=proto.patch_size,
            temporal_patch_size=proto.temporal_patch_size,
            spatial_merge_size=proto.spatial_merge_size,
            normalize_mean=list(proto.normalize_mean),
            normalize_std=list(proto.normalize_std),
        )


class GenieMetadata(BaseQAIHMConfig):
    """Genie SDK metadata for on-device deployment."""

    chat_template: GenieChatTemplate
    context_lengths: list[int]
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_thinking: bool = False
    pipeline: GeniePipeline | None = None
    sample_inputs: list[GenieSampleInput] | None = None
    vision_preprocessing: GenieVisionPreprocessing | None = None

    def to_proto(self) -> model_metadata_pb2.GenieMetadata:
        return model_metadata_pb2.GenieMetadata(
            chat_template=self.chat_template.to_proto(),
            context_lengths=self.context_lengths,
            supports_streaming=self.supports_streaming,
            supports_vision=self.supports_vision,
            supports_thinking=self.supports_thinking,
            pipeline=self.pipeline.to_proto() if self.pipeline else None,
            sample_inputs=[s.to_proto() for s in self.sample_inputs]
            if self.sample_inputs
            else [],
            vision_preprocessing=self.vision_preprocessing.to_proto()
            if self.vision_preprocessing
            else None,
        )

    @classmethod
    def from_proto(cls, proto: model_metadata_pb2.GenieMetadata) -> GenieMetadata:
        return cls(
            chat_template=GenieChatTemplate.from_proto(proto.chat_template),
            context_lengths=list(proto.context_lengths),
            supports_streaming=proto.supports_streaming,
            supports_vision=proto.supports_vision,
            supports_thinking=proto.supports_thinking,
            pipeline=(
                GeniePipeline.from_proto(proto.pipeline)
                if proto.HasField("pipeline")
                else None
            ),
            sample_inputs=[GenieSampleInput.from_proto(s) for s in proto.sample_inputs]
            or None,
            vision_preprocessing=(
                GenieVisionPreprocessing.from_proto(proto.vision_preprocessing)
                if proto.HasField("vision_preprocessing")
                else None
            ),
        )


class ChipsetAttributes(ChipsetYaml):
    """Chipset attributes for model metadata - adds name field to ChipsetYaml."""

    name: str

    def to_proto(self) -> platform_pb2.ChipsetInfo:  # type: ignore[override]
        return super().to_proto(self.name)

    @classmethod
    def from_proto(cls, proto: platform_pb2.ChipsetInfo) -> ChipsetAttributes:
        return cls(
            name=proto.name,
            aliases=list(proto.aliases),
            marketing_name=proto.marketing_name,
            world=website_world_from_proto(proto.world),
            supports_fp16=proto.supports_fp16,
            htp_version=proto.htp_version,
            soc_model=proto.soc_model,
            reference_device=proto.reference_device,
            supports_weight_sharing=proto.supports_weight_sharing,
        )

    @staticmethod
    def from_hub_device(hub_device: hub.Device | None) -> ChipsetAttributes | None:
        if not hub_device:
            return None
        device = RegisteredDevice.get(hub_device.name, return_unregistered=True)
        world = WebsiteWorld.from_form_factor(device.form_factor)
        return ChipsetAttributes(
            name=device.chipset,
            aliases=device.chipset_aliases,
            marketing_name=device.chipset,  # raw slug, no marketing formatting
            world=world,
            supports_fp16=device.supports_fp16_npu,
            htp_version=device.hexagon_version,
            soc_model=device.soc_model,
            reference_device=device.device_name,
            supports_weight_sharing=device.supports_weight_sharing,
        )


class TokenizerMetadata(BaseQAIHMConfig):
    """
    Structured description of the tokenizer required to run a model.

    This is embedded in ModelMetadata so that applications can discover
    tokenizer parameters directly from metadata.json without consulting
    external documentation.

    Attributes
    ----------
    vocab_file
        File name of the vocabulary file, relative to the export directory.
    type
        Tokenizer algorithm, e.g. "wordpiece" or "bpe".
    max_length
        Maximum sequence length the model accepts.
    do_lower_case
        Whether the tokenizer lowercases input text before tokenizing.
    """

    vocab_file: str
    type: str
    max_length: int
    do_lower_case: bool

    def to_proto(self) -> model_metadata_pb2.TokenizerMetadata:
        return model_metadata_pb2.TokenizerMetadata(
            vocab_file=self.vocab_file,
            type=self.type,
            max_length=self.max_length,
            do_lower_case=self.do_lower_case,
        )

    @classmethod
    def from_proto(
        cls, proto: model_metadata_pb2.TokenizerMetadata
    ) -> TokenizerMetadata:
        return cls(
            vocab_file=proto.vocab_file,
            type=proto.type,
            max_length=proto.max_length,
            do_lower_case=proto.do_lower_case,
        )


class ModelMetadata(BaseQAIHMConfig):
    """
    Metadata for a model that may have multiple model files.

    For single-file models (e.g., ResNet50), this will have one entry.
    For multi-file models (e.g., Stable Diffusion with text_encoder.bin, unet.bin, vae_decoder.bin),
    this will have multiple entries mapping model file names to their metadata.

    The keys in model_files are the actual file names (relative to the root of the export directory),
    making it clear what each metadata entry corresponds to in the exported files.

    This is the class that gets saved to disk as metadata.json.

    Attributes
    ----------
    model_id
        The model identifier (folder name), e.g. "mobilenet_v2".
    model_name
        The human-readable model name from info.yaml, e.g. "MobileNet-v2".
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
    genie
        Optional Genie SDK metadata for on-device deployment (chat template, pipeline topology, etc.).
    chipset_attributes
        Hardware attributes relevant to the compiled model asset.
        Only populated for AoT-compiled models; None for JIT runtimes.
    tokenizer
        Optional structured tokenizer parameters required to preprocess inputs.
        When present, applications can use this to configure the tokenizer without
        consulting external documentation.
    use_case
        The model's use case (e.g. text generation, object detection). Sourced
        from info.yaml.
    """

    model_id: str = ""
    model_name: str = ""
    runtime: TargetRuntime
    precision: Precision
    tool_versions: ToolVersions
    model_files: dict[str, ModelFileMetadata]
    supplementary_files: dict[str, str] = {}
    genie: GenieMetadata | None = None
    chipset_attributes: ChipsetAttributes | None = None
    tokenizer: TokenizerMetadata | None = None
    use_case: MODEL_USE_CASE | None = None

    def to_json(self, path: str | os.PathLike, **kwargs: Any) -> None:  # type: ignore[override]
        """Write this metadata to ``path`` as the JSON form of a ``ModelMetadata`` proto."""
        write_proto_json(self.to_proto(pkg_version.__version__), path)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> ModelMetadata:
        """Read a metadata.json written by ``to_json``."""
        return cls.from_proto(read_proto_json(path, model_metadata_pb2.ModelMetadata))

    def to_yaml(self, path: str | os.PathLike, **kwargs: Any) -> bool:  # type: ignore[override]
        """Write this metadata to ``path`` as the YAML form of a ``ModelMetadata`` proto."""
        write_proto_yaml(self.to_proto(pkg_version.__version__), path)
        return True

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike, create_empty_if_no_file: bool = False
    ) -> ModelMetadata:
        """Read a metadata YAML written by ``to_yaml``."""
        return cls.from_proto(read_proto_yaml(path, model_metadata_pb2.ModelMetadata))

    @classmethod
    def from_proto(cls, proto: model_metadata_pb2.ModelMetadata) -> ModelMetadata:
        return cls(
            model_id=proto.model_id,
            model_name=proto.model_name,
            runtime=runtime_from_proto(proto.runtime),
            precision=precision_from_proto(proto.precision),
            tool_versions=ToolVersions.from_proto(proto.tool_versions),
            model_files={
                f.filename: ModelFileMetadata.from_proto(f) for f in proto.model_files
            },
            supplementary_files={
                f.filename: f.description for f in proto.supplementary_files
            },
            genie=(
                GenieMetadata.from_proto(proto.genie)
                if proto.HasField("genie")
                else None
            ),
            chipset_attributes=(
                ChipsetAttributes.from_proto(proto.chipset_attributes)
                if proto.HasField("chipset_attributes")
                else None
            ),
            tokenizer=(
                TokenizerMetadata.from_proto(proto.tokenizer)
                if proto.HasField("tokenizer")
                else None
            ),
            use_case=use_case_from_proto(proto.use_case) if proto.use_case else None,
        )

    def to_proto(self, aihm_version: str) -> model_metadata_pb2.ModelMetadata:
        model_files = [
            file_meta.to_proto(filename)
            for filename, file_meta in self.model_files.items()
        ]
        supplementary_files = [
            model_metadata_pb2.SupplementaryFileMetadata(
                filename=filename, description=description
            )
            for filename, description in self.supplementary_files.items()
        ]
        return model_metadata_pb2.ModelMetadata(
            aihm_version=aihm_version,
            model_id=self.model_id,
            model_name=self.model_name,
            runtime=runtime_to_proto(self.runtime),
            precision=precision_to_proto(self.precision),
            tool_versions=self.tool_versions.to_proto(),
            model_files=model_files,
            supplementary_files=supplementary_files,
            genie=self.genie.to_proto() if self.genie else None,
            chipset_attributes=self.chipset_attributes.to_proto()
            if self.chipset_attributes
            else None,
            tokenizer=self.tokenizer.to_proto() if self.tokenizer else None,
            use_case=use_case_to_proto(self.use_case) if self.use_case else 0,
        )


# TODO (https://github.com/qcom-ai-hub/tetracode/issues/21114)
# We only need this because we can't reliably probe the name change
# from Workbench. If this is added, we can remove this.
def _runtime_sanitized(name: str) -> str:
    """Replicate common on-device name sanitization (context binaries)"""
    return name.replace("/", "_").replace(".", "_")


def _lookup_spec(
    name: str, available: dict[str, TensorSpec]
) -> tuple[str, TensorSpec] | None:
    """Resolve name against compiled metadata, allowing for sanitization."""
    if name in available:
        return name, available[name]
    sanitized = _runtime_sanitized(name)
    if sanitized in available:
        return sanitized, available[sanitized]
    return None


def merge_input_metadata(
    model_file_metadata: ModelFileMetadata,
    input_spec: InputSpec,
    target_runtime: TargetRuntime,
) -> None:
    """
    Merge semantic metadata from get_input_spec() into ModelFileMetadata.

    This function enriches the TensorSpec entries in model_file_metadata.inputs
    with additional metadata (io_type, image_metadata, value_range, description)
    from the model's get_input_spec() return value.

    TensorSpec entries in input_spec may have metadata fields set (io_type,
    image_metadata, etc.). This function copies those fields to the corresponding
    TensorSpec in model_file_metadata.inputs.

    Parameters
    ----------
    model_file_metadata
        The ModelFileMetadata to enrich (modified in place).
    input_spec
        The InputSpec from model.get_input_spec(). TensorSpec entries with
        metadata fields will have their metadata merged.
    target_runtime
        The runtime for which the model was compiled.

    Raises
    ------
    ValueError
        If an input in input_spec is not found in model_file_metadata.inputs.
        This indicates a bug - the input names should match.
    """
    # Link job models may have empty I/O specs (workbench bug).
    # Skip metadata merge entirely — compiled shapes can differ from
    # get_input_spec() (e.g. NCHW -> NHWC conversion during compile).
    if not model_file_metadata.inputs:
        return

    for input_name, spec in input_spec.items():
        found = _lookup_spec(input_name, model_file_metadata.inputs)
        if found is None:
            raise ValueError(
                f"Input '{input_name}' from get_input_spec() not found in compiled "
                f"model metadata. Available inputs: {list(model_file_metadata.inputs.keys())}"
            )
        _, target_spec = found

        # Skip if not a TensorSpec (plain tuple has no metadata to merge)
        if not isinstance(spec, TensorSpec):
            continue

        # Copy metadata fields if present
        if spec.io_type is not None:
            target_spec.io_type = spec.io_type
        if spec.image_metadata is not None:
            target_spec.image_metadata = spec.image_metadata
        if spec.description is not None:
            target_spec.description = spec.description
        if spec.value_range != (float("-inf"), float("inf")):
            target_spec.value_range = spec.value_range


def merge_output_metadata(
    model_file_metadata: ModelFileMetadata,
    output_spec: OutputSpec,
    target_runtime: TargetRuntime,
) -> None:
    """
    Merge semantic metadata from get_output_spec() into ModelFileMetadata.

    This function enriches the TensorSpec entries in model_file_metadata.outputs
    with additional metadata (io_type, bbox_metadata, description) from the
    model's get_output_spec() return value.

    Parameters
    ----------
    model_file_metadata
        The ModelFileMetadata to enrich (modified in place).
    output_spec
        The OutputSpec from model.get_output_spec(). TensorSpec entries with
        metadata fields will have their metadata merged.
    target_runtime
        The runtime for which the model was compiled.

    Raises
    ------
    ValueError
        If an output in output_spec is not found in model_file_metadata.outputs.
    """
    if not model_file_metadata.outputs:
        return

    for output_name, spec in output_spec.items():
        found = _lookup_spec(output_name, model_file_metadata.outputs)
        if found is None:
            raise ValueError(
                f"Output '{output_name}' from get_output_spec() not found in compiled "
                f"model metadata. Available outputs: {list(model_file_metadata.outputs.keys())}"
            )
        _, target_spec = found

        if spec.io_type is not None:
            target_spec.io_type = spec.io_type
        if spec.bbox_metadata is not None:
            target_spec.bbox_metadata = spec.bbox_metadata
        if spec.description is not None:
            target_spec.description = spec.description
        if spec.value_range != (float("-inf"), float("inf")):
            target_spec.value_range = spec.value_range
        if spec.softmax_applied:
            target_spec.softmax_applied = spec.softmax_applied
        if spec.labels_file is not None:
            target_spec.labels_file = spec.labels_file
        if spec.image_metadata is not None:
            target_spec.image_metadata = spec.image_metadata
