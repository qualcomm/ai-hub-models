# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import tempfile
from pathlib import Path

from qai_hub_models.configs.metadata_yaml import (
    ModelFileMetadata,
    ModelMetadata,
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.utils.asset_loaders import load_yaml

# Default values for required ModelMetadata fields in tests
DEFAULT_RUNTIME = TargetRuntime.TFLITE
DEFAULT_PRECISION = Precision.float
DEFAULT_TOOL_VERSIONS = ToolVersions(ai_hub_models="0.0.0")


def test_tensor_spec_creation() -> None:
    """Test creating a TensorSpec with and without quantization."""
    # Without quantization
    tensor_spec = TensorSpec(
        shape=[1, 3, 224, 224],
        dtype="float32",
    )
    assert tensor_spec.shape == [1, 3, 224, 224]
    assert tensor_spec.dtype == "float32"
    assert tensor_spec.quantization_parameters is None

    # With quantization
    quant_params = QuantizationParameters(scale=0.003921, zero_point=128)
    tensor_spec_quant = TensorSpec(
        shape=[1, 1000],
        dtype="float32",
        quantization_parameters=quant_params,
    )
    assert tensor_spec_quant.quantization_parameters is not None
    assert tensor_spec_quant.quantization_parameters.scale == 0.003921
    assert tensor_spec_quant.quantization_parameters.zero_point == 128


def test_model_file_metadata_creation() -> None:
    """Test creating ModelFileMetadata with inputs and outputs."""
    inputs = {
        "image": TensorSpec(
            shape=[1, 3, 224, 224],
            dtype="float32",
        )
    }
    outputs = {
        "logits": TensorSpec(
            shape=[1, 1000],
            dtype="float32",
        )
    }

    metadata = ModelFileMetadata(inputs=inputs, outputs=outputs)
    assert len(metadata.inputs) == 1
    assert len(metadata.outputs) == 1
    assert "image" in metadata.inputs
    assert "logits" in metadata.outputs


def test_model_metadata_single_component() -> None:
    """Test ModelMetadata with a single component."""
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=[1, 3, 224, 224], dtype="float32")},
        outputs={"logits": TensorSpec(shape=[1, 1000], dtype="float32")},
    )

    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"ResNet50": file_metadata},
    )
    assert len(model_metadata.model_files) == 1
    assert "ResNet50" in model_metadata.model_files


def test_model_metadata_multiple_components() -> None:
    """Test ModelMetadata with multiple components (like Stable Diffusion)."""
    text_encoder = ModelFileMetadata(
        inputs={"input_ids": TensorSpec(shape=[1, 77], dtype="int32")},
        outputs={"last_hidden_state": TensorSpec(shape=[1, 77, 768], dtype="float32")},
    )

    unet = ModelFileMetadata(
        inputs={"sample": TensorSpec(shape=[1, 4, 64, 64], dtype="float32")},
        outputs={"out_sample": TensorSpec(shape=[1, 4, 64, 64], dtype="float32")},
    )

    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={
            "text_encoder": text_encoder,
            "unet": unet,
        },
    )

    assert len(model_metadata.model_files) == 2
    assert "text_encoder" in model_metadata.model_files
    assert "unet" in model_metadata.model_files


def test_metadata_yaml_roundtrip() -> None:
    """Test saving and loading metadata YAML with flow-style lists."""
    # Create metadata
    file_metadata = ModelFileMetadata(
        inputs={"image": TensorSpec(shape=[1, 3, 224, 224], dtype="float32")},
        outputs={"logits": TensorSpec(shape=[1, 1000], dtype="float32")},
    )
    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"ResNet50": file_metadata},
    )

    # Save to YAML
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "metadata.yaml"
        model_metadata.to_yaml(yaml_path)

        # Verify file exists
        assert yaml_path.exists()

        # Load back and verify structure
        loaded_dict = load_yaml(yaml_path)
        assert "model_files" in loaded_dict
        assert "ResNet50" in loaded_dict["model_files"]
        assert "inputs" in loaded_dict["model_files"]["ResNet50"]
        assert "outputs" in loaded_dict["model_files"]["ResNet50"]

        # Verify flow-style lists (should be on one line)
        content = yaml_path.read_text()
        assert "[1, 3, 224, 224]" in content  # Flow style
        assert "[1, 1000]" in content  # Flow style


def test_metadata_with_quantization_roundtrip() -> None:
    """Test metadata with quantization parameters."""
    quant_params = QuantizationParameters(scale=0.003921568859368563, zero_point=128)
    file_metadata = ModelFileMetadata(
        inputs={
            "image": TensorSpec(
                shape=[1, 3, 224, 224],
                dtype="float32",
                quantization_parameters=quant_params,
            )
        },
        outputs={
            "logits": TensorSpec(
                shape=[1, 1000],
                dtype="float32",
                quantization_parameters=QuantizationParameters(
                    scale=0.00390625, zero_point=0
                ),
            )
        },
    )
    model_metadata = ModelMetadata(
        runtime=DEFAULT_RUNTIME,
        precision=DEFAULT_PRECISION,
        tool_versions=DEFAULT_TOOL_VERSIONS,
        model_files={"ResNet50": file_metadata},
    )

    # Save and load
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "metadata.yaml"
        model_metadata.to_yaml(yaml_path)

        loaded_dict = load_yaml(yaml_path)

        # Verify quantization parameters are present
        input_spec = loaded_dict["model_files"]["ResNet50"]["inputs"]["image"]
        assert "quantization_parameters" in input_spec
        assert input_spec["quantization_parameters"]["scale"] == 0.003921568859368563
        assert input_spec["quantization_parameters"]["zero_point"] == 128

        output_spec = loaded_dict["model_files"]["ResNet50"]["outputs"]["logits"]
        assert "quantization_parameters" in output_spec
        assert output_spec["quantization_parameters"]["scale"] == 0.00390625
        assert output_spec["quantization_parameters"]["zero_point"] == 0
