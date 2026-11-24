# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Field coverage tests for proto <-> pydantic mappings.

For each pydantic class with a to_proto() method, verify that every pydantic
field has a corresponding proto field. Proto messages may have extra fields
(e.g. aihm_version passed as an argument) — those are not checked here.

Known mismatches are explicitly listed per class so they fail loudly when
the proto schema is updated.
"""

from __future__ import annotations

from google.protobuf.descriptor import Descriptor
from qai_hub_models_cli.proto import (
    model_manifest_pb2,
    model_metadata_pb2,
    model_numerics_pb2,
    model_release_assets_pb2,
    platform_pb2,
)
from qai_hub_models_cli.proto.shared import tensor_spec_pb2, tool_versions_pb2

from qai_hub_models.configs._info_yaml_llm_details import LLMDetails
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.model_metadata import (
    ChipsetAttributes,
    GenieChatTemplate,
    GenieMetadata,
    GeniePipeline,
    GeniePipelineConnection,
    GenieSampleInput,
    GenieVisionPreprocessing,
    ModelFileMetadata,
    ModelMetadata,
)
from qai_hub_models.configs.numerics_benchmark import NumericsAccuracyBenchmark
from qai_hub_models.configs.tensor_spec import (
    ImageMetadata,
    QuantizationParameters,
    TensorSpec,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.utils.base_config import BaseQAIHMConfig


def _check_coverage(
    pydantic_cls: type[BaseQAIHMConfig],
    proto_descriptor: Descriptor,
    pydantic_field_renames: dict[str, str] | None = None,
    pydantic_only: set[str] | None = None,
    proto_only: set[str] | None = None,
) -> None:
    pydantic_field_renames = pydantic_field_renames or {}
    pydantic_only = pydantic_only or set()
    proto_only = proto_only or set()

    py_fields = set(pydantic_cls.model_fields.keys())
    proto_fields = {f.name for f in proto_descriptor.fields}

    mapped_py = {pydantic_field_renames.get(f, f) for f in py_fields - pydantic_only}
    unmapped_py = mapped_py - proto_fields
    assert not unmapped_py, (
        f"{pydantic_cls.__name__}: pydantic fields missing from proto: {unmapped_py}"
    )

    unmapped_proto = proto_fields - mapped_py - proto_only
    assert not unmapped_proto, (
        f"{pydantic_cls.__name__}: proto fields missing from pydantic: {unmapped_proto}"
    )


# ---------------------------------------------------------------------------
# model_metadata.proto
# ---------------------------------------------------------------------------


class TestModelMetadataFieldCoverage:
    def test_model_metadata(self) -> None:
        _check_coverage(
            ModelMetadata,
            model_metadata_pb2.ModelMetadata.DESCRIPTOR,
            proto_only={
                "aihm_version",
            },
        )

    def test_chipset_attributes(self) -> None:
        _check_coverage(
            ChipsetAttributes,
            platform_pb2.ChipsetInfo.DESCRIPTOR,
        )

    def test_model_file_metadata(self) -> None:
        _check_coverage(
            ModelFileMetadata,
            model_metadata_pb2.ModelFileMetadata.DESCRIPTOR,
            proto_only={"filename"},
        )

    def test_genie_metadata(self) -> None:
        _check_coverage(
            GenieMetadata,
            model_metadata_pb2.GenieMetadata.DESCRIPTOR,
        )

    def test_genie_chat_template(self) -> None:
        _check_coverage(
            GenieChatTemplate,
            model_metadata_pb2.GenieChatTemplate.DESCRIPTOR,
        )

    def test_genie_pipeline(self) -> None:
        _check_coverage(
            GeniePipeline,
            model_metadata_pb2.GenieMetadata.GeniePipeline.DESCRIPTOR,
        )

    def test_genie_pipeline_connection(self) -> None:
        _check_coverage(
            GeniePipelineConnection,
            model_metadata_pb2.GenieMetadata.GeniePipeline.GeniePipelineConnection.DESCRIPTOR,
        )

    def test_genie_sample_input(self) -> None:
        _check_coverage(
            GenieSampleInput,
            model_metadata_pb2.GenieMetadata.GenieSampleInput.DESCRIPTOR,
        )

    def test_genie_vision_preprocessing(self) -> None:
        _check_coverage(
            GenieVisionPreprocessing,
            model_metadata_pb2.GenieMetadata.GenieVisionPreprocessing.DESCRIPTOR,
        )


# ---------------------------------------------------------------------------
# shared protos
# ---------------------------------------------------------------------------


class TestSharedFieldCoverage:
    def test_tool_versions(self) -> None:
        _check_coverage(
            ToolVersions,
            tool_versions_pb2.ToolVersions.DESCRIPTOR,
        )

    def test_tensor_spec(self) -> None:
        _check_coverage(
            TensorSpec,
            tensor_spec_pb2.TensorSpec.DESCRIPTOR,
            proto_only={"name"},
            pydantic_only={"apply_runtime_channel_reordering"},
        )

    def test_quantization_parameters(self) -> None:
        _check_coverage(
            QuantizationParameters,
            tensor_spec_pb2.QuantizationParameters.DESCRIPTOR,
        )

    def test_image_metadata(self) -> None:
        _check_coverage(
            ImageMetadata,
            tensor_spec_pb2.ImageMetadata.DESCRIPTOR,
        )


# ---------------------------------------------------------------------------
# model_manifest.proto
# ---------------------------------------------------------------------------


# Still declared in the proto, but no longer carried by any YAML schema.
_UNUSED_INFO_PROTO_FIELDS = {"dataset", "form_factors"}


class TestInfoFieldCoverage:
    def test_model_info_manifest_half(self) -> None:
        _check_coverage(
            QAIHMModelManifest,
            model_manifest_pb2.ModelManifest.DESCRIPTOR,
            pydantic_field_renames={"license": "license_url"},
            proto_only={"aihm_version", "_internal"} | _UNUSED_INFO_PROTO_FIELDS,
        )

    def test_model_info_internal_half(self) -> None:
        _check_coverage(
            QAIHMModelInternalConfig,
            model_manifest_pb2.ModelManifest.Internal.DESCRIPTOR,
        )

    def test_llm_details(self) -> None:
        _check_coverage(
            LLMDetails,
            model_manifest_pb2.ModelManifest.Internal.LLMDetails.DESCRIPTOR,
        )

    def test_numerics_accuracy_benchmark(self) -> None:
        _check_coverage(
            NumericsAccuracyBenchmark,
            model_numerics_pb2.NumericsAccuracyBenchmark.DESCRIPTOR,
        )


# ---------------------------------------------------------------------------
# model_numerics.proto
# ---------------------------------------------------------------------------


class TestNumericsFieldCoverage:
    def test_numerics_metric(self) -> None:
        _check_coverage(
            QAIHMModelNumerics.MetricDetails,
            model_numerics_pb2.ModelNumerics.NumericsMetric.DESCRIPTOR,
            pydantic_field_renames={"device_metric": "device_metrics"},
        )

    def test_device_numerics_metrics(self) -> None:
        _check_coverage(
            QAIHMModelNumerics.DeviceDetails,
            model_numerics_pb2.ModelNumerics.NumericsMetric.DeviceNumericsMetrics.DESCRIPTOR,
            proto_only={"device", "precision", "runtime", "tool_versions"},
        )


# ---------------------------------------------------------------------------
# model_release_assets.proto
# ---------------------------------------------------------------------------


class TestReleaseAssetsFieldCoverage:
    def test_asset_details(self) -> None:
        _check_coverage(
            QAIHMModelReleaseAssets.AssetDetails,
            model_release_assets_pb2.ModelReleaseAssets.AssetDetails.DESCRIPTOR,
            proto_only={"precision", "runtime", "chipset"},
        )
