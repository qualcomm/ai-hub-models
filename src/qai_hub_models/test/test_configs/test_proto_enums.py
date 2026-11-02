# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Exhaustiveness tests for proto enum mappings.

Each test verifies two directions:
  1. Every Python enum member has a proto mapping entry.
  2. Every proto enum value (except UNSPECIFIED) has a Python mapping entry.
"""

from __future__ import annotations

from google.protobuf.descriptor import EnumDescriptor
from qai_hub_models_cli.proto import info_pb2, platform_pb2
from qai_hub_models_cli.proto.shared import (
    precision_pb2,
    runtime_pb2,
    tensor_spec_pb2,
)

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import (
    MODEL_DOMAIN,
    MODEL_LICENSE,
    MODEL_STATUS,
    MODEL_TAG,
    MODEL_USE_CASE,
    VOICE_AI_SDK,
)
from qai_hub_models.configs._info_yaml_llm_details import LLM_CALL_TO_ACTION
from qai_hub_models.configs.proto_helpers import (
    _CALL_TO_ACTION_TO_PROTO,
    _DOMAIN_TO_PROTO,
    _FORM_FACTOR_TO_PROTO,
    _LICENSE_TO_PROTO,
    _PRECISION_TO_PROTO,
    _RUNTIME_TO_PROTO,
    _STATUS_TO_PROTO,
    _TAG_TO_PROTO,
    _USE_CASE_TO_PROTO,
    _VOICE_AI_SDK_TO_PROTO,
)
from qai_hub_models.configs.tensor_spec import (
    _BBOX_FORMAT_TO_PROTO,
    _COLOR_FORMAT_TO_PROTO,
    _IO_TYPE_TO_PROTO,
    BboxFormat,
    ColorFormat,
    IoType,
)
from qai_hub_models.utils.device import FormFactor

_ALL_PRECISIONS = [
    Precision.float,
    Precision.w8a8,
    Precision.w8a16,
    Precision.w16a16,
    Precision.w4a16,
    Precision.w4,
    Precision.w8a8_mixed_int16,
    Precision.w8a16_mixed_int16,
    Precision.w8a8_mixed_fp16,
    Precision.w8a16_mixed_fp16,
    Precision.mxfp4,
    Precision.q8_0,
    Precision.q4_0,
    Precision.mixed,
    Precision.mixed_with_float,
]


def _proto_enum_values(descriptor: EnumDescriptor) -> set[int]:
    return {v.number for v in descriptor.values if not v.name.endswith("UNSPECIFIED")}


class TestPrecisionMapping:
    def test_python_to_proto(self) -> None:
        for p in _ALL_PRECISIONS:
            assert str(p) in _PRECISION_TO_PROTO, (
                f"Precision '{p}' has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            precision_pb2.DESCRIPTOR.enum_types_by_name["Precision"]
        )
        mapped = set(_PRECISION_TO_PROTO.values())
        assert proto_values == mapped


class TestRuntimeMapping:
    def test_python_to_proto(self) -> None:
        for rt in TargetRuntime:
            assert rt.value in _RUNTIME_TO_PROTO, (
                f"TargetRuntime.{rt.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            runtime_pb2.DESCRIPTOR.enum_types_by_name["Runtime"]
        )
        mapped = set(_RUNTIME_TO_PROTO.values())
        assert proto_values == mapped


class TestFormFactorMapping:
    def test_python_to_proto(self) -> None:
        for ff in FormFactor:
            assert ff.value in _FORM_FACTOR_TO_PROTO, (
                f"FormFactor.{ff.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            platform_pb2.DESCRIPTOR.enum_types_by_name["FormFactor"]
        )
        mapped = set(_FORM_FACTOR_TO_PROTO.values())
        assert proto_values == mapped


class TestVoiceAISDKMapping:
    def test_python_to_proto(self) -> None:
        for lic in VOICE_AI_SDK:
            assert lic.value in _VOICE_AI_SDK_TO_PROTO, (
                f"VOICE_AI_SDK.{lic.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            info_pb2.DESCRIPTOR.enum_types_by_name["VoiceAISdk"]
        )
        mapped = set(_VOICE_AI_SDK_TO_PROTO.values())
        assert proto_values == mapped


class TestLicenseMapping:
    def test_python_to_proto(self) -> None:
        for lic in MODEL_LICENSE:
            assert lic.value in _LICENSE_TO_PROTO, (
                f"MODEL_LICENSE.{lic.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            info_pb2.DESCRIPTOR.enum_types_by_name["ModelLicense"]
        )
        mapped = set(_LICENSE_TO_PROTO.values())
        assert proto_values == mapped


class TestDomainMapping:
    def test_python_to_proto(self) -> None:
        for d in MODEL_DOMAIN:
            assert d.value in _DOMAIN_TO_PROTO, (
                f"MODEL_DOMAIN.{d.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            info_pb2.DESCRIPTOR.enum_types_by_name["ModelDomain"]
        )
        mapped = set(_DOMAIN_TO_PROTO.values())
        assert proto_values == mapped


class TestTagMapping:
    def test_python_to_proto(self) -> None:
        for t in MODEL_TAG:
            assert t.value in _TAG_TO_PROTO, f"MODEL_TAG.{t.name} has no proto mapping"

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            info_pb2.DESCRIPTOR.enum_types_by_name["ModelTag"]
        )
        mapped = set(_TAG_TO_PROTO.values())
        assert proto_values == mapped


class TestStatusMapping:
    def test_python_to_proto(self) -> None:
        for s in MODEL_STATUS:
            assert s.value in _STATUS_TO_PROTO, (
                f"MODEL_STATUS.{s.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            info_pb2.DESCRIPTOR.enum_types_by_name["ModelStatus"]
        )
        mapped = set(_STATUS_TO_PROTO.values())
        assert proto_values == mapped


class TestUseCaseMapping:
    def test_python_to_proto(self) -> None:
        for uc in MODEL_USE_CASE:
            assert uc.value in _USE_CASE_TO_PROTO, (
                f"MODEL_USE_CASE.{uc.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            info_pb2.DESCRIPTOR.enum_types_by_name["ModelUseCase"]
        )
        mapped = set(_USE_CASE_TO_PROTO.values())
        assert proto_values == mapped


class TestCallToActionMapping:
    def test_python_to_proto(self) -> None:
        for cta in LLM_CALL_TO_ACTION:
            assert cta.value in _CALL_TO_ACTION_TO_PROTO, (
                f"LLM_CALL_TO_ACTION.{cta.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        msg_descriptor = info_pb2.DESCRIPTOR.message_types_by_name["ModelInfo"]
        llm_descriptor = msg_descriptor.nested_types_by_name["LLMDetails"]
        proto_values = _proto_enum_values(
            llm_descriptor.enum_types_by_name["CallToAction"]
        )
        mapped = set(_CALL_TO_ACTION_TO_PROTO.values())
        assert proto_values == mapped


class TestIoTypeMapping:
    def test_python_to_proto(self) -> None:
        for io in IoType:
            assert io.value in _IO_TYPE_TO_PROTO, (
                f"IoType.{io.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            tensor_spec_pb2.DESCRIPTOR.enum_types_by_name["IoType"]
        )
        mapped = set(_IO_TYPE_TO_PROTO.values())
        assert proto_values == mapped


class TestColorFormatMapping:
    def test_python_to_proto(self) -> None:
        for cf in ColorFormat:
            assert cf.value in _COLOR_FORMAT_TO_PROTO, (
                f"ColorFormat.{cf.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            tensor_spec_pb2.DESCRIPTOR.enum_types_by_name["ColorFormat"]
        )
        mapped = set(_COLOR_FORMAT_TO_PROTO.values())
        assert proto_values == mapped


class TestBboxFormatMapping:
    def test_python_to_proto(self) -> None:
        for bf in BboxFormat:
            assert bf.value in _BBOX_FORMAT_TO_PROTO, (
                f"BboxFormat.{bf.name} has no proto mapping"
            )

    def test_proto_to_python(self) -> None:
        proto_values = _proto_enum_values(
            tensor_spec_pb2.DESCRIPTOR.enum_types_by_name["BboxFormat"]
        )
        mapped = set(_BBOX_FORMAT_TO_PROTO.values())
        assert proto_values == mapped
