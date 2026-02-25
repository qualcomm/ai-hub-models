# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any

import numpy as np
import qai_hub as hub
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from qai_hub.client import QuantizeDtype, QuantizeJob
from qai_hub.hub import _global_client
from qai_hub.public_rest_api import get_framework_list
from typing_extensions import assert_never


class QAIRTVersion:
    # Map of <Workbench URL -> Valid AI Hub Workbench QAIRT Versions>
    _FRAMEWORKS: dict[str, list[QAIRTVersion.ParsedFramework]] = {}
    _HUB_DEFAULT_FRAMEWORK: dict[str, QAIRTVersion.ParsedFramework] = {}
    HUB_FLAG = "--qairt_version"
    DEFAULT_AIHUB_TAG = "default"
    LATEST_AIHUB_TAG = "latest"

    def __init__(
        self,
        version_or_tag: str | QAIRTVersion.ParsedFramework,
        return_default_if_does_not_exist: bool = False,
        validate_exists_on_ai_hub: bool = True,
    ) -> None:
        """
        Create this QAIRT Version.

        By default, the version or tag is validated against AI Hub Workbench's valid versions / tags, and an error is raised
        if the version or tag isn't valid.

        If AI Hub Workbench is not configured on this machine, returns a partially completed object. Some fields may be set to
        UNKNOWN because we can't get information from AI Hub Workbench.

        Parameters
        ----------
        version_or_tag
            QAIRT version or AI Hub Workbench version tag.

        return_default_if_does_not_exist
            If true, return the default AI Hub Workbench QAIRT version if this QAIRT version does not exist on AI Hub Workbench.

        validate_exists_on_ai_hub
            If this is True and AI Hub Workbench is reachable,
            raises an exception if this QAIRT version is not available on AI Hub Workbench.

            If this is False, and this version does not exist on AI Hub Workbench,
            acts as if AI Hub Workbench is not configured on this machine and returns a partially completed object.
            Some fields may be set to UNKNOWN because we can't get version information from AI Hub Workbench.
        """
        user_parsed_framework = (
            version_or_tag
            if isinstance(version_or_tag, QAIRTVersion.ParsedFramework)
            else QAIRTVersion.ParsedFramework.parse_opt(version_or_tag)
        )

        (
            self._api_url,
            valid_hub_frameworks,
            default_framework,
        ) = QAIRTVersion._load_frameworks()
        chosen_framework: QAIRTVersion.ParsedFramework | None = None
        if self._api_url and validate_exists_on_ai_hub:
            # Try to match the version_or_tag with a known QAIRT framework from AI Hub Workbench.
            for hub_framework in valid_hub_frameworks:
                if user_parsed_framework is None:
                    if version_or_tag in hub_framework.tags:
                        chosen_framework = hub_framework
                        break
                elif hub_framework.version_eq(user_parsed_framework):
                    chosen_framework = hub_framework
                    break

            # If no AI Hub Workbench QAIRT framework matches, then use a default.
            if (
                chosen_framework is None
                and return_default_if_does_not_exist
                and default_framework is not None
            ):
                chosen_framework = default_framework

            # Frameworks loaded from Hub never have the flavor, so add it manually here if the user provided one.
            if chosen_framework is not None and user_parsed_framework is not None:
                chosen_framework = chosen_framework.copy()
                chosen_framework.flavor = user_parsed_framework.flavor
        else:
            chosen_framework = user_parsed_framework or QAIRTVersion.ParsedFramework(
                major=0,
                minor=0,
                tags=(
                    [version_or_tag]
                    if isinstance(version_or_tag, str)
                    and version_or_tag in self.all_tags()
                    else []
                ),
            )

        if chosen_framework is None:
            # No valid framework available; raise.
            all_versions_str = "\n    ".join([str(x) for x in QAIRTVersion.all()])
            raise ValueError(
                f"QAIRT version {version_or_tag} is not supported by AI Hub Workbench. Available versions are:\n    {all_versions_str}"
            )

        self.framework = chosen_framework

    @property
    def api_version(self) -> str:
        return self.framework.api_version

    @property
    def full_version(self) -> str:
        return self.framework.full_version

    @property
    def full_version_with_flavor(self) -> str:
        return self.framework.full_version_with_flavor

    @property
    def sdk_flavor(self) -> str | None:
        return self.framework.flavor

    @property
    def tags(self) -> list[str]:
        return self.framework.tags

    @property
    def hub_option(self) -> str:
        """String flag to pass to hub to use this Hub version."""
        if self.is_default:
            return ""
        return (
            f"{QAIRTVersion.HUB_FLAG} {self.tags[0]}"
            if self.tags
            else f"{QAIRTVersion.HUB_FLAG} {self.api_version}"
        )

    @property
    def explicit_hub_option(self) -> str:
        """String flag to pass to hub to use this Hub version. This flag always uses the "API version" number, never the tag."""
        return f"{QAIRTVersion.HUB_FLAG} {self.api_version}"

    @property
    def is_default(self) -> bool:
        """Whether this is the default AI Hub Workbench QAIRT version."""
        return QAIRTVersion.DEFAULT_AIHUB_TAG in self.tags

    @staticmethod
    def all_tags() -> list[str]:
        """All potential (known) tag names."""
        return [
            QAIRTVersion.DEFAULT_AIHUB_TAG,
            QAIRTVersion.LATEST_AIHUB_TAG,
        ]

    def __eq__(self, other: str | QAIRTVersion) -> bool:
        other_framework = None
        if isinstance(other, str):
            if other in self.tags:
                return True
            other_framework = QAIRTVersion.ParsedFramework.parse_opt(other)
        if isinstance(other, QAIRTVersion):
            other_framework = other.framework
        return (
            other_framework is not None
            and self.framework.version_eq(other_framework)
            and self.framework.flavor == other_framework.flavor
        )

    def __str__(self) -> str:
        return (
            f"QAIRT v{self.api_version}"
            + (
                " | UNVERIFIED - NO AI HUB ACCESS"
                if not self._api_url
                else f" | {self.full_version_with_flavor}"
            )
            + (f" | {', '.join(self.tags)}" if self.tags else "")
        )

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def default() -> QAIRTVersion:
        """Default QAIRT version on AI Hub Workbench."""
        return QAIRTVersion(QAIRTVersion.DEFAULT_AIHUB_TAG)

    @staticmethod
    def latest() -> QAIRTVersion:
        """Latest QAIRT version on AI Hub Workbench."""
        return QAIRTVersion(QAIRTVersion.LATEST_AIHUB_TAG)

    @staticmethod
    def all() -> list[QAIRTVersion]:
        _, frameworks, _ = QAIRTVersion._load_frameworks()
        return [QAIRTVersion(f) for f in frameworks]

    @staticmethod
    def _load_frameworks() -> tuple[str, list[ParsedFramework], ParsedFramework | None]:
        """
        Load frameworks from AI Hub Workbench and populate cache.

        Returns
        -------
        api_url : str
            Currently active Hub api_url.
        valid_frameworks : list[ParsedFramework]
            All valid frameworks on this Hub version.
        default_framework : ParsedFramework | None
            Framework that corresponds with AI Hub Workbench default.
        """
        try:
            api_url = _global_client.config.api_url
            if api_url not in QAIRTVersion._FRAMEWORKS:
                QAIRTVersion._FRAMEWORKS[api_url] = [
                    QAIRTVersion.ParsedFramework.parse(f.full_version, list(f.api_tags))
                    for f in get_framework_list(_global_client.config).frameworks
                    if f.name == "QAIRT"
                ]
                for framework in QAIRTVersion._FRAMEWORKS[api_url]:
                    if QAIRTVersion.DEFAULT_AIHUB_TAG in framework.tags:
                        QAIRTVersion._HUB_DEFAULT_FRAMEWORK[api_url] = framework

        except Exception:
            # No hub API Access
            api_url = ""
            QAIRTVersion._FRAMEWORKS[api_url] = []

        return (
            api_url,
            QAIRTVersion._FRAMEWORKS[api_url],
            QAIRTVersion._HUB_DEFAULT_FRAMEWORK.get(api_url),
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Defines the parsing & serialization when QAIRTVersion is used in BaseQAIHMConfig objects."""
        return core_schema.with_info_after_validator_function(
            lambda obj, _: (
                cls(obj, validate_exists_on_ai_hub=False)
                if isinstance(obj, str)
                else obj
            ),
            handler(Any),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v.full_version_with_flavor, when_used="json"
            ),
        )

    @dataclass
    class ParsedFramework:
        """
        QAIRT version parsed into components, + any related tags.

        major / minor are required for this to be meaningful in all sigurations.
        patch / ident / flavor are optional as they can be inferred from AI Hub Workbench's list of valid QAIRT versions.
        """

        major: int
        minor: int
        patch: int | None = None
        ident: str | None = None
        flavor: str | None = None
        tags: list[str] = field(default_factory=list)

        @property
        def api_version(self) -> str:
            return f"{self.major}.{self.minor}"

        @property
        def full_version(self) -> str:
            return (
                self.api_version
                + (f".{self.patch}" if self.patch is not None else "")
                + (f".{self.ident}" if self.ident is not None else "")
            )

        @property
        def full_version_with_flavor(self) -> str:
            return self.full_version + (f"-{self.flavor}" if self.flavor else "")

        def version_eq(self, other: QAIRTVersion.ParsedFramework) -> bool:
            """Return true if this version matches the other version."""
            return (
                isinstance(other, QAIRTVersion.ParsedFramework)
                and self.major == other.major
                and self.minor == other.minor
                and (
                    self.patch is None
                    or other.patch is None
                    or self.patch == other.patch
                )
                and (
                    self.ident is None
                    or other.ident is None
                    or other.ident.startswith(self.ident)
                    or self.ident.startswith(other.ident)
                )
                and (
                    self.flavor is None
                    or other.flavor is None
                    or self.flavor == other.flavor
                )
            )

        def copy(self) -> QAIRTVersion.ParsedFramework:
            return QAIRTVersion.ParsedFramework(
                self.major, self.minor, self.patch, self.ident, self.flavor, self.tags
            )

        @staticmethod
        def parse(
            version: str, tags: list[str] | None = None
        ) -> QAIRTVersion.ParsedFramework:
            res = QAIRTVersion.ParsedFramework.parse_opt(version, tags)
            if not res:
                raise ValueError(f"Unable to parse QAIRT version string {version}.")
            return res

        @staticmethod
        def parse_opt(
            version: str, tags: list[str] | None = None
        ) -> QAIRTVersion.ParsedFramework | None:
            if m := re.search(
                r"v?(?P<major>\d+)\.(?P<minor>\d+)(?P<patch>\.\d+)?(?P<ident>\.\d+\_?\d+)?(?P<flavor>\-.*)?",
                version,
            ):
                g = m.groupdict()
                major = int(g["major"])
                minor = int(g["minor"])
                patch = int(r[1:]) if (r := g.get("patch")) else None
                ident = r[1:] if (r := g.get("ident")) else None
                flavor = r[1:] if (r := g.get("flavor")) else None
                return QAIRTVersion.ParsedFramework(
                    major, minor, patch, ident, flavor, tags or []
                )
            return None

    def __hash__(self) -> int:
        return hash(self.full_version_with_flavor)


@unique
class InferenceEngine(Enum):
    """The inference engine that executes a TargetRuntime asset."""

    TFLITE = "tflite"
    QNN = "qnn"
    ONNX = "onnx"
    GENIE = "genie"
    LLAMA_CPP_CPU = "llama_cpp_cpu"
    LLAMA_CPP_GPU = "llama_cpp_gpu"
    LLAMA_CPP_NPU = "llama_cpp_npu"

    @property
    def full_package_name(self) -> str:
        if self == InferenceEngine.TFLITE:
            return "TensorFlow Lite"
        if self == InferenceEngine.QNN:
            return "QAIRT (Qualcomm AI Runtime)"
        if self == InferenceEngine.ONNX:
            return "ONNX Runtime"
        if self == InferenceEngine.GENIE:
            return "Genie (Qualcomm GenAI Inference Extensions)"
        if self == InferenceEngine.LLAMA_CPP_CPU:
            return "Llama.cpp (CPU)"
        if self == InferenceEngine.LLAMA_CPP_GPU:
            return "Llama.cpp (GPU)"
        if self == InferenceEngine.LLAMA_CPP_NPU:
            return "Llama.cpp (NPU)"
        assert_never(self)

    @property
    def supported_version(self) -> str | None:
        """
        Supported version of this inference engine.

        If None, any version is supported.
        """
        if self == InferenceEngine.ONNX:
            return "1.22.2"
        return None

    @property
    def default_qairt_version(self: InferenceEngine) -> QAIRTVersion:
        """Default QAIRT version used by this inference engine."""
        qairt_version = "2.42" if self == InferenceEngine.ONNX else "2.43"

        try:
            return QAIRTVersion(qairt_version)
        except ValueError as e:
            msg = e.args[0]
            if "is not supported by AI Hub" in msg:
                runtime_version_str = (
                    f"This version of AI Hub Models was tested against {self.full_package_name} (v{self.supported_version}) which is bundled with QAIRT {qairt_version}.\n"
                    if self.supported_version is not None
                    else ""
                )
                raise ValueError(
                    msg
                    + f"\n\nAI Hub Workbench no longer supports the standard QAIRT version ({qairt_version}) used by this version of AI Hub Models.\n"
                    + runtime_version_str
                    + "You have two options:\n"
                    f" 1. Pass --fetch-static-assets to the export.py script, to fetch numbers / model files created for this release, that are compatible with QAIRT {qairt_version}.\n"
                    f" OR\n"
                    f" 2. Pass --compile-options='--qairt_version=default' and/or --profile-options='--qairt_version=default' to use the current default available on AI Hub Workbench. "
                    "DO THIS AT YOUR OWN RISK -- Older versions of AI Hub Models are not guaranteed to work with newer versions of QAIRT."
                ) from None
            raise


@unique
class ConversionToolchain(Enum):
    """The toolchain used to convert this asset from the source model format."""

    AIHUB_TFLITE_CONVERTER = "tflite"
    AIHUB_ONNX_CONVERTER = "onnx"
    QAIRT_CONVERTER = "qairt_converters"


@unique
class TargetRuntime(Enum):
    """Compilation target."""

    # TensorFlow Lite Runtime (renamed to LiteRT)
    # https://ai.google.dev/edge/litert
    TFLITE = "tflite"

    # QAIRT (Qualcomm AI Runtime) Device-Agnostic Graph IR
    # https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
    QNN_DLC = "qnn_dlc"

    # QAIRT (Qualcomm AI Runtime) Device-Specific, Precompiled Binary for HTP
    # https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk
    QNN_CONTEXT_BINARY = "qnn_context_binary"

    # ONNX Runtime
    # https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
    ONNX = "onnx"

    # QAIRT Context Binary Embedded within ONNX File
    # https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#qnn-context-binary-cache-feature)
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"

    # Qualcomm GenAI Inference Extensions Bundle
    # https://www.qualcomm.com/developer/software/gen-ai-inference-extensions
    GENIE = "genie"

    # ONNX Runtime GenAI Bundle
    # https://github.com/microsoft/onnxruntime-genai
    ONNXRUNTIME_GENAI = "onnxruntime_genai"

    # Llama.cpp runtime variants
    # https://github.com/ggml-org/llama.cpp
    LLAMA_CPP_CPU = "llama_cpp_cpu"
    LLAMA_CPP_GPU = "llama_cpp_gpu"
    LLAMA_CPP_NPU = "llama_cpp_npu"

    @staticmethod
    def from_hub_model_type(model_type: hub.SourceModelType) -> TargetRuntime:
        for rt in TargetRuntime:
            if rt.hub_model_type == model_type:
                return rt
        raise ValueError(f"Unsupported Hub model type: {model_type}")

    @property
    def inference_engine(self) -> InferenceEngine:
        if self == TargetRuntime.TFLITE:
            return InferenceEngine.TFLITE
        if (
            self == TargetRuntime.QNN_CONTEXT_BINARY  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == TargetRuntime.QNN_DLC
        ):
            return InferenceEngine.QNN
        if (
            self == TargetRuntime.ONNX  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == TargetRuntime.PRECOMPILED_QNN_ONNX
            or self == TargetRuntime.ONNXRUNTIME_GENAI
        ):
            return InferenceEngine.ONNX
        if self == TargetRuntime.GENIE:
            return InferenceEngine.GENIE
        if self == TargetRuntime.LLAMA_CPP_CPU:
            return InferenceEngine.LLAMA_CPP_CPU
        if self == TargetRuntime.LLAMA_CPP_GPU:
            return InferenceEngine.LLAMA_CPP_GPU
        if self == TargetRuntime.LLAMA_CPP_NPU:
            return InferenceEngine.LLAMA_CPP_NPU
        assert_never(self)

    @property
    def file_extension(self) -> str:
        """The file extension (without the .) assets for this runtime use."""
        if self == TargetRuntime.TFLITE:
            return "tflite"
        if self == TargetRuntime.QNN_CONTEXT_BINARY:
            return "bin"
        if self == TargetRuntime.QNN_DLC:
            return "dlc"
        if self == TargetRuntime.ONNX:
            return "onnx.zip"
        if self == TargetRuntime.PRECOMPILED_QNN_ONNX:
            return "onnx.zip"
        if self == TargetRuntime.GENIE:
            return "genie.zip"
        if self == TargetRuntime.ONNXRUNTIME_GENAI:
            return "onnxruntime_genai.zip"
        if (
            self == TargetRuntime.LLAMA_CPP_CPU  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == TargetRuntime.LLAMA_CPP_GPU
            or self == TargetRuntime.LLAMA_CPP_NPU
        ):
            return "gguf"

        assert_never(self)

    @property
    def hub_model_type(self) -> hub.SourceModelType:
        """The associated hub SourceModelType for assets for this TargetRuntime."""
        if self == TargetRuntime.QNN_CONTEXT_BINARY:
            return hub.SourceModelType.QNN_CONTEXT_BINARY
        if self == TargetRuntime.QNN_DLC:
            return hub.SourceModelType.QNN_DLC
        if self == TargetRuntime.PRECOMPILED_QNN_ONNX or self == TargetRuntime.ONNX:  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            return hub.SourceModelType.ONNX
        if self == TargetRuntime.TFLITE:
            return hub.SourceModelType.TFLITE
        if (
            self == TargetRuntime.GENIE  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == TargetRuntime.ONNXRUNTIME_GENAI
            or self == TargetRuntime.LLAMA_CPP_CPU
            or self == TargetRuntime.LLAMA_CPP_GPU
            or self == TargetRuntime.LLAMA_CPP_NPU
        ):
            raise ValueError(f"No Hub model type is applicable for {self.value}")
        assert_never(self)

    @property
    def channel_last_native_execution(self) -> bool:
        """
        If true, this runtime natively executes ops in NHWC (channel-last) format.
        If false, the runtime executes ops in NCHW (channel-first) format.
        """
        return self.is_aot_compiled or self.inference_engine in [
            InferenceEngine.QNN,
            InferenceEngine.TFLITE,
        ]

    @property
    def qairt_version_changes_compilation(self) -> bool:
        """Returns true if different versions of Qualcomm AI Runtime will affect how this runtime is compiled."""
        return self.is_aot_compiled or self.inference_engine == InferenceEngine.QNN

    def supports_precision(self, precision: Precision) -> bool:
        # Float is always supported.
        if precision == Precision.float:
            return True

        # All of our runtimes might support this. We don't have enough information to say "no", as we don't have access to each component precision.
        if precision == Precision.mixed_with_float:
            return True

        # Check quantized types
        if self == TargetRuntime.TFLITE:
            # Precision.mixed is not supported because it is guaranteed to have at least 1 layer that is not int8.
            # Since "mixed" cannot have floating point layers, it must have multiple quantized precisions. TF Lite only supports one.
            # Otherwise, the user would have picked a single precision to represent the whole component model.
            return precision == Precision.w8a8

        if self == TargetRuntime.ONNX:
            return precision in [
                Precision.w8a8,
                Precision.w8a16,
                # The following three are enabled tentatively
                # (not experimentally verified)
                Precision.w16a16,
                Precision.w4a16,
                Precision.w4,
                # Mixed-precision profile
                Precision.w8a8_mixed_int16,
                Precision.w8a16_mixed_int16,
                Precision.w8a8_mixed_fp16,
                Precision.w8a16_mixed_fp16,
                Precision.mixed,
            ]

        if (
            self == TargetRuntime.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == TargetRuntime.QNN_CONTEXT_BINARY
            # The following have QAIRT Context binary embedded within,
            # so they support the same precision set as QAIRT paths
            or self == TargetRuntime.PRECOMPILED_QNN_ONNX
            or self == TargetRuntime.GENIE
            or self == TargetRuntime.ONNXRUNTIME_GENAI
        ):
            return precision in [
                Precision.w8a8,
                Precision.w8a16,
                Precision.w4a16,
                Precision.w4,
                Precision.w16a16,
                # Mixed-precision profile
                Precision.w8a8_mixed_int16,
                Precision.w8a16_mixed_int16,
                Precision.w8a8_mixed_fp16,
                Precision.w8a16_mixed_fp16,
                Precision.mixed,
            ]
        if (
            self == TargetRuntime.LLAMA_CPP_CPU  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == TargetRuntime.LLAMA_CPP_GPU
            or self == TargetRuntime.LLAMA_CPP_NPU
        ):
            # Llama.cpp supports various GGUF quantization formats
            return precision in [
                Precision.mxfp4,
                Precision.q8_0,
                Precision.q4_0,
            ]

        assert_never(self)

    @property
    def aihub_target_runtime_flag(self) -> str:
        """AI Hub Workbench job flag for compiling to this runtime."""
        # LLAMA_CPP_* runtimes don't go through AI Hub compilation
        if self in [
            TargetRuntime.LLAMA_CPP_CPU,
            TargetRuntime.LLAMA_CPP_GPU,
            TargetRuntime.LLAMA_CPP_NPU,
        ]:
            raise ValueError(
                f"Runtime {self.name} does not support AI Hub compilation. "
                "llama.cpp runtimes use pre-built GGUF models instead."
            )

        if self.is_exclusively_for_genai:
            hub_target_runtime_flag = TargetRuntime.QNN_DLC.value
        else:
            hub_target_runtime_flag = self.value

        return f"--target_runtime {hub_target_runtime_flag}"

    @property
    def default_qairt_version(self: TargetRuntime) -> QAIRTVersion:
        """
        Default QAIRT version used by this runtime.

        THIS MIGHT BE DIFFERENT THAN AI HUB's DEFAULT VERSION.
        """
        if self == TargetRuntime.GENIE:
            return QAIRTVersion("2.42")
        return self.inference_engine.default_qairt_version

    @property
    def is_aot_compiled(self) -> bool:
        """
        Returns true if this asset is fully compiled ahead of time (before running on target).
        This means the compiled asset contains a QNN context binary.
        """
        return self in [
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
            TargetRuntime.GENIE,
            TargetRuntime.ONNXRUNTIME_GENAI,
        ]

    @property
    def is_exclusively_for_genai(self) -> bool:
        """Returns true if this runtime is exclusively used to execute GenAI models."""
        return self in [
            TargetRuntime.GENIE,
            TargetRuntime.ONNXRUNTIME_GENAI,
            TargetRuntime.LLAMA_CPP_CPU,
            TargetRuntime.LLAMA_CPP_GPU,
            TargetRuntime.LLAMA_CPP_NPU,
        ]


class _FloatDtype(Enum):
    """
    Temporary Enum to represent floating-point precision types not yet included in QuantizeDtype (Supported data types for quantize jobs).
    Currently includes FP16 for compatibility with precision handling logic.
    """

    FP16 = 1


# Used to avoid type name shadowing in the Precision class.
# With in the precision class scope, typings for function parameters might refer to "float".
# Mypy thinks that "float" is Precision.float in that context, rather than the primitive type.
floating = float


class Precision:
    """
    Stores a precision configuration for a model's graph.

    Precision can represent any precision that AI Hub Workbench's QuantizeJob supports.
    It also can represent floating point (no quantization).
    """

    # Common Precision Instances
    float: Precision
    w8a8: Precision
    w8a16: Precision
    w16a16: Precision
    w4a16: Precision
    w4: Precision

    # Mixed Precision Instances
    w8a8_mixed_int16: Precision
    w8a16_mixed_int16: Precision
    w8a8_mixed_fp16: Precision
    w8a16_mixed_fp16: Precision

    # GGUF-specific Precision (for llama.cpp)
    mxfp4: Precision
    q8_0: Precision
    q4_0: Precision

    # Component model with parts that each have a different precision.
    mixed: Precision  # No component has floating point layers.
    mixed_with_float: Precision  # At least one component has floating point layers.

    _allowed_override_dtypes: set[QuantizeDtype | _FloatDtype] = {
        QuantizeDtype.INT16,
        _FloatDtype.FP16,
    }

    def __init__(
        self,
        weights_type: QuantizeDtype | None,
        activations_type: QuantizeDtype | None,
        override_type: QuantizeDtype | _FloatDtype | None = None,
        _custom_name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        weights_type
            Quantization type for model weights. If None, weights are floating point.
        activations_type
            Quantization type for model activations. If None, activations are floating point.
        override_type
            Secondary type for mixed-precision quantization. If None, no mixed-precision is used.
            When provided, this overrides both `weights_type` and `activations_type` for the most sensitive layer(s),
            applying the same dtype to both weights and activations.
        _custom_name
            `_custom_name` is used for special precision formats (e.g., GGUF formats like mxfp4)
            that don't fit the standard wXaY naming convention.
        """
        if (
            override_type is not None
            and override_type not in self._allowed_override_dtypes
        ):
            raise ValueError(
                f"Invalid override_type: {override_type}. Supported: {','.join([str(x) for x in self._allowed_override_dtypes])}"
            )

        self.weights_type: QuantizeDtype | None = weights_type
        self.activations_type: QuantizeDtype | None = activations_type
        self.override_type: QuantizeDtype | _FloatDtype | None = override_type
        self._custom_name: str | None = _custom_name

        if self._custom_name is not None:
            assert (
                self.weights_type is None
                and self.activations_type is None
                and self.override_type is None
            ), "If _custom_name is set, all other Precision parameters must be None."

    @staticmethod
    def _parse_override_type(
        type_str: str | None,
    ) -> QuantizeDtype | _FloatDtype | None:
        """
        Parse an override type string to its enum value.

        Parameters
        ----------
        type_str
            The override type string (e.g., "int16", "fp16", "INT16", "FP16").

        Returns
        -------
        value: QuantizeDtype | _FloatDtype | None
            The corresponding enum value, or None if type_str is None.

        Raises
        ------
        ValueError
            If the type string is not a valid override type.
        """
        if type_str is None:
            return None

        enum_name = type_str.upper()

        # Check if it's an allowed override dtype
        is_allowed = any(
            enum_name == otype.name for otype in Precision._allowed_override_dtypes
        )
        if not is_allowed:
            raise ValueError(
                f"Invalid override type '{type_str}'. "
                f"Supported: {', '.join(otype.name.lower() for otype in Precision._allowed_override_dtypes)}"
            )

        # Resolve from QuantizeDtype or _FloatDtype
        for enum_type in (QuantizeDtype, _FloatDtype):
            if enum_name in enum_type._member_names_:
                return enum_type[enum_name]

        return None

    @staticmethod
    def parse(obj: Any) -> Precision:
        if isinstance(obj, Precision):
            return obj
        if not isinstance(obj, str):
            raise TypeError(f"Unknown type {obj} for parsing to Precision")
        string = obj
        if string == "float":
            return Precision.float
        if string == "mxfp4":
            return Precision.mxfp4
        if string == "q8_0":
            return Precision.q8_0
        if string == "q4_0":
            return Precision.q4_0
        if string == "mixed":
            return Precision.mixed
        if string == "mixed_with_float":
            return Precision.mixed_with_float

        atype = None
        wtype = None
        otype_str = None

        if match := re.match(r"(.*)_mixed_(.*)", string):
            otype_str = match.group(2)
        if match := re.match(r"(w\d+)?(a\d+)?", string):
            wtype = match.group(1)
            atype = match.group(2)
        if match := re.match(r"(a\d+)?(w\d+)?", string):
            atype = atype or match.group(1)
            wtype = wtype or match.group(2)

        if not atype and not wtype:
            raise ValueError(
                "Unsupported quantization type string. Example valid strings: float, w8a16, a8w8, w8, a16, ..."
            )

        atype_enum_name = f"INT{atype[1:]}" if atype else None
        wtype_enum_name = f"INT{wtype[1:]}" if wtype else None

        for enum_name, bit_width, name in [
            (atype_enum_name, atype, "activations"),
            (wtype_enum_name, wtype, "weights"),
        ]:
            if not bit_width:
                continue
            if enum_name not in QuantizeDtype._member_names_:
                raise ValueError(
                    f"Unsupported bit width {bit_width} for quantization {name}. Supported: {','.join([str(x) for x in QuantizeDtype])}"
                )

        return Precision(
            QuantizeDtype[wtype_enum_name] if wtype_enum_name else None,
            QuantizeDtype[atype_enum_name] if atype_enum_name else None,
            Precision._parse_override_type(otype_str),
        )

    @property
    def has_quantized_activations(self) -> bool:
        """True if ANY model activations are quantized (NOT floating point)."""
        if self._custom_name in ["mixed", "mixed_with_float"]:
            return True

        return (
            self.activations_type is not None
            and self.activations_type not in _FloatDtype
        ) or (self.override_type is not None and self.override_type not in _FloatDtype)

    @property
    def has_float_activations(self) -> bool:
        """True if ANY model activations are floating point (not quantized)."""
        if self._custom_name == "mixed":
            return False

        return (
            self.activations_type is None
            or self.activations_type in _FloatDtype
            or (self.override_type is not None and self.override_type in _FloatDtype)
        )

    def __str__(self) -> str:
        if self._custom_name:
            return self._custom_name
        if self == Precision.float:
            return "float"
        if self._custom_name is not None:
            return self._custom_name

        precision_name = (
            f"w{self.weights_type.name[3:]}" if self.weights_type else ""
        ) + (f"a{self.activations_type.name[3:]}" if self.activations_type else "")

        if self.override_type:
            precision_name = f"{precision_name}_mixed_{self.override_type.name.lower()}"

        return precision_name

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Precision):
            return False
        return (
            value.activations_type == self.activations_type
            and value.weights_type == self.weights_type
            and value.override_type == self.override_type
            and value._custom_name == self._custom_name
        )

    def __hash__(self) -> int:
        return hash(str(self))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            lambda obj, _: cls.parse(obj),
            handler(Any),
            serialization=core_schema.plain_serializer_function_ser_schema(
                Precision.__str__, when_used="json"
            ),
        )

    def get_hub_quantize_options(
        self, litemp_percentage: floating | None = None
    ) -> str:
        """
        Generates CLI flags used when submitting an AI Hub Workbench quantized job that targets this precision.

        Parameters
        ----------
        litemp_percentage
            The Lite-MP percentage value for mixed precision quantization.
            Required when this precision has an override_type (mixed precision).

        Returns
        -------
        options: str
            The quantize options string to pass to AI Hub.

        Raises
        ------
        ValueError
            If this is a mixed precision but litemp_percentage is not provided.
        """
        if self.override_type is not None:
            if litemp_percentage is None:
                raise ValueError(
                    "litemp_percentage is required for mixed precision quantization."
                )
            override_qtype = self.override_type.name.lower()
            return f"--range_scheme min_max --lite_mp percentage={litemp_percentage};override_qtype={override_qtype}"
        if self == Precision.w8a16:
            return "--range_scheme min_max"
        return ""

    @staticmethod
    def from_quantize_job(job: QuantizeJob) -> Precision:
        """
        Create a Precision from a QuantizeJob.

        Parameters
        ----------
        job
            The AI Hub quantize job to extract precision from.

        Returns
        -------
        Precision
            The precision that corresponds to the quantize job's settings.
        """
        override_type_str: str | None = None
        if match := re.search(r"override_qtype=(\w+)", job.options):
            override_type_str = match.group(1)

        return Precision(
            job.weights_dtype,
            job.activations_dtype,
            Precision._parse_override_type(override_type_str),
        )


Precision.float = Precision(None, None)
Precision.w8a8 = Precision(QuantizeDtype.INT8, QuantizeDtype.INT8)
Precision.w8a16 = Precision(QuantizeDtype.INT8, QuantizeDtype.INT16)
Precision.w16a16 = Precision(QuantizeDtype.INT16, QuantizeDtype.INT16)
Precision.w4a16 = Precision(QuantizeDtype.INT4, QuantizeDtype.INT16)
Precision.w4 = Precision(QuantizeDtype.INT4, None)
Precision.w8a8_mixed_int16 = Precision(
    QuantizeDtype.INT8, QuantizeDtype.INT8, QuantizeDtype.INT16
)
Precision.w8a16_mixed_int16 = Precision(
    QuantizeDtype.INT8, QuantizeDtype.INT16, QuantizeDtype.INT16
)
Precision.w8a8_mixed_fp16 = Precision(
    QuantizeDtype.INT8, QuantizeDtype.INT8, _FloatDtype.FP16
)
Precision.w8a16_mixed_fp16 = Precision(
    QuantizeDtype.INT8, QuantizeDtype.INT16, _FloatDtype.FP16
)
Precision.mxfp4 = Precision(None, None, _custom_name="mxfp4")
Precision.q8_0 = Precision(None, None, _custom_name="q8_0")
Precision.q4_0 = Precision(None, None, _custom_name="q4_0")
Precision.mixed = Precision(None, None, _custom_name="mixed")
Precision.mixed_with_float = Precision(None, None, _custom_name="mixed_with_float")


@unique
class SourceModelFormat(Enum):
    ONNX = 0
    TORCHSCRIPT = 1


SampleInputsType = dict[str, list[np.ndarray]]
