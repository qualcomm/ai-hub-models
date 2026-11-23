# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, EnumMeta, unique
from typing import TYPE_CHECKING, cast

from typing_extensions import assert_never

from qai_hub_models import (
    Precision,
    QAIRTVersion,
    TargetRuntime,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.scorecard.envvars import (
    DeploymentEnvvar,
    EnabledPathsEnvvar,
    QAIRTVersionEnvvar,
    SpecialPathSetting,
)
from qai_hub_models.scorecard.path_compile import ScorecardCompilePath
from qai_hub_models.utils.base_config import EnumListWithParseableAll
from qai_hub_models.utils.hub_clients import (
    default_hub_client_as,
    get_scorecard_client_or_raise,
)

if TYPE_CHECKING:
    from qai_hub_models.scorecard import ScorecardDevice


class ScorecardInferencePathMeta(EnumMeta):
    def __iter__(self) -> Iterator[ScorecardInferencePath]:  # type: ignore[override]
        return (  # type:ignore[var-annotated]
            cast(ScorecardInferencePath, member) for member in super().__iter__()
        )


@unique
class ScorecardInferencePath(Enum, metaclass=ScorecardInferencePathMeta):
    TFLITE = "tflite"
    TFL_CONVERTER = "tfl_converter"
    QNN_DLC = "qnn_dlc"
    QNN_CONTEXT_BINARY = "qnn_context_binary"
    ONNX = "onnx"
    PRECOMPILED_QNN_ONNX = "precompiled_qnn_onnx"
    GENIE = "genie"
    GENIEX_QAIRT = "geniex_qairt"
    GENIEX_LLAMACPP = "geniex_llamacpp"
    VOICE_AI = "voice_ai"
    ONNX_DML_GPU = "onnx_dml_gpu"
    QNN_DLC_GPU = "qnn_dlc_gpu"

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def from_runtime(runtime: TargetRuntime) -> ScorecardInferencePath:
        """Get the scorecard path that corresponds with default behavior of the given target runtime."""
        return ScorecardInferencePath(runtime.value)

    @property
    def tool_versions(self) -> ToolVersions:
        """Get the versions (currently enabled on Hub) of each runtime that this scorecard path will use."""
        tool_versions = ToolVersions()
        with default_hub_client_as(
            get_scorecard_client_or_raise(DeploymentEnvvar.get())
        ):
            tool_versions.qairt = QAIRTVersionEnvvar.get_qairt_version(self.runtime)
        return tool_versions

    @staticmethod
    def default_paths() -> list[ScorecardInferencePath]:
        """
        The set of first-class scorecard paths.

        Membership here means the path is "published": eligible for
        perf.yaml/numerics.yaml, release-asset publication, spreadsheet
        naming via inference_engine, etc. Not every published path is
        exercised in a default scorecard sweep -- see default_sweep_paths().
        """
        return [
            ScorecardInferencePath.TFLITE,
            ScorecardInferencePath.ONNX,
            ScorecardInferencePath.PRECOMPILED_QNN_ONNX,
            ScorecardInferencePath.QNN_DLC,
            ScorecardInferencePath.QNN_CONTEXT_BINARY,
            ScorecardInferencePath.GENIE,
            ScorecardInferencePath.GENIEX_QAIRT,
            ScorecardInferencePath.GENIEX_LLAMACPP,
            ScorecardInferencePath.VOICE_AI,
        ]

    @staticmethod
    def default_sweep_paths() -> list[ScorecardInferencePath]:
        """
        Paths compiled/profiled when test_runtimes resolves to 'default'.

        Subset of default_paths(): excludes paths that must be requested
        explicitly (currently just GENIE -- kept published so historical
        perf data still renders, but no longer part of the default sweep).
        """
        return [
            p
            for p in ScorecardInferencePath.default_paths()
            if p is not ScorecardInferencePath.GENIE
        ]

    @property
    def spreadsheet_name(self) -> str:
        """Returns the name used for the 'runtime' column in the scorecard results spreadsheet."""
        if self in ScorecardInferencePath.default_paths():
            # Maps both precompiled_context_binary and dlc to "qnn",
            # and both onnx and precompiled onnx to "onnx".
            return self.runtime.inference_engine.value
        return self.value

    @staticmethod
    def _validate_requested_paths(
        valid_test_runtimes: set[str | SpecialPathSetting],
    ) -> None:
        """
        Raise if a requested path name matches no path, prefix, or special setting.

        An unrecognized name would otherwise silently match nothing, leaving
        every path disabled and yielding an empty test parameterization -- a
        scorecard run that passes collection while running zero jobs.

        Parameters
        ----------
        valid_test_runtimes
            The parsed contents of EnabledPathsEnvvar.

        Raises
        ------
        ValueError
            If any requested name is not a path value, an inference engine
            prefix, or a SpecialPathSetting.
        """
        recognized = {p.value for p in ScorecardInferencePath} | {
            p.runtime.inference_engine.value for p in ScorecardInferencePath
        }
        if unknown := sorted(
            x for x in valid_test_runtimes if isinstance(x, str) and x not in recognized
        ):
            raise ValueError(
                f"Unrecognized {EnabledPathsEnvvar.VARNAME} value(s): "
                f"{', '.join(unknown)}. "
                f"Valid path names: {', '.join(sorted(p.value for p in ScorecardInferencePath))}. "
                f"Valid engine prefixes: "
                f"{', '.join(sorted({p.runtime.inference_engine.value for p in ScorecardInferencePath}))}. "
                f"Valid special settings: "
                f"{', '.join(sorted(s.value for s in SpecialPathSetting))}."
            )

    @property
    def enabled(self) -> bool:
        valid_test_runtimes = EnabledPathsEnvvar.get()
        self._validate_requested_paths(valid_test_runtimes)

        # The "default" keyword enables only the default sweep set
        # (default_paths minus GENIE); users can still opt into non-sweep
        # published paths by naming them explicitly.
        if (
            SpecialPathSetting.DEFAULT in valid_test_runtimes
            and self in ScorecardInferencePath.default_sweep_paths()
        ):
            return True

        # Allows users to set 'qnn' to enable both precompiled and dlc,
        # or to set 'onnx' to enable onnx and precompiled onnx.
        if (
            self in ScorecardInferencePath.default_paths()
            and self.runtime.inference_engine.value in valid_test_runtimes
        ):
            return True

        return self.value in valid_test_runtimes

    def should_run_path_for_model(
        self,
        precision: Precision,
        model_supported_runtimes: dict[Precision, list[TargetRuntime]],
    ) -> bool:
        """
        Whether this path should be run for a model with the given supported
        runtimes at the given precision.

        Resolution order based on what is in the EnabledPathsEnvvar:
        1. "default" keyword: enables default paths whose runtime is in the
           model's supported runtimes.
        2. Explicit path name (e.g. qnn_dlc_gpu): enables the path
           if the model supports any runtime with the same inference engine.
        3. Engine prefix (e.g. "qnn"): enables default paths whose runtime
           is in the model's supported runtimes. Non-default paths are not
           matched by a prefix and must be named explicitly (rule 2).

        Parameters
        ----------
        precision
            The precision to check.
        model_supported_runtimes
            Mapping of precision to supported runtimes for the model.

        Returns
        -------
        should_run : bool
            True if this path should be run for the model at the given precision.
        """
        # supported_runtimes lists all valid paths for this model at the
        # given precision. Paths are only excluded from this list for hard
        # constraints (e.g. timeouts, or unsupported combos like w8a16 on
        # tflite). We intersect requested paths with this list so that even
        # explicitly requested paths are skipped when those constraints apply.
        supported_runtimes = model_supported_runtimes.get(precision, [])
        if not supported_runtimes or not self.supports_precision(precision):
            return False

        valid_test_runtimes = EnabledPathsEnvvar.get()
        self._validate_requested_paths(valid_test_runtimes)

        # 1. "default": exact runtime match against supported runtimes.
        # Uses default_sweep_paths() (default_paths minus GENIE) so that
        # 'test_runtimes=default' never fans out to genie without an
        # explicit opt-in via `genie` or the `geniex` engine prefix.
        if (
            SpecialPathSetting.DEFAULT in valid_test_runtimes
            and self in self.default_sweep_paths()
            and self.runtime in supported_runtimes
        ):
            return True

        # 2. Explicit path name: use inference engine match (looser) so that
        #    e.g. requesting "qnn_dlc_gpu" works for a model that only
        #    lists qnn_context_binary (same QNN engine). However, if the
        #    engine itself is excluded (e.g. QNN timed out), no QNN path runs.
        if self.value in valid_test_runtimes:
            supported_engines = {r.inference_engine for r in supported_runtimes}
            return self.runtime.inference_engine in supported_engines

        # 3. Engine prefix: exact runtime match against supported runtimes.
        # Restricted to default_paths() to match the `enabled` property, which
        # gates the same rule. Without this, a prefix like 'qnn' selects
        # unpublished paths (qnn_dlc_gpu) that `enabled` reports as off, and
        # callers keyed on `enabled` blow up.
        return (
            self in ScorecardInferencePath.default_paths()
            and self.runtime.inference_engine.value in valid_test_runtimes
            and self.runtime in supported_runtimes
        )

    def supports_precision(self, precision: Precision) -> bool:
        """Whether this profile path applies to the given model precision."""
        if self == ScorecardInferencePath.QNN_DLC_GPU:
            return not precision.has_quantized_activations

        return self.compile_path.supports_precision(precision)

    @property
    def is_published(self) -> bool:
        """Whether a path is included in perf.yaml, numerics.yaml, allowed in export scripts, etc."""
        return self in ScorecardInferencePath.default_paths()

    @property
    def runtime(self) -> TargetRuntime:
        if (
            self == ScorecardInferencePath.TFLITE  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardInferencePath.TFL_CONVERTER
        ):
            return TargetRuntime.TFLITE
        if (
            self == ScorecardInferencePath.ONNX  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardInferencePath.ONNX_DML_GPU
        ):
            return TargetRuntime.ONNX
        if self == ScorecardInferencePath.PRECOMPILED_QNN_ONNX:
            return TargetRuntime.PRECOMPILED_QNN_ONNX
        if self == ScorecardInferencePath.QNN_CONTEXT_BINARY:
            return TargetRuntime.QNN_CONTEXT_BINARY
        if (
            self == ScorecardInferencePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardInferencePath.QNN_DLC_GPU
        ):
            return TargetRuntime.QNN_DLC
        if self == ScorecardInferencePath.GENIE:
            return TargetRuntime.GENIE
        if self == ScorecardInferencePath.GENIEX_QAIRT:
            return TargetRuntime.GENIEX_QAIRT
        if self == ScorecardInferencePath.GENIEX_LLAMACPP:
            return TargetRuntime.GENIEX_LLAMACPP
        if self == ScorecardInferencePath.VOICE_AI:
            return TargetRuntime.VOICE_AI
        assert_never(self)

    @property
    def compile_path(self) -> ScorecardCompilePath:
        if self == ScorecardInferencePath.TFLITE:
            return ScorecardCompilePath.TFLITE
        if self == ScorecardInferencePath.TFL_CONVERTER:
            return ScorecardCompilePath.TFL_CONVERTER
        if self == ScorecardInferencePath.ONNX:
            return ScorecardCompilePath.ONNX
        if self == ScorecardInferencePath.PRECOMPILED_QNN_ONNX:
            return ScorecardCompilePath.PRECOMPILED_QNN_ONNX
        if self == ScorecardInferencePath.ONNX_DML_GPU:
            return ScorecardCompilePath.ONNX_FP16
        if self == ScorecardInferencePath.QNN_CONTEXT_BINARY:
            return ScorecardCompilePath.QNN_CONTEXT_BINARY
        if (
            self == ScorecardInferencePath.QNN_DLC  # noqa: PLR1714 | Can't merge comparisons and use assert_never
            or self == ScorecardInferencePath.QNN_DLC_GPU
        ):
            return ScorecardCompilePath.QNN_DLC
        if self == ScorecardInferencePath.GENIE:
            return ScorecardCompilePath.GENIE
        if self == ScorecardInferencePath.GENIEX_QAIRT:
            return ScorecardCompilePath.GENIEX_QAIRT
        if self == ScorecardInferencePath.GENIEX_LLAMACPP:
            return ScorecardCompilePath.GENIEX_LLAMACPP
        if self == ScorecardInferencePath.VOICE_AI:
            return ScorecardCompilePath.VOICE_AI
        assert_never(self)

    def get_inference_options(
        self,
        precision: Precision,
        device: ScorecardDevice,
        include_default_qaihm_qnn_version: bool = False,
    ) -> str:
        out = ""
        if (
            self == ScorecardInferencePath.ONNX
            and not precision.has_float_activations
            and not device.supports_fp16_npu
        ):
            out = out + " --onnx_execution_providers qnn"
        if self == ScorecardInferencePath.ONNX_DML_GPU:
            out = out + " --onnx_execution_providers directml"
        if self == ScorecardInferencePath.QNN_DLC_GPU:
            out = (
                out
                + " --compute_unit gpu --qnn_options default_graph_gpu_precision=FLOAT16"
            )

        qairt_version_str = QAIRTVersionEnvvar.get()
        if QAIRTVersionEnvvar.is_default(qairt_version_str):
            # We typically don't want the default QAIRT version added here if it matches with the AI Hub Models default.
            # This allows the export script (which scorecard relies on) to pass in the default version that users will see when they use the CLI.
            #
            # Static models do need this included explicitly because they don't rely on export scripts.
            if include_default_qaihm_qnn_version:
                # Certain runtimes use their own default version of QAIRT.
                # If the user picks our the qaihm_default tag, we need use the runtime's
                # default QAIRT version instead.
                qairt_version = self.runtime.default_qairt_version
                out = out + f" {qairt_version.explicit_hub_option}"
        else:
            qairt_version = QAIRTVersion(qairt_version_str)

            # The explicit option will always pass `--qairt_version 2.XX`,
            # regardless of whether this is the AI Hub Workbench default.
            #
            # This is useful for tracking what QAIRT version applies for scorecard jobs.
            out = out + f" {qairt_version.explicit_hub_option}"

        return out.strip()

    @property
    def website_runtime_name(self) -> str:
        """The name of the runtime on the website that corresponds to this path."""
        if self in [
            ScorecardInferencePath.VOICE_AI,
            ScorecardInferencePath.GENIEX_QAIRT,
            ScorecardInferencePath.GENIE,
            ScorecardInferencePath.GENIEX_LLAMACPP,
        ]:
            return self.value
        return self.runtime.inference_engine.value


class ScorecardInferencePathJITParseableAllList(
    EnumListWithParseableAll[ScorecardInferencePath]
):
    """
    Helper class for parsing.
    If "all" is in the list in the yaml, then it will parse to all JIT
    (on device prepare) profile path values.
    """

    EnumType = ScorecardInferencePath
    ALL = [
        x
        for x in ScorecardInferencePath
        if not x.runtime.is_aot_compiled and not x.runtime.is_orchestrator_runtime
    ]
