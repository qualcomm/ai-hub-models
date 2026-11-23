# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os

from qai_hub.public_api_pb2 import Framework

from qai_hub_models import (
    InferenceEngine,
    Precision,
    QAIRTVersion,
    TargetRuntime,
)
from qai_hub_models.scorecard.device import cs_universal
from qai_hub_models.scorecard.envvars import QAIRTVersionEnvvar
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.test.test_models.test_common import reset_hub_frameworks_patches
from qai_hub_models.utils.set_env import set_temp_env


def test_profile_qnn_version() -> None:
    """
    This verifies behavior of ScorecardInferencePath.get_inference_options(Precision.float, cs_universal) with
    different combinations of:
      * default AI Hub Workbench QAIRT version
      * default AI Hub Models QAIRT version
      * QAIRT version override environment variables in the test environment
    """
    # Patch frameworks so this test continues to work regardless of AI Hub Workbench version changes.
    frameworks = [
        Framework(
            name="QAIRT",
            api_tags=[],
            api_version="2.31",
            full_version="2.31.0.250130151446_114721",
        ),
        Framework(
            name="QAIRT",
            api_tags=[QAIRTVersion.DEFAULT_AIHUB_TAG],
            api_version="2.32",
            full_version="2.32.6.250402152434_116405",
        ),
        Framework(
            name="QAIRT",
            api_tags=[QAIRTVersion.LATEST_AIHUB_TAG],
            api_version="2.33",
            full_version="2.33.0.250327124043_117917",
        ),
    ]

    # Test working AI Hub Workbench instance
    for api_version, per_runtime_api_version in (
        ("2.32", {InferenceEngine.ONNX: "2.33"}),
        ("2.31", {InferenceEngine.ONNX: "2.32"}),
    ):
        with reset_hub_frameworks_patches(
            frameworks, api_version, per_runtime_api_version
        ):
            profile_path = ScorecardInferencePath.TFLITE
            assert not profile_path.runtime.qairt_version_changes_compilation
            assert profile_path.runtime not in per_runtime_api_version
            default_qaihm_qnn_flag = (
                profile_path.runtime.default_qairt_version.explicit_hub_option
            )

            # Spome paths have a special QAIRT version that differs from AI Hub Models' default.
            paths_with_different_qairt_version: list[
                tuple[ScorecardInferencePath, str]
            ] = []
            for profile_path_with_different_qairt_version in ScorecardInferencePath:
                if (
                    profile_path_with_different_qairt_version.runtime.inference_engine
                    not in per_runtime_api_version
                ):
                    continue

                assert (
                    not profile_path_with_different_qairt_version.runtime.is_aot_compiled
                    or profile_path_with_different_qairt_version.runtime.qairt_version_changes_compilation
                )
                path_flag_with_different_qairt_version = TargetRuntime.PRECOMPILED_QNN_ONNX.default_qairt_version.explicit_hub_option
                paths_with_different_qairt_version.append(
                    (
                        profile_path_with_different_qairt_version,
                        path_flag_with_different_qairt_version,
                    )
                )

            # QAIRT version set to QAIHM default
            with set_temp_env(
                {QAIRTVersionEnvvar.VARNAME: QAIRTVersionEnvvar.default()}
            ):
                assert QAIRTVersion.HUB_FLAG not in profile_path.get_inference_options(
                    Precision.float, cs_universal
                )
                assert default_qaihm_qnn_flag in profile_path.get_inference_options(
                    Precision.float,
                    cs_universal,
                    include_default_qaihm_qnn_version=True,
                )
                for (
                    profile_path_with_different_qairt_version,
                    path_flag_with_different_qairt_version,
                ) in paths_with_different_qairt_version:
                    assert (
                        QAIRTVersion.HUB_FLAG
                        not in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float, cs_universal
                        )
                    )
                    assert (
                        path_flag_with_different_qairt_version
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float,
                            cs_universal,
                            include_default_qaihm_qnn_version=True,
                        )
                    )

            # No flag set (same behavior as if flag was the same as the default QAIRT version)
            with set_temp_env({QAIRTVersionEnvvar.VARNAME: None}):
                assert QAIRTVersion.HUB_FLAG not in profile_path.get_inference_options(
                    Precision.float, cs_universal
                )
                assert default_qaihm_qnn_flag in profile_path.get_inference_options(
                    Precision.float,
                    cs_universal,
                    include_default_qaihm_qnn_version=True,
                )
                for (
                    profile_path_with_different_qairt_version,
                    path_flag_with_different_qairt_version,
                ) in paths_with_different_qairt_version:
                    assert (
                        QAIRTVersion.HUB_FLAG
                        not in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float, cs_universal
                        )
                    )
                    assert (
                        path_flag_with_different_qairt_version
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float,
                            cs_universal,
                            include_default_qaihm_qnn_version=True,
                        )
                    )

            # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag.
            with set_temp_env({QAIRTVersionEnvvar.VARNAME: "2.31"}):
                os.environ[QAIRTVersionEnvvar.VARNAME] = "2.31"
                override_qairt_flag = QAIRTVersion("2.31").explicit_hub_option
                assert override_qairt_flag in profile_path.get_inference_options(
                    Precision.float, cs_universal
                )
                assert override_qairt_flag in profile_path.get_inference_options(
                    Precision.float,
                    cs_universal,
                    include_default_qaihm_qnn_version=True,
                )
                for (
                    profile_path_with_different_qairt_version,
                    _,
                ) in paths_with_different_qairt_version:
                    assert (
                        override_qairt_flag
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float, cs_universal
                        )
                    )
                    assert (
                        override_qairt_flag
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float,
                            cs_universal,
                            include_default_qaihm_qnn_version=True,
                        )
                    )

            # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag
            # It is still explicit even if it's the default AI Hub Workbench version.
            with set_temp_env(
                {QAIRTVersionEnvvar.VARNAME: QAIRTVersion.DEFAULT_AIHUB_TAG}
            ):
                os.environ[QAIRTVersionEnvvar.VARNAME] = QAIRTVersion.DEFAULT_AIHUB_TAG
                override_qairt_flag = QAIRTVersion(
                    QAIRTVersion.DEFAULT_AIHUB_TAG
                ).explicit_hub_option
                assert override_qairt_flag in profile_path.get_inference_options(
                    Precision.float, cs_universal
                )
                assert override_qairt_flag in profile_path.get_inference_options(
                    Precision.float,
                    cs_universal,
                    include_default_qaihm_qnn_version=True,
                )
                for (
                    profile_path_with_different_qairt_version,
                    _,
                ) in paths_with_different_qairt_version:
                    assert (
                        override_qairt_flag
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float, cs_universal
                        )
                    )
                    assert (
                        override_qairt_flag
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float,
                            cs_universal,
                            include_default_qaihm_qnn_version=True,
                        )
                    )

            # The QAIRT version is always included explicitly if not set to the AI Hub Models default tag.
            # It is still explicit even if the selected API version is the same as the AI Hub Models default tag.
            with set_temp_env(
                {
                    QAIRTVersionEnvvar.VARNAME: profile_path.runtime.default_qairt_version.api_version
                }
            ):
                override_qairt_flag = (
                    profile_path.runtime.default_qairt_version.explicit_hub_option
                )
                assert override_qairt_flag in profile_path.get_inference_options(
                    Precision.float, cs_universal
                )
                assert override_qairt_flag in profile_path.get_inference_options(
                    Precision.float,
                    cs_universal,
                    include_default_qaihm_qnn_version=True,
                )
                for (
                    profile_path_with_different_qairt_version,
                    _,
                ) in paths_with_different_qairt_version:
                    assert (
                        override_qairt_flag
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float, cs_universal
                        )
                    )
                    assert (
                        override_qairt_flag
                        in profile_path_with_different_qairt_version.get_inference_options(
                            Precision.float,
                            cs_universal,
                            include_default_qaihm_qnn_version=True,
                        )
                    )
            # fmt: on


def test_single_public_path_per_runtime() -> None:
    public_paths_per_runtime: dict[TargetRuntime, ScorecardInferencePath] = {}
    for path in ScorecardInferencePath:
        if path.is_published:
            if other_path := public_paths_per_runtime.get(path.runtime):
                raise ValueError(
                    f"Only one ScorecardInferencePath can be public for each runtime. Both {path} and {other_path} are public and use {path.runtime}."
                )
            public_paths_per_runtime[path.runtime] = path
