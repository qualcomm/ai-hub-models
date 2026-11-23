# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest
import qai_hub as hub
from qai_hub.client import JobStatus, JobType
from qai_hub.public_api_pb2 import (
    CompileDetail,
    CompileJobResult,
    InferenceJobResult,
    JobResult,
    ProfileDetail,
    ToolVersion,
)

import qai_hub_models._version as pkg_version
from qai_hub_models import QAIRTVersion, TargetRuntime
from qai_hub_models.configs.tool_versions import ToolVersions

RESULTS_PATCH_TARGET = "qai_hub_models.configs.tool_versions.get_job_results"


def test_geniex_field_round_trips_to_proto() -> None:
    tv = ToolVersions(
        geniex="v0.7.0",
        qairt=QAIRTVersion("2.45", validate_exists_on_ai_hub=False),
    )
    proto = tv.to_proto()
    assert proto.geniex == "v0.7.0"
    assert proto.qairt == "2.45"

    empty_proto = ToolVersions().to_proto()
    assert not empty_proto.HasField("geniex")

    assert ToolVersions(geniex="v0.7.0") == ToolVersions(geniex="v0.7.0")
    assert ToolVersions(geniex="v0.7.0") != ToolVersions(geniex="v0.7.1")


def test_extract_tool_versions_from_compiled_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pkg_version, "__version__", "1.2.3")
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QAIRT_SDK_VERSION: "2.28"},
    )
    m.get_producer.return_value = MagicMock(
        spec=hub.CompileJob, _job_type=JobType.COMPILE
    )
    assert ToolVersions.from_compiled_model(m, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False),
        ai_hub_models="1.2.3",
    )

    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
    )
    m.get_producer.return_value = MagicMock(
        spec=hub.CompileJob, _job_type=JobType.COMPILE
    )
    assert ToolVersions.from_compiled_model(m, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
        ai_hub_models="1.2.3",
    )

    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25"},
    )
    m.get_producer.return_value = None
    with pytest.raises(
        expected_exception=ValueError,
        match=r"Model must be compiled with AI Hub Workbench to extract tool versions\.",
    ):
        ToolVersions.from_compiled_model(m)


def test_extract_tool_versions_from_compile_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pkg_version, "__version__", "1.2.3")

    def _make_results(versions: list[ToolVersion]) -> JobResult:
        return JobResult(
            compile_job_result=CompileJobResult(
                compile_detail=CompileDetail(tool_versions=versions)
            )
        )

    # Compile job: Success
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
    )
    m.get_producer.return_value = j
    j.get_target_model = lambda: m
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
        ai_hub_models="1.2.3",
    )

    # Compile job: Success with explicit qairt version in options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
        options="--qairt_version latest",
    )
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
    )
    m.get_producer.return_value = j
    j.get_target_model = lambda: m
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
        ai_hub_models="1.2.3",
    )

    # Compile job: Failed with no options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions()
    assert ToolVersions.from_job(j, parse_version_tags=True) == ToolVersions()

    # Compile job: Failed
    for rt in TargetRuntime:
        if rt == TargetRuntime.GENIEX_LLAMACPP:
            # GenieX llama.cpp does not compile via AI Hub.
            continue
        j = MagicMock(
            spec=hub.CompileJob,
            _job_type=JobType.COMPILE,
            get_status=lambda: JobStatus(JobStatus.State.FAILED),
            options=rt.aihub_target_runtime_flag,
        )
        if rt.qairt_version_changes_compilation:
            assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
                qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
            )
            assert ToolVersions.from_job(
                j, add_aihm_version=True, parse_version_tags=True
            ) == ToolVersions(qairt=QAIRTVersion.default())
        else:
            assert ToolVersions.from_job(j) == ToolVersions()
            assert (
                ToolVersions.from_job(j, add_aihm_version=True, parse_version_tags=True)
                == ToolVersions()
            )

    # Compile job: Failed with qairt version in options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=2.28",
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(
        job=j, add_aihm_version=True, parse_version_tags=True
    ) == ToolVersions(qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False))

    # Compile job: Failed with explicit qairt version tag in options
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=latest",
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("latest", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(
        j, add_aihm_version=True, parse_version_tags=True
    ) == ToolVersions(qairt=QAIRTVersion.latest())

    # Compile job: Not QAIRT asset
    j = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
        _owner=MagicMock(),
        job_id="0",
    )
    m = MagicMock(spec=hub.Model, metadata={})
    m.get_producer.return_value = j
    j.get_target_model = lambda: m
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results([ToolVersion(name="ONNX", version="1.22.1")]),
    ):
        assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
            onnx="1.22.1", ai_hub_models="1.2.3"
        )


def test_extract_tool_versions_from_link_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pkg_version, "__version__", "1.2.3")
    # LINK job: Success
    j = MagicMock(
        spec=hub.LinkJob,
        _job_type=JobType.LINK,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
    )
    m.get_producer.return_value = j
    j.get_target_model = lambda: m
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
        ai_hub_models="1.2.3",
    )

    # LINK job: Failed
    jc = MagicMock(
        spec=hub.CompileJob,
        _job_type=JobType.COMPILE,
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
        options="--qairt_version latest",
    )
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
    )
    m.get_producer.return_value = jc
    j = MagicMock(
        spec=hub.LinkJob,
        _job_type=JobType.LINK,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
        models=[m, m],
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
        ai_hub_models="1.2.3",
    )

    # Link job: Failed, but source model was not created by AI Hub
    m = MagicMock(
        spec=hub.Model,
        metadata={hub.ModelMetadataKey.QNN_SDK_VERSION: "2.25.1234"},
    )
    m.get_producer.return_value = None
    j = MagicMock(
        spec=hub.LinkJob,
        _job_type=JobType.LINK,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
        models=[m, m],
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions()


def test_extract_tool_versions_from_inference_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pkg_version, "__version__", "1.2.3")

    def _make_results(versions: list[ToolVersion]) -> JobResult:
        return JobResult(
            inference_job_result=InferenceJobResult(
                detail=ProfileDetail(tool_versions=versions)
            )
        )

    # Inference job: Success
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results([ToolVersion(name="QAIRT", version="2.25.1234")]),
    ):
        assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
            qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
            ai_hub_models="1.2.3",
        )

    # Inference job: Success (QNN instead of QAIRT)
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(
        RESULTS_PATCH_TARGET,
        lambda *_: _make_results(
            [
                ToolVersion(name="QNN", version="2.25.1234"),
                ToolVersion(name="TensorFlow Lite", version="1.22.1"),
                ToolVersion(name="Lite RT", version="1.4.2"),
            ]
        ),
    ):
        assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
            qairt=QAIRTVersion("2.25.1234", validate_exists_on_ai_hub=False),
            tflite="1.22.1",
            ai_hub_models="1.2.3",
            litert="1.4.2",
        )

    # Inference job: Success (no QAIRT version in job results)
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        _owner=MagicMock(),
        job_id="0",
        get_status=lambda: JobStatus(JobStatus.State.SUCCESS),
    )
    with patch(RESULTS_PATCH_TARGET, lambda *_: _make_results([])):
        assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
            ai_hub_models="1.2.3"
        )

    # Inference job: Failed with no options
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="",
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(
        j, add_aihm_version=True, parse_version_tags=True
    ) == ToolVersions(qairt=QAIRTVersion.default())

    # Inference job: Failed with qairt version in options
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=2.28",
    )
    assert ToolVersions.from_job(job=j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(
        job=j, add_aihm_version=True, parse_version_tags=True
    ) == ToolVersions(qairt=QAIRTVersion("2.28", validate_exists_on_ai_hub=False))

    # Inference job: Failed with explicit qairt version tag in options
    j = MagicMock(
        spec=hub.InferenceJob,
        _job_type=JobType.INFERENCE,
        get_status=lambda: JobStatus(JobStatus.State.FAILED),
        options="--qairt_version=latest",
    )
    assert ToolVersions.from_job(j, add_aihm_version=True) == ToolVersions(
        qairt=QAIRTVersion("default", validate_exists_on_ai_hub=False)
    )
    assert ToolVersions.from_job(
        j, add_aihm_version=True, parse_version_tags=True
    ) == ToolVersions(qairt=QAIRTVersion.latest())
