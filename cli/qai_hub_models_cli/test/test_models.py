# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Generator
from unittest.mock import patch

import pytest

from qai_hub_models_cli.cli import main
from qai_hub_models_cli.proto.model_manifest_pb2 import ModelDomain, ModelTag
from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    PlatformInfo,
    RuntimeInfo,
)
from qai_hub_models_cli.proto.release_manifest_pb2 import (
    ReleaseManifest,
    ReleaseManifestModelEntry,
)
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime


def _fake_manifest() -> ReleaseManifest:
    return ReleaseManifest(
        version="0.99.0",
        models=[
            ReleaseManifestModelEntry(
                id="mobilenet_v2",
                display_name="MobileNet V2",
                domain=ModelDomain.MODEL_DOMAIN_COMPUTER_VISION,
                is_quantized=True,
                supported_runtimes=[Runtime.RUNTIME_TFLITE, Runtime.RUNTIME_ONNX],
                supported_chipsets=["qualcomm-snapdragon-8gen3"],
                tags=[ModelTag.MODEL_TAG_BACKBONE, ModelTag.MODEL_TAG_REAL_TIME],
            ),
            ReleaseManifestModelEntry(
                id="whisper_small",
                display_name="Whisper Small",
                domain=ModelDomain.MODEL_DOMAIN_AUDIO,
                is_quantized=False,
                supported_runtimes=[Runtime.RUNTIME_ONNX],
                supported_chipsets=["qualcomm-snapdragon-x-elite"],
                tags=[ModelTag.MODEL_TAG_FOUNDATION],
            ),
            ReleaseManifestModelEntry(
                id="qwen3_5_2b",
                display_name="Qwen3.5-2B",
                domain=ModelDomain.MODEL_DOMAIN_GENERATIVE_AI,
                is_quantized=True,
                supported_runtimes=[Runtime.RUNTIME_GENIE],
                supported_chipsets=["qualcomm-snapdragon-x-elite"],
                tags=[ModelTag.MODEL_TAG_LLM, ModelTag.MODEL_TAG_GENERATIVE_AI],
            ),
            ReleaseManifestModelEntry(
                id="qwen3_vl_4b_instruct",
                display_name="Qwen3-VL-4B-Instruct",
                domain=ModelDomain.MODEL_DOMAIN_GENERATIVE_AI,
                is_quantized=True,
                supported_runtimes=[Runtime.RUNTIME_GENIE],
                supported_chipsets=["qualcomm-snapdragon-x-elite"],
                tags=[
                    ModelTag.MODEL_TAG_LLM,
                    ModelTag.MODEL_TAG_VLM,
                    ModelTag.MODEL_TAG_GENERATIVE_AI,
                ],
            ),
            ReleaseManifestModelEntry(
                id="pi05",
                display_name="Pi0.5",
                domain=ModelDomain.MODEL_DOMAIN_MULTIMODAL,
                is_quantized=True,
                supported_runtimes=[Runtime.RUNTIME_ONNX],
                supported_chipsets=["qualcomm-snapdragon-x-elite"],
                tags=[
                    ModelTag.MODEL_TAG_GENERATIVE_AI,
                    ModelTag.MODEL_TAG_ROBOTICS,
                ],
            ),
        ],
    )


def _fake_platform() -> PlatformInfo:
    return PlatformInfo(
        chipsets=[
            ChipsetInfo(
                name="qualcomm-snapdragon-8gen3",
                marketing_name="Snapdragon 8 Gen 3",
            ),
            ChipsetInfo(
                name="qualcomm-snapdragon-x-elite",
                marketing_name="Snapdragon X Elite",
            ),
        ],
        devices=[
            DeviceInfo(name="Samsung Galaxy S24", chipset="qualcomm-snapdragon-8gen3"),
        ],
        runtimes=[
            RuntimeInfo(runtime=Runtime.RUNTIME_TFLITE, is_aot_compiled=True),
            RuntimeInfo(
                runtime=Runtime.RUNTIME_ONNX,
                is_aot_compiled=False,
                display_name="ONNX Runtime",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _skip_version_check() -> Generator[None]:
    with patch("qai_hub_models_cli.cli._check_version_match"):
        yield


@pytest.fixture
def manifest() -> Generator[None]:
    with (
        patch(
            "qai_hub_models_cli.cli.get_release_manifest", return_value=_fake_manifest()
        ),
        patch("qai_hub_models_cli.cli.get_platform", return_value=_fake_platform()),
    ):
        yield


def test_models_table(manifest: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["models", "-v", "0.57.0"])
    output = capsys.readouterr().out
    assert "MobileNet V2" in output
    assert "Whisper Small" in output
    assert "Qwen3.5-2B" in output
    assert "Qwen3-VL-4B-Instruct" in output
    assert "Pi0.5" in output
    assert "Computer Vision" in output
    assert "Audio" in output
    assert "Total: 5 models" in output
    # Use Case + Quantized + Runtimes columns.
    assert "Quantized" in output and "Runtimes" in output
    assert "tflite" in output
    # Model ID is what `export`/`evaluate` take, so the table must show it
    # next to the display name users would otherwise copy.
    assert "Model ID" in output
    assert "mobilenet_v2" in output


def test_models_quiet(manifest: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["models", "-q"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert sorted(lines) == sorted(
        ["mobilenet_v2", "whisper_small", "qwen3_5_2b", "qwen3_vl_4b_instruct", "pi05"]
    )


def test_models_domain_filter(
    manifest: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["models", "--domain", "audio", "-q"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == ["whisper_small"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--quantized"],
            ["mobilenet_v2", "pi05", "qwen3_5_2b", "qwen3_vl_4b_instruct"],
        ),
        # Repeated -r is ANDed and accumulates: mobilenet has onnx but not qnn_dlc.
        # (A single-flag overwrite bug would wrongly keep only the last -r and match.)
        (["-r", "qnn_dlc", "-r", "onnx"], []),
        # The Runtimes column shows display names, so users filter on them too.
        (["-r", "ONNX Runtime"], ["mobilenet_v2", "pi05", "whisper_small"]),
        (["-t", "foundation"], ["whisper_small"]),
        (["-t", "robotics"], ["pi05"]),
        (["-d", "Samsung Galaxy S24"], ["mobilenet_v2"]),  # device resolves to chipset
        (["--aot"], ["mobilenet_v2"]),  # has tflite (AOT); whisper is onnx-only
        (["--jit"], ["mobilenet_v2", "pi05", "whisper_small"]),  # all have onnx (JIT)
        # --llm includes both text-only LLM and VLM (which is also tagged llm).
        (["--llm"], ["qwen3_5_2b", "qwen3_vl_4b_instruct"]),
        # --vlm is VLM-only.
        (["--vlm"], ["qwen3_vl_4b_instruct"]),
    ],
)
def test_models_filters(
    manifest: None,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    expected: list[str],
) -> None:
    main(["models", "-v", "0.57.0", *args, "-q"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines == (expected or ["No models found."])


def test_models_filter_version_gated(
    manifest: None, capsys: pytest.CaptureFixture[str]
) -> None:
    # Non-domain filters require >= 0.57.0; older releases reject them but the
    # new table columns are hidden rather than shown blank.
    main(["models", "-v", "0.55.0", "--quantized"])
    assert "requires version 0.56.0" in capsys.readouterr().out

    main(["models", "-v", "0.55.0"])
    out = capsys.readouterr().out
    assert "Quantized" not in out and "Runtimes" not in out
    assert "Model ID" in out
