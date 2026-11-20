# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from collections.abc import Generator
from unittest.mock import patch

import pytest

from qai_hub_models_cli.cli import main
from qai_hub_models_cli.proto.model_manifest_pb2 import (
    ModelDomain,
    ModelLicense,
    ModelManifest,
    ModelTag,
    ModelUseCase,
)
from qai_hub_models_cli.proto.model_release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.platform_pb2 import ChipsetInfo, PlatformInfo
from qai_hub_models_cli.proto.release_manifest_pb2 import ReleaseManifestModelEntry
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions
from qai_hub_models_cli.versions import MIN_MODEL_FILTER_VERSION


def _fake_model_info() -> ModelManifest:
    return ModelManifest(
        id="mobilenet_v2",
        name="MobileNet V2",
        description="Imagenet classifier and general-purpose backbone.",
        domain=ModelDomain.MODEL_DOMAIN_COMPUTER_VISION,
        use_case=ModelUseCase.MODEL_USE_CASE_IMAGE_CLASSIFICATION,
        tags=[ModelTag.MODEL_TAG_REAL_TIME],
        license_type=ModelLicense.MODEL_LICENSE_APACHE_2_0,
        technical_details=[
            ModelManifest.TechnicalDetail(
                key="Number of parameters", int_value=3500000
            ),
            ModelManifest.TechnicalDetail(key="Model size (MB)", float_value=14.2),
            ModelManifest.TechnicalDetail(key="Architecture", string_value="CNN"),
        ],
        runtime_technical_details=[
            ModelManifest.RuntimeTechnicalDetails(
                runtime=Runtime.RUNTIME_TFLITE,
                technical_details=[
                    ModelManifest.TechnicalDetail(
                        key="Model size (MB)", float_value=5.1
                    ),
                ],
            ),
        ],
    )


def _fake_release_assets() -> ModelReleaseAssets:
    return ModelReleaseAssets(
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_TFLITE,
                download_url="https://example.com/mobilenet_v2-tflite-float.zip",
                tool_versions=ToolVersions(tflite="2.16", ai_hub_models="0.45.0"),
            ),
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_QNN_CONTEXT_BINARY,
                chipset="qualcomm-snapdragon-8-gen-3",
                download_url="https://example.com/mobilenet_v2-qnn-float-sd8g3.zip",
                tool_versions=ToolVersions(qairt="2.31"),
            ),
        ],
    )


def _fake_platform() -> PlatformInfo:
    return PlatformInfo(
        chipsets=[
            ChipsetInfo(
                name="qualcomm-snapdragon-8-gen-3",
                marketing_name="Snapdragon 8 Gen 3",
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _skip_version_check() -> Generator[None]:
    with patch("qai_hub_models_cli.cli._check_version_match"):
        yield


def _fake_manifest_entry() -> ReleaseManifestModelEntry:
    return ReleaseManifestModelEntry(
        id="mobilenet_v2",
        is_quantized=True,
        supported_runtimes=[Runtime.RUNTIME_TFLITE, Runtime.RUNTIME_ONNX],
        supported_chipsets=["qualcomm-snapdragon-8-gen-3"],
    )


@pytest.fixture
def info_mocks() -> Generator[None]:
    with (
        patch(
            "qai_hub_models_cli.cli.get_model_manifest",
            return_value=_fake_model_info(),
        ),
        patch(
            "qai_hub_models_cli.cli.get_model_release_assets",
            return_value=_fake_release_assets(),
        ),
        patch(
            "qai_hub_models_cli.cli.get_platform",
            return_value=_fake_platform(),
        ),
        patch(
            "qai_hub_models_cli.cli.get_release_manifest_entry",
            return_value=_fake_manifest_entry(),
        ),
    ):
        yield


def test_info_full_output(info_mocks: None, capsys: pytest.CaptureFixture[str]) -> None:
    main(["info", "mobilenet_v2", "-v", str(MIN_MODEL_FILTER_VERSION)])
    output = capsys.readouterr().out
    assert "MobileNet V2" in output
    assert "Imagenet classifier" in output
    assert "Computer Vision" in output
    assert "Image Classification" in output
    assert "Real-Time" in output
    assert "Apache-2.0" in output
    assert "3500000" in output
    assert "14.2" in output
    assert "CNN" in output
    # Quantized + Supported Runtimes/Chipsets (from the manifest entry);
    # chipset shows its marketing name.
    assert "Quantized" in output
    assert "Supported Runtimes" in output
    assert "Supported Chipsets" in output
    assert "Snapdragon 8 Gen 3" in output
    # Runtime-specific technical details get their own titled section.
    assert "Technical Details (tflite)" in output
    assert "5.1" in output


def test_info_download_options(
    info_mocks: None, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["info", "mobilenet_v2"])
    output = capsys.readouterr().out
    assert "Download Options" in output
    assert "tflite" in output
    assert "Universal" in output
    assert "qnn_context_binary" in output
    # Chipsets are shown by marketing name, not raw id.
    assert "Snapdragon 8 Gen 3" in output
    assert "qualcomm-snapdragon-8-gen-3" not in output
    assert "qai-hub-models fetch mobilenet_v2" in output
    assert "-c '<chipset>'" in output
    # SDK Versions column lists all set tool versions.
    assert "SDK Versions" in output
    assert "QAIRT 2.31" in output
    assert "TFLite 2.16" in output
    assert "AI Hub Models 0.45.0" in output


def test_info_minimal_model(capsys: pytest.CaptureFixture[str]) -> None:
    minimal_info = ModelManifest(id="my_model", name="My Model")
    empty_assets = ModelReleaseAssets(model_id="my_model", assets=[])
    with (
        patch("qai_hub_models_cli.cli.get_model_manifest", return_value=minimal_info),
        patch(
            "qai_hub_models_cli.cli.get_model_release_assets",
            return_value=empty_assets,
        ),
        patch(
            "qai_hub_models_cli.cli.get_platform",
            return_value=_fake_platform(),
        ),
        patch(
            "qai_hub_models_cli.cli.get_release_manifest_entry",
            return_value=_fake_manifest_entry(),
        ),
    ):
        main(["info", "my_model"])
    output = capsys.readouterr().out
    assert "My Model" in output
    assert "Domain:" not in output
    assert "Technical Details:" not in output
