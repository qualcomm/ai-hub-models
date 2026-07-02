# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import ruamel.yaml
from google.protobuf.json_format import Parse
from qai_hub_models_cli.proto import manifest_pb2

from qai_hub_models._version import __version__
from qai_hub_models.configs._info_yaml_enums import MODEL_USE_CASE
from qai_hub_models.scripts.build_release_proto import (
    _manifest_filter_fields,
    _simplify_enum_values_for_website_import,
    cmd_aws,
    cmd_website,
)

SAMPLE_MODELS = {"mobilenet_v2", "aotgan"}
RESTRICTED_MODEL = "yolov8_det"


@patch(
    "qai_hub_models.scripts.build_release_proto._similar_chipsets",
    return_value=frozenset({"qualcomm-sa8255p", "qualcomm-qcs8275"}),
)
def test_manifest_drops_similar_chipsets(mock_similar: MagicMock) -> None:
    """Similar-device chipsets are dropped from a model's supported_chipsets."""
    perf = MagicMock()
    perf.for_each_entry = MagicMock()
    perf.supported_chipsets = [
        "qualcomm-snapdragon-8-gen-3",  # workbench: kept
        "qualcomm-sa8255p",  # similar: dropped
        "qualcomm-qcs8275",  # similar: dropped
    ]

    prec_details = MagicMock()
    prec_details.universal_assets = {}
    prec_details.chipset_assets = {}
    release_assets = MagicMock()
    release_assets.precisions = {MagicMock(): prec_details}

    info = MagicMock()
    info.tags = []
    info.use_case = MODEL_USE_CASE.IMAGE_CLASSIFICATION

    fields = _manifest_filter_fields(release_assets, perf, info)
    assert fields["supported_chipsets"] == ["qualcomm-snapdragon-8-gen-3"]


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "release_output"


def test_simplify_enum_values_for_website_import() -> None:
    assert _simplify_enum_values_for_website_import("PRECISION_W8A8") == "w8a8"
    assert (
        _simplify_enum_values_for_website_import("RUNTIME_QNN_CONTEXT_BINARY")
        == "qnn_context_binary"
    )
    assert _simplify_enum_values_for_website_import("RUNTIME_TFLITE") == "tflite"
    assert _simplify_enum_values_for_website_import("PRECISION_FLOAT") == "float"
    assert (
        _simplify_enum_values_for_website_import("some_other_string")
        == "some_other_string"
    )
    assert _simplify_enum_values_for_website_import(
        {"precision": "PRECISION_W8A16", "name": "foo"}
    ) == {
        "precision": "w8a16",
        "name": "foo",
    }
    assert _simplify_enum_values_for_website_import(["RUNTIME_ONNX", "hello"]) == [
        "onnx",
        "hello",
    ]


def test_cmd_website(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir), version=__version__, models=SAMPLE_MODELS
    )
    cmd_website(args)

    assert output_dir.exists()
    assert (output_dir / "asset_bases.yaml").exists()
    assert (output_dir / "devices_and_chipsets.yaml").exists()

    asset_bases_text = (output_dir / "asset_bases.yaml").read_text()
    version_tag = f"v{__version__}" if not __version__.startswith("v") else __version__
    assert f"ai-hub-models/blob/{version_tag}" in asset_bases_text
    assert "ai-hub-models/blob/main" not in asset_bases_text
    assert "ai-hub-models/tree/main" not in asset_bases_text
    assert "ai-hub-apps/tree/main" in asset_bases_text

    for model_id in sorted(SAMPLE_MODELS):
        model_dir = output_dir / "models" / model_id
        assert model_dir.exists()
        assert (model_dir / "info.yaml").exists()

        release_assets_yaml = model_dir / "release-assets.yaml"
        if release_assets_yaml.exists():
            text = release_assets_yaml.read_text()
            assert "PRECISION_" not in text, f"Unsanitized precision enum in {model_id}"
            assert "RUNTIME_" not in text, f"Unsanitized runtime enum in {model_id}"

            data = ruamel.yaml.YAML().load(text)
            assert isinstance(data, dict)


def test_cmd_aws(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir),
        version=__version__,
        models=SAMPLE_MODELS,
        upload=False,
    )
    cmd_aws(args)

    assert output_dir.exists()

    for variant in ["public", "internal"]:
        variant_dir = output_dir / variant
        assert variant_dir.exists()
        assert (variant_dir / "platform.json").exists()
        assert (variant_dir / "platform.pb").exists()
        manifest_json = variant_dir / "manifest.json"
        manifest_pb = variant_dir / "manifest.pb"
        assert manifest_json.exists()
        assert manifest_pb.exists()

        for manifest_path, ext in [(manifest_json, ".json"), (manifest_pb, ".pb")]:
            manifest = manifest_pb2.ReleaseManifest()
            if ext == ".json":
                Parse(manifest_path.read_text(), manifest)
            else:
                manifest.ParseFromString(manifest_path.read_bytes())

            assert manifest.version == __version__
            assert manifest.platform_url.endswith(f"platform{ext}")

            model_ids_in_manifest = {entry.id for entry in manifest.models}
            for model_id in sorted(SAMPLE_MODELS):
                assert model_id in model_ids_in_manifest

            for entry in manifest.models:
                assert entry.display_name
                assert entry.domain
                assert entry.manifest_urls.info.endswith(f"/info{ext}")

        for model_id in sorted(SAMPLE_MODELS):
            model_dir = variant_dir / "models" / model_id
            assert model_dir.exists()
            assert (model_dir / "info.json").exists()
            assert (model_dir / "info.pb").exists()

            release_assets_json = model_dir / "release-assets.json"
            if release_assets_json.exists():
                data = json.loads(release_assets_json.read_text())
                for asset in data.get("assets", []):
                    assert asset["precision"].startswith("PRECISION_"), (
                        f"JSON should preserve proto enum names, got {asset['precision']}"
                    )
                    assert asset["runtime"].startswith("RUNTIME_"), (
                        f"JSON should preserve proto enum names, got {asset['runtime']}"
                    )


def test_restricted_model_excludes_release_assets(output_dir: Path) -> None:
    models = {RESTRICTED_MODEL}
    args_website = argparse.Namespace(
        output_dir=str(output_dir / "website"), version=__version__, models=models
    )
    cmd_website(args_website)
    model_dir = output_dir / "website" / "models" / RESTRICTED_MODEL
    assert not (model_dir / "release-assets.yaml").exists()

    args_aws = argparse.Namespace(
        output_dir=str(output_dir / "aws"),
        version=__version__,
        models=models,
        upload=False,
    )
    cmd_aws(args_aws)
    model_dir = output_dir / "aws" / "models" / RESTRICTED_MODEL
    assert not (model_dir / "release-assets.json").exists()
    assert not (model_dir / "release-assets.pb").exists()
