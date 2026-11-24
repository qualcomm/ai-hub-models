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
from google.protobuf.json_format import Parse
from qai_hub_models_cli.proto import model_manifest_pb2, release_manifest_pb2
from qai_hub_models_cli.proto_helpers._common import read_proto_yaml, safe_load_yaml

from qai_hub_models import Precision
from qai_hub_models._version import __version__
from qai_hub_models.configs._info_yaml_enums import MODEL_USE_CASE
from qai_hub_models.scorecard import ScorecardInferencePath
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scripts.build_release_proto import (
    _build_release_assets_proto,
    _manifest_filter_fields,
    cmd_aws,
    cmd_website,
)
from qai_hub_models.utils._platform_helpers import platform_chipsets
from qai_hub_models.utils.path_helpers import MODEL_IDS

SAMPLE_MODELS = {"mobilenet_v2", "aotgan"}
RESTRICTED_MODEL = "yolov8_det"


@patch(
    "qai_hub_models.scripts.build_release_proto._similar_chipsets",
    return_value=frozenset({"qualcomm-sa8255p", "qualcomm-sa8650p"}),
)
def test_manifest_drops_similar_chipsets(mock_similar: MagicMock) -> None:
    """Similar-device chipsets are dropped from a model's supported_chipsets."""
    perf = MagicMock()
    perf.for_each_entry = MagicMock()
    perf.supported_chipsets = [
        "qualcomm-snapdragon-8-gen-3",  # workbench: kept
        "qualcomm-sa8255p",  # similar: dropped
        "qualcomm-sa8650p",  # similar: dropped
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


def test_release_asset_chipsets_match_published_proto() -> None:
    """Every chipset a hardcoded release asset references must be in the published platform.

    Similar-device chipsets are not in the platform (their perf is borrowed, not
    measured), so an asset that references one would ship for a chipset the CLI
    does not know about.
    """
    # Map every chipset referenced by a hardcoded release asset to the models using it.
    # Only ``chipset_assets`` contribute; universal assets are chipset-agnostic.
    used: dict[str, list[str]] = {}
    for model_id in MODEL_IDS:
        release_assets = QAIHMModelReleaseAssets.from_model(
            model_id, not_exists_ok=True
        )
        for prec_details in release_assets.precisions.values():
            for chipset in prec_details.chipset_assets:
                used.setdefault(chipset, []).append(model_id)

    published_chipsets = set(platform_chipsets())
    missing = {c: models for c, models in used.items() if c not in published_chipsets}
    assert not missing, (
        "These chipsets are referenced by hardcoded release assets but are not in the "
        "published platform proto (similar-device chipsets are never published, since "
        "their perf is borrowed rather than measured):\n"
        + "\n".join(f"  {c}: used by {sorted(models)}" for c, models in missing.items())
    )


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "release_output"


def test_cmd_website(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir), version=__version__, models=SAMPLE_MODELS
    )
    cmd_website(args)

    assert output_dir.exists()
    assert (output_dir / "asset_bases.yaml").exists()
    assert (output_dir / "platform.yaml").exists()

    asset_bases_text = (output_dir / "asset_bases.yaml").read_text()
    version_tag = f"v{__version__}" if not __version__.startswith("v") else __version__
    assert f"ai-hub-models/blob/{version_tag}" in asset_bases_text
    assert "ai-hub-models/blob/main" not in asset_bases_text
    assert "ai-hub-models/tree/main" not in asset_bases_text
    assert "ai-hub-apps/tree/main" in asset_bases_text

    for model_id in sorted(SAMPLE_MODELS):
        model_dir = output_dir / "models" / model_id
        assert model_dir.exists()
        assert not (model_dir / "info.yaml").exists()

        internal_manifest = read_proto_yaml(
            model_dir / "internal_manifest.yaml", model_manifest_pb2.ModelManifest
        )
        assert internal_manifest.id == model_id
        assert internal_manifest.aihm_version == __version__
        assert (
            internal_manifest._internal.status
            == model_manifest_pb2.ModelManifest.Internal.MODEL_STATUS_PUBLISHED
        )

        release_assets_yaml = model_dir / "release-assets.yaml"
        if release_assets_yaml.exists():
            text = release_assets_yaml.read_text()
            data = safe_load_yaml(text)
            assert isinstance(data, dict)
            assert data["assets"]
            assert all(a["precision"].startswith("PRECISION_") for a in data["assets"])
            assert all(a["runtime"].startswith("RUNTIME_") for a in data["assets"])
            assert all(a["download_url"] for a in data["assets"])
            assert not any("s3_key" in a for a in data["assets"])


def test_cmd_aws(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir),
        version=__version__,
        models=SAMPLE_MODELS,
        upload=False,
        dangerously_upload_partial_manifest=True,
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
            manifest = release_manifest_pb2.ReleaseManifest()
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
                assert entry.manifest_urls.manifest.endswith(f"/manifest{ext}")
                if variant == "internal":
                    assert entry.manifest_urls.internal_manifest.endswith(
                        f"/internal_manifest{ext}"
                    )
                else:
                    assert not entry.manifest_urls.internal_manifest

        for model_id in sorted(SAMPLE_MODELS):
            model_dir = variant_dir / "models" / model_id
            assert model_dir.exists()
            assert (model_dir / "manifest.json").exists()
            assert (model_dir / "manifest.pb").exists()

            internal_json = model_dir / "internal_manifest.json"
            internal_pb = model_dir / "internal_manifest.pb"
            if variant == "public":
                assert not internal_json.exists()
                assert not internal_pb.exists()
            else:
                from_pb = model_manifest_pb2.ModelManifest()
                from_pb.ParseFromString(internal_pb.read_bytes())
                from_json = model_manifest_pb2.ModelManifest()
                Parse(internal_json.read_text(), from_json)
                assert from_pb == from_json
                assert from_pb.id == model_id
                assert from_pb.HasField("_internal")

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


def test_cmd_aws_partial_selection_skips_manifest(output_dir: Path) -> None:
    args = argparse.Namespace(
        output_dir=str(output_dir),
        version=__version__,
        models=SAMPLE_MODELS,
        upload=False,
        dangerously_upload_partial_manifest=False,
    )
    cmd_aws(args)

    for variant in ["public", "internal"]:
        assert (output_dir / variant / "platform.json").exists()
        assert not (output_dir / variant / "manifest.json").exists()
        assert not (output_dir / variant / "manifest.pb").exists()


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
        dangerously_upload_partial_manifest=False,
    )
    cmd_aws(args_aws)
    model_dir = output_dir / "aws" / "models" / RESTRICTED_MODEL
    assert not (model_dir / "release-assets.json").exists()
    assert not (model_dir / "release-assets.pb").exists()


def test_chipset_assets_mirrored_onto_compute_peer(tmp_path: Path) -> None:
    """X Plus reuses the X Elite artifact, so its entry carries the same URL."""
    assets = QAIHMModelReleaseAssets()
    assets.add_asset(
        QAIHMModelReleaseAssets.AssetDetails(s3_key="k/x-elite-genie"),
        precision=Precision.w4a16,
        chipset="qualcomm-snapdragon-x-elite",
        path=ScorecardInferencePath.GENIE,
    )
    model_id = "llama_v3_2_3b_instruct"
    (tmp_path / model_id).mkdir()
    with patch(
        "qai_hub_models.scorecard.release_assets_yaml.QAIHM_MODELS_ROOT", tmp_path
    ):
        assets.to_model_yaml(model_id)
        proto = _build_release_assets_proto(model_id, "0.0.0")

    assert proto is not None
    urls = {a.chipset: a.download_url for a in proto.assets}
    assert (
        urls["qualcomm-snapdragon-x-plus-8-core"] == urls["qualcomm-snapdragon-x-elite"]
    )
