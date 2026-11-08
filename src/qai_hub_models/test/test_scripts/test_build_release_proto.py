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

from qai_hub_models import Precision
from qai_hub_models._version import __version__
from qai_hub_models.configs._info_yaml_enums import MODEL_USE_CASE
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.devices_and_chipsets_yaml import (
    ALLOWED_SIMILAR_DEVICES,
    DevicesAndChipsetsYaml,
    _load_similar_devices_raw,
)
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scripts.build_release_proto import (
    _build_release_assets_proto,
    _manifest_filter_fields,
    cmd_aws,
    cmd_website,
)
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
    """Cross-check hardcoded release-asset chipsets against the published platform proto.

    Similar-device chipsets are stripped from the platform proto by default (their
    perf is borrowed, not measured), except those kept via ``ALLOWED_SIMILAR_DEVICES``.
    This enforces both directions of that allowlist in a single pass over all models
    (scanning every model's release-assets.yaml is expensive, so we only do it once):

    1. Every chipset a release asset references must survive into the published proto,
       or the CLI ships assets for a chipset that isn't in its platform.
    2. Every allowlisted similar device must actually be needed by a release asset, so
       the allowlist doesn't re-add borrowed-perf chipsets to the proto for no reason.
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

    proto = DevicesAndChipsetsYaml.load().to_proto("0.0.0")
    published_chipsets = {c.name for c in proto.chipsets}

    # (1) No used chipset may be missing from the published proto.
    missing = {c: models for c, models in used.items() if c not in published_chipsets}
    assert not missing, (
        "These chipsets are referenced by hardcoded release assets but are stripped "
        "from the published platform proto (they belong to 'similar' devices whose "
        "perf is borrowed rather than measured):\n"
        + "\n".join(f"  {c}: used by {sorted(models)}" for c, models in missing.items())
        + "\n\nAdd the corresponding device(s) to ALLOWED_SIMILAR_DEVICES in "
        "qai_hub_models/scorecard/devices_and_chipsets_yaml.py so the chipset is "
        "kept in the platform proto."
    )

    # (2) Every allowlisted similar device must be needed by some release asset.
    similar_devices = _load_similar_devices_raw().devices
    unused = {
        name: similar_devices[name].chipset
        for name in ALLOWED_SIMILAR_DEVICES
        if similar_devices[name].chipset not in used
    }
    assert not unused, (
        "These devices are in ALLOWED_SIMILAR_DEVICES but no release asset references "
        "their chipset:\n"
        + "\n".join(f"  {name} ({chipset})" for name, chipset in unused.items())
        + "\n\nRemove them from ALLOWED_SIMILAR_DEVICES in "
        "qai_hub_models/scorecard/devices_and_chipsets_yaml.py; keeping them re-adds "
        "borrowed-perf chipsets to the published platform proto for no reason."
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
            data = ruamel.yaml.YAML().load(text)
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
        path=ScorecardProfilePath.GENIE,
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
