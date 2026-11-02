# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

from qai_hub_models._version import __version__ as qaihm_version
from qai_hub_models.configs.manifest_yaml import MODEL_STATUS, QAIHMModelManifest
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scripts.release_huggingface_model_cards import (
    HF_REPO_NAMES_TO_NEVER_DEPRECATE,
    release_hf_model_cards,
)
from qai_hub_models.scripts.utils.huggingface_model_card_helpers import (
    generate_hf_manifest,
)
from qai_hub_models.utils.asset_loaders import load_json
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT

# Fake deprecated model name that doesn't exist in MODEL_IDS
FAKE_DEPRECATED_MODEL = "fake_deprecated_model_for_testing"
# Model from HF_REPO_NAMES_TO_NEVER_DEPRECATE that should be skipped
NEVER_DEPRECATE_MODEL = next(iter(HF_REPO_NAMES_TO_NEVER_DEPRECATE))


def test_generate_and_dry_run_release_hf_model_cards() -> None:
    """Test release_hf_model_cards generates model cards for multiple models in dry-run mode."""
    # Get all models
    model_ids = MODEL_IDS
    all_model_names = [
        QAIHMModelManifest.from_model(model_id).name for model_id in model_ids
    ]

    # Create mock HF models list that includes all real models plus a fake deprecated one
    mock_hf_models = [
        # All existing model names as if they were on Hugging Face
        # Leave out 1 model to simulate an un-published model.
        *[
            SimpleNamespace(id=f"qualcomm/{name}", tags=[])
            for name in all_model_names[:-1]
        ],
        # Fake deprecated model
        SimpleNamespace(id=f"qualcomm/{FAKE_DEPRECATED_MODEL}", tags=[]),
        # Model from HF_REPO_NAMES_TO_NEVER_DEPRECATE (should not be deprecated)
        SimpleNamespace(id=f"qualcomm/{NEVER_DEPRECATE_MODEL}", tags=[]),
    ]
    hf_list_models_patch = mock.patch(
        "qai_hub_models.scripts.release_huggingface_model_cards.HfApi.list_models",
        return_value=mock_hf_models,
    )

    # Mock fetch_static_assets to avoid S3 access (used by get_download_links_rows and generate_hf_manifest)
    def mock_fetch_static_assets(
        model_id: str,
        runtime: object,
        precision: object,
        device_or_chipset: str | None = None,
        **kwargs: object,
    ) -> tuple[None, str]:
        # Return a mock download URL based on inputs
        chipset_part = f"_{device_or_chipset}" if device_or_chipset else ""
        url = f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/0.0.0/{model_id}/{precision}_{runtime.name}{chipset_part}.bin"  # type: ignore[attr-defined]
        return None, url

    fetch_static_assets_patch = mock.patch(
        "qai_hub_models.scripts.utils.huggingface_model_card_helpers.fetch_static_assets",
        side_effect=mock_fetch_static_assets,
    )

    with (
        hf_list_models_patch,
        fetch_static_assets_patch,
        TemporaryDirectory() as tmpdir,
    ):
        output_dir = Path(tmpdir)
        failed_models = release_hf_model_cards(
            model_list=model_ids,
            version="0.0.0",
            deprecate_removed_models=True,
            output_dir=output_dir,
            dry_run=True,
        )

        # Verify no models failed
        assert failed_models == [], f"Expected no failures, but got: {failed_models}"

        # Verify output for each model
        for model_id in model_ids:
            model_output_dir = output_dir / model_id
            if (
                QAIHMModelInternalConfig.from_model(model_id).status
                != MODEL_STATUS.PUBLISHED
            ):
                assert not model_output_dir.exists(), (
                    f"Expected non-published model {model_id} to be skipped, but {model_output_dir} exists"
                )
                continue
            assert model_output_dir.is_dir(), (
                f"Expected directory {model_output_dir} to exist"
            )
            assert (model_output_dir / "README.md").is_file(), (
                f"Expected {model_output_dir / 'README.md'} to exist"
            )
            assert (model_output_dir / "LICENSE").is_file(), (
                f"Expected {model_output_dir / 'LICENSE'} to exist"
            )

        # Verify deprecated model was processed
        deprecations_dir = output_dir / "deprecations"
        assert deprecations_dir.is_dir(), "Expected deprecations directory to exist"

        deprecated_model_dir = deprecations_dir / FAKE_DEPRECATED_MODEL
        assert deprecated_model_dir.is_dir(), (
            f"Expected deprecated model dir {deprecated_model_dir} to exist"
        )

        readme_path = deprecated_model_dir / "README.md"
        assert readme_path.is_file(), f"Expected {readme_path} to exist"

        readme_content = readme_path.read_text()
        assert "deprecated" in readme_content.lower(), (
            f"README.md for {FAKE_DEPRECATED_MODEL} should contain deprecation notice"
        )

        # Verify model from HF_REPO_NAMES_TO_NEVER_DEPRECATE was NOT deprecated
        never_deprecate_dir = deprecations_dir / NEVER_DEPRECATE_MODEL
        assert not never_deprecate_dir.exists(), (
            f"Model {NEVER_DEPRECATE_MODEL} should NOT be deprecated (in HF_REPO_NAMES_TO_NEVER_DEPRECATE)"
        )

        # Verify release_assets.json is created for models with release assets
        models_with_manifest = 0
        for model_id in model_ids:
            sc = QAIHMModelInternalConfig.from_model(model_id)
            if sc.status != MODEL_STATUS.PUBLISHED:
                continue
            manifest = sc.manifest
            release_assets = QAIHMModelReleaseAssets.from_model(
                model_id, not_exists_ok=True
            )

            model_output_dir = output_dir / model_id
            release_manifest_path = model_output_dir / "release_assets.json"

            # Models with release assets and no sharing restriction should have manifest
            if not release_assets.empty and not manifest.restrict_model_sharing:
                assert release_manifest_path.is_file(), (
                    f"Expected {release_manifest_path} to exist for model {model_id}"
                )
                models_with_manifest += 1

                # Verify manifest structure
                release_manifest = load_json(release_manifest_path)
                assert "version" in release_manifest, (
                    "release_assets.json should have 'version' key"
                )
                assert "precisions" in release_manifest, (
                    "release_assets.json should have 'precisions' key"
                )

                # Verify at least one precision has download_url
                for precision_data in release_manifest["precisions"].values():
                    if "universal_assets" in precision_data:
                        for runtime, asset_data in precision_data[
                            "universal_assets"
                        ].items():
                            assert "download_url" in asset_data, (
                                f"Asset {runtime} should have 'download_url'"
                            )
            # Models without release assets should not have manifest
            elif release_assets.empty:
                assert not release_manifest_path.exists(), (
                    f"Model {model_id} has no release assets, should not have release_assets.json"
                )

        # Ensure we actually tested some models with manifests
        assert models_with_manifest > 0, (
            "Expected at least one model with release assets to verify release_assets.json"
        )


def test_generate_hf_manifest() -> None:
    """Test generate_hf_manifest() creates manifest with download URLs."""
    # Test with a known model that has release assets
    model_id = "quicksrnetlarge"  # Known to have release-assets.yaml

    # Mock fetch_static_assets to avoid S3 access
    def mock_fetch_static_assets(
        model_id: str,
        runtime: object,
        precision: object,
        device_or_chipset: str | None = None,
        **kwargs: object,
    ) -> tuple[None, str]:
        # Return a mock download URL based on inputs
        chipset_part = f"_{device_or_chipset}" if device_or_chipset else ""
        url = f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/{qaihm_version}/{model_id}/{precision}_{runtime.name}{chipset_part}.bin"  # type: ignore[attr-defined]
        return None, url

    with mock.patch(
        "qai_hub_models.scripts.utils.huggingface_model_card_helpers.fetch_static_assets",
        side_effect=mock_fetch_static_assets,
    ):
        manifest = generate_hf_manifest(model_id, qaihm_version)

        # Should return a valid manifest
        assert manifest is not None, f"Expected manifest for {model_id}"
        assert not manifest.empty, "Manifest should not be empty"

        # Verify structure
        assert len(manifest.precisions) > 0, (
            "Manifest should have at least one precision"
        )

        # Verify download URLs are populated (not s3_keys)
        for precision_details in manifest.precisions.values():
            # Check universal assets
            for path, asset_details in precision_details.universal_assets.items():
                assert asset_details.download_url is not None, (
                    f"Universal asset {path} should have download_url"
                )
                assert asset_details.s3_key is None, (
                    "Manifest should not include s3_key (only download_url)"
                )

            # Check chipset assets
            for chipset, chipset_paths in precision_details.chipset_assets.items():
                for path, asset_details in chipset_paths.items():
                    assert asset_details.download_url is not None, (
                        f"Chipset asset {chipset}/{path} should have download_url"
                    )
                    assert asset_details.s3_key is None, (
                        "Manifest should not include s3_key (only download_url)"
                    )

    # Verify to_json / from_json round-trip
    with TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "release_assets.json"
        manifest.to_json(json_path)
        assert json_path.exists()
        loaded = QAIHMModelReleaseAssets.from_json(json_path)
        assert loaded == manifest

    # Test edge case: model with no release assets
    # Find a model without release-assets.yaml
    model_without_assets = next(
        model_id
        for model_id in MODEL_IDS
        if not (QAIHM_MODELS_ROOT / model_id / "release-assets.yaml").exists()
    )

    manifest_none = generate_hf_manifest(model_without_assets, qaihm_version)
    assert manifest_none is None, (
        f"Model {model_without_assets} has no release assets, should return None"
    )
