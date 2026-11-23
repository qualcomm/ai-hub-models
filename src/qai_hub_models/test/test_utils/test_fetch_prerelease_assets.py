# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import pytest
import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.utils.fetch_prerelease_assets import (
    fetch_prerelease_assets,
)


def s3_download_patch(success: bool = True) -> mock._patch[object]:
    """Patches s3_download to either do nothing or raise an error."""

    def _s3_download(*args: Any, **kwargs: Any) -> None:
        if not success:
            raise ValueError("TEST: BAD DOWNLOAD!")

    return mock.patch(
        "qai_hub_models.scorecard.utils.fetch_prerelease_assets.s3_download",
        _s3_download,
    )


def s3_file_exists_patch(exists: bool = True) -> mock._patch[object]:
    """Patches s3_file_exists to return the given exists boolean value."""

    def _s3_file_exists(*args: Any, **kwargs: Any) -> bool:
        return exists

    return mock.patch(
        "qai_hub_models.scorecard.utils.fetch_prerelease_assets.s3_file_exists",
        _s3_file_exists,
    )


def get_qaihm_s3_patch() -> mock._patch:
    """Patches get_qaihm_s3 to return a mock bucket."""
    mock_bucket = mock.MagicMock()
    return mock.patch(
        "qai_hub_models.scorecard.utils.fetch_prerelease_assets.get_qaihm_s3",
        return_value=(mock_bucket, None),
    )


def release_assets_from_model_patch(
    assets: QAIHMModelReleaseAssets | None = None,
) -> mock._patch[object]:
    """Patches QAIHMModelReleaseAssets.from_model to return the given assets object."""
    if assets is None:
        assets = QAIHMModelReleaseAssets()

    def _from_model(
        model_id: str, not_exists_ok: bool = False
    ) -> QAIHMModelReleaseAssets:
        return assets

    return mock.patch(
        "qai_hub_models.scorecard.utils.fetch_prerelease_assets.QAIHMModelReleaseAssets.from_model",
        _from_model,
    )


class TestFetchPrereleaseAssetsSuccess:
    """Tests for successful S3 asset fetching."""

    def test_fetch_universal_asset_success(self) -> None:
        """Test successful fetch of a universal asset (non-AOT)."""
        assets = QAIHMModelReleaseAssets(
            precisions={
                Precision.float: QAIHMModelReleaseAssets.PrecisionDetails(
                    universal_assets={
                        ScorecardInferencePath.TFLITE: QAIHMModelReleaseAssets.AssetDetails(
                            s3_key="models/mobilenet_v2/v0.44.0/mobilenet_v2_float.tflite"
                        )
                    }
                )
            }
        )
        with (
            release_assets_from_model_patch(assets),
            get_qaihm_s3_patch(),
            s3_file_exists_patch(exists=True),
            s3_download_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path = fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                output_folder=tmpdir,
            )
            # Assets are packaged as .zip files
            assert str(path).endswith(".zip")
            assert tmpdir in str(path)

    def test_fetch_device_specific_asset_success(self) -> None:
        """Test successful fetch of a device-specific (chipset) asset."""
        chipset = "qualcomm-snapdragon-8-gen-3"
        assets = QAIHMModelReleaseAssets(
            precisions={
                Precision.float: QAIHMModelReleaseAssets.PrecisionDetails(
                    chipset_assets={
                        chipset: {
                            ScorecardInferencePath.QNN_CONTEXT_BINARY: QAIHMModelReleaseAssets.AssetDetails(
                                s3_key=f"models/mobilenet_v2/v0.44.0/{chipset}/mobilenet_v2_float.bin"
                            )
                        }
                    }
                )
            }
        )
        with (
            release_assets_from_model_patch(assets),
            get_qaihm_s3_patch(),
            s3_file_exists_patch(exists=True),
            s3_download_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path = fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.QNN_CONTEXT_BINARY,
                Precision.float,
                device_or_chipset=chipset,
                output_folder=tmpdir,
            )
            # Assets are packaged as .zip files
            assert str(path).endswith(".zip")
            assert tmpdir in str(path)

    def test_fetch_with_hub_device(self) -> None:
        """Test fetch with a hub.Device object (converted to chipset)."""
        # Samsung Galaxy S24 uses qualcomm-snapdragon-8gen3 chipset
        chipset = "qualcomm-snapdragon-8gen3"
        assets = QAIHMModelReleaseAssets(
            precisions={
                Precision.float: QAIHMModelReleaseAssets.PrecisionDetails(
                    chipset_assets={
                        chipset: {
                            ScorecardInferencePath.QNN_CONTEXT_BINARY: QAIHMModelReleaseAssets.AssetDetails(
                                s3_key=f"models/mobilenet_v2/v0.44.0/{chipset}/mobilenet_v2_float.bin"
                            )
                        }
                    }
                )
            }
        )
        with (
            release_assets_from_model_patch(assets),
            get_qaihm_s3_patch(),
            s3_file_exists_patch(exists=True),
            s3_download_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            device = hub.Device("Samsung Galaxy S24")
            path = fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.QNN_CONTEXT_BINARY,
                Precision.float,
                device_or_chipset=device,
                output_folder=tmpdir,
            )
            # Assets are packaged as .zip files
            assert str(path).endswith(".zip")
            assert tmpdir in str(path)

    def test_fetch_with_scorecard_profile_path(self) -> None:
        """Test fetch with ScorecardInferencePath instead of TargetRuntime."""
        assets = QAIHMModelReleaseAssets(
            precisions={
                Precision.float: QAIHMModelReleaseAssets.PrecisionDetails(
                    universal_assets={
                        ScorecardInferencePath.TFLITE: QAIHMModelReleaseAssets.AssetDetails(
                            s3_key="models/mobilenet_v2/v0.44.0/mobilenet_v2_float.tflite"
                        )
                    }
                )
            }
        )
        with (
            release_assets_from_model_patch(assets),
            get_qaihm_s3_patch(),
            s3_file_exists_patch(exists=True),
            s3_download_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path = fetch_prerelease_assets(
                "mobilenet_v2",
                ScorecardInferencePath.TFLITE,
                Precision.float,
                output_folder=tmpdir,
            )
            # Assets are packaged as .zip files
            assert str(path).endswith(".zip")


class TestFetchPrereleaseAssetsErrors:
    """Tests for error cases."""

    def test_fetch_aot_without_device_raises_error(self) -> None:
        """Test that fetching AOT asset without device raises ValueError."""
        with pytest.raises(
            ValueError,
            match=r"You must specify a device or chipset",
        ):
            fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.QNN_CONTEXT_BINARY,
                Precision.float,
            )

    def test_fetch_asset_not_found_raises_error(self) -> None:
        """Test that missing asset in release-assets.yaml raises ValueError."""
        assets = QAIHMModelReleaseAssets()  # Empty assets
        with (
            release_assets_from_model_patch(assets),
            pytest.raises(
                ValueError,
                match=r"No pre-release assets found for the specified configuration",
            ),
        ):
            fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
            )

    def test_fetch_s3_file_not_found_raises_error(self) -> None:
        """Test that missing S3 file raises ValueError."""
        assets = QAIHMModelReleaseAssets(
            precisions={
                Precision.float: QAIHMModelReleaseAssets.PrecisionDetails(
                    universal_assets={
                        ScorecardInferencePath.TFLITE: QAIHMModelReleaseAssets.AssetDetails(
                            s3_key="models/mobilenet_v2/v0.44.0/mobilenet_v2_float.tflite"
                        )
                    }
                )
            }
        )
        with (
            release_assets_from_model_patch(assets),
            get_qaihm_s3_patch(),
            s3_file_exists_patch(exists=False),
            pytest.raises(
                ValueError,
                match=r"No pre-release assets found for the specified configuration",
            ),
        ):
            fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
            )


class TestFetchPrereleaseAssetsNoOutputFolder:
    """Tests for fetch without output folder (download to current directory)."""

    def test_fetch_without_output_folder(self) -> None:
        """Test fetch downloads to current directory when output_folder is None."""
        assets = QAIHMModelReleaseAssets(
            precisions={
                Precision.float: QAIHMModelReleaseAssets.PrecisionDetails(
                    universal_assets={
                        ScorecardInferencePath.TFLITE: QAIHMModelReleaseAssets.AssetDetails(
                            s3_key="models/mobilenet_v2/v0.44.0/mobilenet_v2_float.tflite"
                        )
                    }
                )
            }
        )
        with (
            release_assets_from_model_patch(assets),
            get_qaihm_s3_patch(),
            s3_file_exists_patch(exists=True),
            s3_download_patch(success=True),
        ):
            path = fetch_prerelease_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                output_folder=None,
            )
            # When output_folder is None, the path is just the filename
            # Assets are packaged as .zip files
            assert str(path).endswith(".zip")
            assert "/" not in str(path)
