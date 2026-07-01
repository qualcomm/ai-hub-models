# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import pytest
import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.utils.fetch_static_assets import fetch_static_assets


def requests_head_patch(status_code: int = 200) -> mock._patch[object]:
    """Patches requests.head to return a response with the given status code."""

    def _head(url: str, *args: Any, **kwargs: Any) -> mock.MagicMock:
        response = mock.MagicMock()
        response.status_code = status_code
        return response

    return mock.patch("qai_hub_models.utils.fetch_static_assets.requests.head", _head)


def download_file_patch(success: bool = True) -> mock._patch[object]:
    """Patches download_file to return the destination path without downloading."""

    def _download_file(url: str, dst_path: str, *args: Any, **kwargs: Any) -> str:
        if not success:
            raise ValueError("TESTING: Download failed")
        return dst_path

    return mock.patch(
        "qai_hub_models.utils.fetch_static_assets.download_file", _download_file
    )


def version_patch(version: str) -> mock._patch[object]:
    """Patches __version__ in fetch_static_assets to simulate release or dev installs."""
    return mock.patch("qai_hub_models.utils.fetch_static_assets.__version__", version)


def fetch_prerelease_assets_patch(
    success: bool = True, return_path: str | None = None
) -> mock._patch[object]:
    """Patches fetch_prerelease_assets to return a path or raise an error."""
    from pathlib import Path

    def _fetch_prerelease_assets(
        model_id: str,
        runtime: TargetRuntime,
        precision: Precision,
        device_or_chipset: Any,
        output_folder: Any,
        asset_config: Any,
        verbose: bool = True,
    ) -> Path:
        if not success:
            raise ValueError(
                "No pre-release assets found for the specified configuration."
            )
        return Path(return_path or "/mock/prerelease/asset.tflite")

    return mock.patch(
        "qai_hub_models.utils.fetch_static_assets.fetch_prerelease_assets",
        _fetch_prerelease_assets,
    )


class TestFetchStaticAssetsPublicAsset:
    """Tests for fetching assets from public release URLs."""

    def test_fetch_universal_asset_success(self) -> None:
        """Test successful fetch of a universal asset (non-AOT)."""
        with (
            version_patch("0.45.0"),
            requests_head_patch(200),
            download_file_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                output_folder=tmpdir,
            )
            assert path is not None
            assert path.is_relative_to(Path(tmpdir))
            # Assets are packaged as .zip files
            assert path.suffix == ".zip"
            assert "mobilenet_v2" in url
            assert url.endswith(".zip")

    def test_fetch_device_specific_asset_success(self) -> None:
        """Test successful fetch of a device-specific asset."""
        with (
            version_patch("0.45.0"),
            requests_head_patch(200),
            download_file_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.PRECOMPILED_QNN_ONNX,
                Precision.float,
                device_or_chipset="qualcomm-snapdragon-8-gen-3",
                output_folder=tmpdir,
            )
            assert path is not None
            assert path.is_relative_to(Path(tmpdir))
            assert "mobilenet_v2" in url
            assert "qualcomm_snapdragon_8_gen_3" in url

    def test_fetch_device_specific_asset_with_hub_device(self) -> None:
        """Test fetch with a hub.Device object."""

        def _load_chipsets_from_previous_release(
            *args: Any, **kwargs: Any
        ) -> DevicesAndChipsetsYaml:
            return DevicesAndChipsetsYaml.load()

        with (
            version_patch("0.45.0"),
            requests_head_patch(200),
            download_file_patch(success=True),
            mock.patch(
                "qai_hub_models.utils.fetch_static_assets._load_chipsets_from_previous_release",
                _load_chipsets_from_previous_release,
            ),
            TemporaryDirectory() as tmpdir,
        ):
            device = hub.Device("Samsung Galaxy S24")
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.PRECOMPILED_QNN_ONNX,
                Precision.float,
                device_or_chipset=device,
                output_folder=tmpdir,
            )
            assert path is not None
            assert path.is_relative_to(Path(tmpdir))
            assert "mobilenet_v2" in url

    def test_fetch_aot_without_device_raises_error(self) -> None:
        """Test that fetching AOT asset without device raises ValueError."""
        with (
            version_patch("0.45.0"),
            pytest.raises(
                ValueError,
                match=r"You must specify a device or chipset",
            ),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.PRECOMPILED_QNN_ONNX,
                Precision.float,
            )


class TestFetchStaticAssetsDevVersion:
    """Tests for prerelease asset fetching (dev version installed)."""

    def test_prerelease_used_when_dev_version(self) -> None:
        """Test that fetch_prerelease_assets is used when __version__ contains 'dev'."""
        with (
            version_patch("0.45.0.dev123"),
            fetch_prerelease_assets_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                qaihm_version_tag=None,
                output_folder=tmpdir,
            )
            assert str(path) == "/mock/prerelease/asset.tflite"
            # URL is empty when using prerelease assets
            assert url == ""

    def test_prerelease_used_even_with_specific_version_tag(self) -> None:
        """Test that prerelease is used when __version__ is dev, even with a specific version tag."""
        with (
            version_patch("0.45.0.dev123"),
            fetch_prerelease_assets_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                qaihm_version_tag="v0.44.0",
                output_folder=tmpdir,
            )
            assert str(path) == "/mock/prerelease/asset.tflite"
            assert url == ""

    def test_public_release_used_when_release_version(self) -> None:
        """Test that public release is used when __version__ does not contain 'dev'."""
        with (
            version_patch("0.45.0"),
            requests_head_patch(200),
            download_file_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                qaihm_version_tag=None,
                output_folder=tmpdir,
            )
            assert path is not None
            assert str(path).startswith(tmpdir)
            assert "mobilenet_v2" in url

    def test_public_release_with_specific_version(self) -> None:
        """Test that public release uses the specified version tag when not dev."""
        with (
            version_patch("0.45.0"),
            requests_head_patch(200),
            download_file_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                qaihm_version_tag="v0.44.0",
                output_folder=tmpdir,
            )
            assert path is not None
            assert str(path).startswith(tmpdir)
            assert "v0.44.0" in url

    def test_error_when_public_release_not_found(self) -> None:
        """Test error when public release returns 404."""
        with (
            version_patch("0.45.0"),
            requests_head_patch(404),
            pytest.raises(FileNotFoundError, match=r"No asset found"),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
            )


class TestFetchStaticAssetsVersionRequirements:
    """Tests for version requirements."""

    def test_old_version_raises_error(self) -> None:
        """Test that fetching from old QAIHM version (< 0.44.0) raises error."""
        with (
            version_patch("0.45.0"),
            pytest.raises(
                ValueError,
                match=r"Fetching device-specific assets is not supported for QAIHM versions < v0.44.0",
            ),
        ):
            fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                qaihm_version_tag="v0.43.0",
            )

    def test_valid_version_allowed(self) -> None:
        """Test that valid QAIHM versions (>= 0.44.0) are allowed."""
        with (
            version_patch("0.45.0"),
            requests_head_patch(200),
            download_file_patch(success=True),
            TemporaryDirectory() as tmpdir,
        ):
            path, url = fetch_static_assets(
                "mobilenet_v2",
                TargetRuntime.TFLITE,
                Precision.float,
                qaihm_version_tag="v0.44.0",
                output_folder=tmpdir,
            )
            assert path is not None
            assert "v0.44.0" in url
            assert "mobilenet_v2" in url
