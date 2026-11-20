# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

from packaging.version import Version

from qai_hub_models_cli.proto.model_release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.release_manifest_pb2 import ReleaseManifest
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime

_RELEASE_VERSION = Version("0.52.0")  # >= MIN_MANIFEST_VERSION (manifest path)


def _s3_release_assets() -> ModelReleaseAssets:
    return ModelReleaseAssets(
        aihm_version="0.50.1",
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_TFLITE,
                download_url="s3://qai-hub-models-private-assets/models/mobilenet_v2/releases/v0.50.1/mobilenet_v2-tflite-float.zip",
            ),
        ],
    )


def test_download_s3_url_extracts_bucket_and_key(tmp_path: Path) -> None:
    """download() parses s3://bucket/key and calls s3_download with correct args."""
    from qai_hub_models_cli.utils import download

    dst = tmp_path / "model.zip"
    with patch("qai_hub_models_cli._internal.aws.s3_download") as mock_s3:
        mock_s3.side_effect = lambda key, path, **kw: path.write_bytes(b"data")
        result = download("s3://my-bucket/path/to/model.zip", dst)

    mock_s3.assert_called_once()
    call_args = mock_s3.call_args
    assert call_args[0][0] == "path/to/model.zip"
    assert call_args[1]["bucket_name"] == "my-bucket"
    assert result == dst


def test_fetch_proto_s3_url(tmp_path: Path) -> None:
    """fetch_proto routes s3:// URLs through download to s3_download."""
    from qai_hub_models_cli.proto_helpers._common import fetch_proto

    manifest = ReleaseManifest(version="0.50.1", models=[])
    cache_path = tmp_path / "manifest.pb"

    with patch("qai_hub_models_cli._internal.aws.s3_download") as mock_s3:
        mock_s3.side_effect = lambda key, path, **kw: path.write_bytes(
            manifest.SerializeToString()
        )
        result = fetch_proto(
            "s3://qai-hub-models-private-assets/releases/v0.50.1/manifest.pb",
            cache_path,
            ReleaseManifest,
        )

    mock_s3.assert_called_once()
    call_args = mock_s3.call_args
    assert call_args[0][0] == "releases/v0.50.1/manifest.pb"
    assert call_args[1]["bucket_name"] == "qai-hub-models-private-assets"
    assert result.version == "0.50.1"


def test_get_asset_url_returns_s3_url_from_internal_proto() -> None:
    """When the proto download_url is s3://, get_asset_url returns it as-is."""
    from qai_hub_models_cli.fetch import get_asset_url
    from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo, RuntimeInfo

    s3_url = "s3://qai-hub-models-private-assets/models/mobilenet_v2/releases/v0.50.1/mobilenet_v2-tflite-float.zip"

    with (
        patch(
            "qai_hub_models_cli.fetch.get_model_release_assets",
            return_value=_s3_release_assets(),
        ),
        patch(
            "qai_hub_models_cli.fetch.get_platform",
            return_value=PlatformInfo(),
        ),
        patch(
            "qai_hub_models_cli.fetch.resolve_runtime",
            return_value=RuntimeInfo(is_aot_compiled=False),
        ),
    ):
        url = get_asset_url(
            model="mobilenet_v2",
            runtime="tflite",
            precision="float",
            version=_RELEASE_VERSION,
        )

    assert url == s3_url


@patch("qai_hub_models_cli.fetch.download")
@patch("qai_hub_models_cli.fetch.get_asset_url")
def test_fetch_end_to_end_s3(
    mock_get_url: MagicMock, mock_download: MagicMock, tmp_path: Path
) -> None:
    """fetch() passes s3:// URL from get_asset_url through to download()."""
    from qai_hub_models_cli.fetch import fetch

    s3_url = "s3://qai-hub-models-private-assets/models/mobilenet_v2/releases/v0.50.1/mobilenet_v2-tflite-float.zip"
    mock_get_url.return_value = s3_url
    mock_download.return_value = tmp_path / "mobilenet_v2-tflite-float.zip"

    fetch(
        model="mobilenet_v2",
        runtime="tflite",
        output_dir=tmp_path,
        version=_RELEASE_VERSION,
    )

    mock_download.assert_called_once()
    assert mock_download.call_args[0][0] == s3_url


def _subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices.keys())
    return set()


def test_validate_aws_subcommand_hidden_when_aws_extra_missing() -> None:
    """Gate closes when the [internal] extra is not installed."""
    from qai_hub_models_cli.cli import _build_parser

    with patch("qai_hub_models_cli.cli._AWS_EXTRA_AVAILABLE", False):
        parser = _build_parser()

    assert "validate_aws_credentials" not in _subcommand_names(parser)


def test_internal_cache_dir_separate_from_public() -> None:
    """Internal and public registries use different cache directories."""
    from qai_hub_models_cli.common import CACHE_DIR
    from qai_hub_models_cli.proto_helpers._common import get_release_cache_dir

    # Compare the path segments each adds under the (temp) cache root, not the
    # absolute path -- the temp dir name itself may contain "internal".
    public = get_release_cache_dir(Version("0.50.1"), internal=False).relative_to(
        CACHE_DIR
    )
    internal = get_release_cache_dir(Version("0.50.1"), internal=True).relative_to(
        CACHE_DIR
    )
    assert public != internal
    assert "internal" not in str(public)
    assert "internal" in str(internal)
