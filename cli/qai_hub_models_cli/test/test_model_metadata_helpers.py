# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.errors import UnsupportedVersionError
from qai_hub_models_cli.proto.model_metadata_pb2 import ModelMetadata
from qai_hub_models_cli.proto_helpers._common import write_proto, write_proto_json
from qai_hub_models_cli.proto_helpers.model_metadata import (
    get_model_metadata,
    metadata_url_for_asset,
    read_model_metadata,
)

_MODULE = "qai_hub_models_cli.proto_helpers.model_metadata"
_ASSET_URL = "https://store/models/m/releases/v0.67.0/m-tflite-float.zip"
_METADATA_URL = "https://store/models/m/releases/v0.67.0/m-tflite-float_metadata.proto"


def _metadata() -> ModelMetadata:
    return ModelMetadata(model_id="m", model_name="M")


def test_metadata_url_for_asset() -> None:
    assert metadata_url_for_asset(_ASSET_URL) == _METADATA_URL
    assert metadata_url_for_asset("a/m.tar.gz") == "a/m_metadata.proto"
    assert metadata_url_for_asset("a/m.bin") == "a/m_metadata.proto"


def test_get_model_metadata_downloads_beside_asset(tmp_path: Path) -> None:
    def _download(url: str, dst: Path, **kwargs: object) -> Path:
        write_proto(Path(dst), _metadata())
        return Path(dst)

    download = MagicMock(side_effect=_download)
    with (
        patch(f"{_MODULE}.get_asset_url", return_value=_ASSET_URL),
        patch(f"{_MODULE}.get_manifest_entry", return_value=MagicMock(id="m")),
        patch(f"{_MODULE}.model_cache_path", return_value=tmp_path / "m.proto"),
        patch("qai_hub_models_cli.proto_helpers._common.download", download),
    ):
        result = get_model_metadata("m", "tflite", "float", version=Version("0.67.0"))

    assert result == _metadata()
    assert download.call_args.args[0] == _METADATA_URL


def test_get_model_metadata_rejects_old_versions() -> None:
    with pytest.raises(UnsupportedVersionError, match="not uploaded separately"):
        get_model_metadata("m", "tflite", "float", version=Version("0.66.0"))


def test_read_model_metadata(tmp_path: Path) -> None:
    json_path = tmp_path / "metadata.json"
    write_proto_json(_metadata(), json_path)
    proto_path = tmp_path / "m_metadata.proto"
    write_proto(proto_path, _metadata())

    assert read_model_metadata(json_path) == _metadata()
    assert read_model_metadata(proto_path) == _metadata()
