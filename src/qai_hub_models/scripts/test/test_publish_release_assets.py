# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from qai_hub_models.scripts import publish_release_assets
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG


def test_publish_latest_mirror_copies_and_deletes_expected_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror step must delete every stale latest/ key and copy every versioned key
    to its translated latest/ location. This catches a prefix-translation bug that
    would otherwise silently point consumers at the wrong version.
    """
    monkeypatch.setattr(
        publish_release_assets, "attempt_with_s3_credentials_warning", lambda fn: fn()
    )
    monkeypatch.setattr(
        ASSET_CONFIG,
        "get_global_release_s3_folder",
        lambda version: "qai-hub-models/releases/v0.61.0/",
    )
    s3_copy_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "s3_copy", s3_copy_mock)

    bucket = MagicMock()
    bucket.name = "qaihub-public-assets"
    bucket.Object.return_value.get.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey"}}, "GetObject"
    )
    stale = [MagicMock(key="qai-hub-models/releases/latest/manifest.json")]
    source = [
        MagicMock(key="qai-hub-models/releases/v0.61.0/manifest.json"),
        MagicMock(key="qai-hub-models/releases/v0.61.0/models/foo/info.json"),
    ]
    bucket.objects.filter.side_effect = lambda Prefix: (
        stale if Prefix == publish_release_assets.LATEST_MIRROR_PREFIX else source
    )

    publish_release_assets.publish_latest_mirror(bucket, "v0.61.0")

    bucket.delete_objects.assert_called_once_with(
        Delete={"Objects": [{"Key": "qai-hub-models/releases/latest/manifest.json"}]}
    )
    copied = {c.kwargs["dst_key"] for c in s3_copy_mock.call_args_list}
    assert copied == {
        "qai-hub-models/releases/latest/manifest.json",
        "qai-hub-models/releases/latest/models/foo/info.json",
    }


def test_publish_latest_mirror_skips_when_existing_is_newer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If latest/manifest.json already points at a newer release (e.g. a bugfix
    re-release of 0.60.x while 0.61.0 is live), the mirror step must be a no-op.
    Otherwise the older version would clobber the newer one under latest/.
    """
    monkeypatch.setattr(
        publish_release_assets, "attempt_with_s3_credentials_warning", lambda fn: fn()
    )
    s3_copy_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "s3_copy", s3_copy_mock)

    bucket = MagicMock()
    bucket.Object.return_value.get.return_value = {
        "Body": MagicMock(read=lambda: json.dumps({"version": "0.62.0"}).encode())
    }

    publish_release_assets.publish_latest_mirror(bucket, "v0.61.0")

    s3_copy_mock.assert_not_called()
    bucket.delete_objects.assert_not_called()


def test_release_asset_skips_write_when_target_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary code-side guard for release archives: never s3_copy over an existing versioned
    key unless --overwrite is explicitly set.
    """
    s3_copy_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "s3_copy", s3_copy_mock)
    monkeypatch.setattr(publish_release_assets, "s3_file_exists", lambda *a, **k: True)
    monkeypatch.setattr(
        ASSET_CONFIG,
        "get_release_asset_s3_key",
        lambda **_: "qai-hub-models/releases/v0.61.0/foo",
    )
    sc_path = MagicMock()
    sc_path.is_published = True
    asset_details = MagicMock()
    asset_details.s3_key = "private/foo"
    asset_details.download_url = None

    publish_release_assets.release_asset(
        s3_private_bucket=MagicMock(),
        s3_public_bucket=MagicMock(),
        qaihm_version="v0.61.0",
        model_id="fake_model",
        precision=MagicMock(),
        chipset=None,
        sc_path=sc_path,
        asset_details=asset_details,
        replace_existing=False,
    )

    s3_copy_mock.assert_not_called()


def test_overwrite_flag_requires_envvar(monkeypatch: pytest.MonkeyPatch) -> None:
    """--overwrite must SystemExit before touching S3 when the envvar isn't set.
    Guarantees CI (which never sets the envvar) can't overwrite even if the flag leaks in.
    """
    monkeypatch.delenv(publish_release_assets.OVERWRITE_ASSETS_ENVVAR, raising=False)
    monkeypatch.setattr(
        publish_release_assets,
        "parse_args",
        lambda: argparse.Namespace(models=None, overwrite=True, version="v0.61.0"),
    )
    monkeypatch.setattr(
        publish_release_assets,
        "validate_and_split_enabled_models",
        lambda models: ({"fake_model"}, set()),
    )
    get_qaihm_s3_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "get_qaihm_s3", get_qaihm_s3_mock)

    with pytest.raises(SystemExit):
        publish_release_assets.main()
    get_qaihm_s3_mock.assert_not_called()


def _release_asset(
    monkeypatch: pytest.MonkeyPatch,
    private_key: str,
    existing_keys: set[tuple[str, str]],
    public_key: str = "qai-hub-models/models/fake_model/releases/v0.67.0/m-tflite-float.zip",
) -> MagicMock:
    s3_copy_mock = MagicMock()
    monkeypatch.setattr(publish_release_assets, "s3_copy", s3_copy_mock)
    monkeypatch.setattr(
        publish_release_assets,
        "s3_file_exists",
        lambda bucket, key: (bucket.name, key) in existing_keys,
    )
    monkeypatch.setattr(
        ASSET_CONFIG, "get_release_asset_s3_key", lambda **_: public_key
    )
    sc_path = MagicMock()
    sc_path.is_published = True
    asset_details = MagicMock()
    asset_details.s3_key = private_key
    asset_details.download_url = None
    private, public = MagicMock(), MagicMock()
    private.name, public.name = "private", "public"

    publish_release_assets.release_asset(
        s3_private_bucket=private,
        s3_public_bucket=public,
        qaihm_version="v0.67.0",
        model_id="fake_model",
        precision=MagicMock(),
        chipset=None,
        sc_path=sc_path,
        asset_details=asset_details,
        replace_existing=False,
    )
    return s3_copy_mock


def test_release_asset_copies_metadata_beside_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_zip = "artifacts/run1/m-tflite-float.zip"
    s3_copy_mock = _release_asset(
        monkeypatch,
        private_zip,
        existing_keys={
            ("private", "artifacts/run1/m-tflite-float_metadata.proto"),
            ("private", "artifacts/run1/m-tflite-float_metadata.json"),
        },
    )

    copied = {
        c.kwargs["src_key"]: c.kwargs["dst_key"] for c in s3_copy_mock.call_args_list
    }
    base = "qai-hub-models/models/fake_model/releases/v0.67.0/m-tflite-float"
    assert copied == {
        private_zip: f"{base}.zip",
        "artifacts/run1/m-tflite-float_metadata.proto": f"{base}_metadata.proto",
        "artifacts/run1/m-tflite-float_metadata.json": f"{base}_metadata.json",
    }
    assert all(c.kwargs["make_dst_public"] for c in s3_copy_mock.call_args_list)
