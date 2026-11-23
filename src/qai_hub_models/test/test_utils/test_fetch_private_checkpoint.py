# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for fetch_default_checkpoint when QAIHM_TEST_CHECKPOINT is set."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from qai_hub_models import Precision
from qai_hub_models.models.templates.llm import model as llm_model
from qai_hub_models.models.templates.llm.model import DynamicQuantizablePreSplitMixin
from qai_hub_models.utils.asset_loaders import ModelZooAssetConfig
from qai_hub_models.utils.aws import QAIHM_PRIVATE_S3_BUCKET
from qai_hub_models.utils.envvars import CheckpointEnvvar

KEY_PREFIX = "ephemeral_quantize_assets/gh_actions"


class _Model(DynamicQuantizablePreSplitMixin):
    model_id = "fake_model"
    model_asset_version = 5
    default_checkpoint = {Precision.w4a16: "fake_model_w4a16"}


class _FakePrivateBucket:
    """Serves bytes by key, in place of the private S3 bucket."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> str:
        self.objects[key] = data
        return key

    def download(self, url: str, dst: str | Path, *_: object, **__: object) -> Path:
        key = url.removeprefix(f"s3://{QAIHM_PRIVATE_S3_BUCKET}/")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(self.objects[key])
        return Path(dst)


@pytest.fixture
def bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakePrivateBucket:
    """A fake private bucket, with the local asset cache redirected to tmp_path."""
    config = ModelZooAssetConfig.from_cfg()
    monkeypatch.setattr(config, "local_store_path", tmp_path / "store")
    monkeypatch.setattr(llm_model, "ASSET_CONFIG", config)
    monkeypatch.setattr(
        "qai_hub_models.utils.private_asset_loaders.can_access_private_s3",
        lambda: True,
    )
    fake = _FakePrivateBucket()
    monkeypatch.setattr(
        "qai_hub_models.utils.private_asset_loaders.download_from_private_s3",
        fake.download,
    )
    return fake


def _fetch(monkeypatch: pytest.MonkeyPatch, key: str) -> Path:
    monkeypatch.setenv(CheckpointEnvvar.VARNAME, key)
    return Path(_Model.fetch_default_checkpoint(Precision.w4a16))


def test_bare_encodings_key_becomes_a_checkpoint_dir(
    monkeypatch: pytest.MonkeyPatch, bucket: _FakePrivateBucket
) -> None:
    key = bucket.put(f"{KEY_PREFIX}/111/model.encodings", b"enc-111")

    out = _fetch(monkeypatch, key)

    assert out.is_dir()
    assert [p.name for p in out.iterdir()] == ["model.encodings"]
    assert (out / "model.encodings").read_bytes() == b"enc-111"


def test_two_runs_do_not_share_a_cached_encodings(
    monkeypatch: pytest.MonkeyPatch, bucket: _FakePrivateBucket
) -> None:
    first = _fetch(monkeypatch, bucket.put(f"{KEY_PREFIX}/111/model.encodings", b"a"))
    second = _fetch(monkeypatch, bucket.put(f"{KEY_PREFIX}/222/model.encodings", b"b"))

    assert first != second
    assert (first / "model.encodings").read_bytes() == b"a"
    assert (second / "model.encodings").read_bytes() == b"b"


def test_zip_key_is_still_extracted(
    monkeypatch: pytest.MonkeyPatch, bucket: _FakePrivateBucket, tmp_path: Path
) -> None:
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("model.encodings", "from-zip")
        z.writestr("config.json", "{}")
    key = bucket.put(f"{KEY_PREFIX}/333/fake_model-w4a16.zip", archive.read_bytes())

    out = _fetch(monkeypatch, key)

    assert (out / "model.encodings").read_text() == "from-zip"
    assert (out / "config.json").exists()
