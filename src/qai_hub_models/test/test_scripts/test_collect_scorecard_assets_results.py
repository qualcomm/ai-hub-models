# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import ScorecardInferencePath
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.yaml import ScorecardAssetYaml
from qai_hub_models.scripts import collect_scorecard_assets_results as mod


def _asset(
    s3_key: str | None = None, download_url: str | None = None
) -> QAIHMModelReleaseAssets.AssetDetails:
    return QAIHMModelReleaseAssets.AssetDetails(
        s3_key=s3_key, download_url=download_url
    )


def _make_assets(entries: dict) -> QAIHMModelReleaseAssets:
    out = QAIHMModelReleaseAssets()
    for precision, kinds in entries.items():
        for path, asset in kinds.get("universal", {}).items():
            out.add_asset(asset, precision=precision, chipset=None, path=path)
        for chipset, path_map in kinds.get("chipset", {}).items():
            for path, asset in path_map.items():
                out.add_asset(asset, precision=precision, chipset=chipset, path=path)
    return out


def test_scoped_merge_preserves_out_of_scope_committed_entry() -> None:
    """A run scoped to (w4a16, snapdragon-8-elite, GENIE) must not touch a
    committed q4_0/GENIEX_LLAMACPP entry that was outside its scope.
    """
    committed = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardInferencePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/model.gguf"
                    ),
                },
            },
        }
    )
    scorecard = _make_assets(
        {
            Precision.w4a16: {
                "chipset": {
                    "qualcomm-snapdragon-8-elite": {
                        ScorecardInferencePath.GENIE: _asset(s3_key="s3/genie.zip"),
                    }
                }
            }
        }
    )
    scope: set[tuple[Precision, str | None, ScorecardInferencePath]] = {
        (
            Precision.w4a16,
            "qualcomm-snapdragon-8-elite",
            ScorecardInferencePath.GENIE,
        )
    }

    committed.drop_entries_in_scope(scope)
    committed.merge_from(scorecard)

    kept = committed.get_asset(
        Precision.q4_0, chipset=None, path=ScorecardInferencePath.GENIEX_LLAMACPP
    )
    assert kept is not None
    assert kept.download_url == "https://hf/model.gguf"
    assert kept.s3_key is None

    fresh = committed.get_asset(
        Precision.w4a16,
        chipset="qualcomm-snapdragon-8-elite",
        path=ScorecardInferencePath.GENIE,
    )
    assert fresh is not None and fresh.s3_key == "s3/genie.zip"


def test_scoped_merge_scorecard_wins_in_scope() -> None:
    """When the scorecard produces a fresh entry for a tuple in scope, it
    replaces the committed one after the drop+merge.
    """
    committed = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardInferencePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/old.gguf"
                    ),
                },
            },
        }
    )
    scorecard = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardInferencePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/new.gguf"
                    ),
                },
            },
        }
    )
    scope: set[tuple[Precision, str | None, ScorecardInferencePath]] = {
        (Precision.q4_0, None, ScorecardInferencePath.GENIEX_LLAMACPP)
    }

    committed.drop_entries_in_scope(scope)
    committed.merge_from(scorecard)

    kept = committed.get_asset(
        Precision.q4_0, chipset=None, path=ScorecardInferencePath.GENIEX_LLAMACPP
    )
    assert kept is not None
    assert kept.download_url == "https://hf/new.gguf"


def test_scoped_merge_drops_in_scope_when_run_produced_nothing() -> None:
    """An in-scope tuple with no replacement in the scorecard is removed --
    matches the prior overwrite semantics inside scope.
    """
    committed = _make_assets(
        {
            Precision.w4a16: {
                "chipset": {
                    "cs": {
                        ScorecardInferencePath.GENIE: _asset(s3_key="s3/old.zip"),
                    }
                }
            }
        }
    )
    scope: set[tuple[Precision, str | None, ScorecardInferencePath]] = {
        (Precision.w4a16, "cs", ScorecardInferencePath.GENIE)
    }

    committed.drop_entries_in_scope(scope)
    committed.merge_from(QAIHMModelReleaseAssets())

    assert committed.empty


@pytest.mark.parametrize("contents", ["", "models: {}\n"])
def test_empty_release_assets_yaml_updates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
    """An empty assets yaml means "no assets this run", not "drop everything".

    combine_split_artifacts always emits release-assets.yaml, so `models: {}`
    reaches this script whenever a sweep produced no export assets. Treating it
    as a result set would strip in-scope committed entries and commit that.
    """
    assets_yaml = tmp_path / "release-assets.yaml"
    assets_yaml.write_text(contents)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_scorecard_assets_results",
            "--models",
            "qwen3_0_6b",
            "--release-assets-yaml",
            str(assets_yaml),
        ],
    )

    with (
        mock.patch.object(QAIHMModelReleaseAssets, "drop_entries_in_scope") as drop,
        mock.patch.object(QAIHMModelReleaseAssets, "to_model_yaml") as write,
    ):
        mod.main()

    drop.assert_not_called()
    write.assert_not_called()


def test_committed_genie_assets_stripped_when_geniex_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Genie is outside the default sweep, so committed Genie assets survive the
    scoped drop; a geniex_qairt failure must still strip them (gemma_4_e4b_it).
    """
    committed = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardInferencePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/model.gguf"
                    ),
                },
            },
            Precision.w4a16: {
                "chipset": {
                    "qualcomm-snapdragon-x-elite": {
                        ScorecardInferencePath.GENIE: _asset(s3_key="s3/genie.zip"),
                    }
                }
            },
        }
    )
    manifest = QAIHMModelManifest.from_model("gemma_4_e4b_it")
    manifest.disabled_paths.get_disable_reasons(
        Precision.w4a16, TargetRuntime.GENIEX_QAIRT
    ).scorecard_accuracy_failure = "too low"

    assets_yaml = tmp_path / "release-assets.yaml"
    ScorecardAssetYaml(models={"gemma_4_e4b_it": QAIHMModelReleaseAssets()}).to_yaml(
        assets_yaml
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_scorecard_assets_results",
            "--models",
            "gemma_4_e4b_it",
            "--release-assets-yaml",
            str(assets_yaml),
        ],
    )
    written: list[QAIHMModelReleaseAssets] = []

    def _capture(self: QAIHMModelReleaseAssets, _model_id: str) -> Path:
        written.append(self)
        return tmp_path / "out.yaml"

    monkeypatch.setattr(QAIHMModelManifest, "from_model", lambda _id: manifest)
    monkeypatch.setattr(
        QAIHMModelReleaseAssets, "from_model", lambda *_a, **_k: committed
    )
    monkeypatch.setattr(QAIHMModelReleaseAssets, "to_model_yaml", _capture)
    monkeypatch.setattr(mod, "_release_assets_scope", lambda *_a: set())
    monkeypatch.setattr(mod, "update_model_publish_status", lambda _m: False)

    mod.main()

    assert len(written) == 1
    assert set(written[0].precisions) == {Precision.q4_0}
