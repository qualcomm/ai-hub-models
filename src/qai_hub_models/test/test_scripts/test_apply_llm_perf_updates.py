# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for scope-based replay of LLM perf updates."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scripts import apply_llm_perf_updates as mod

# A real recipe: the writers read its manifest for the component name, and
# drop/upsert key perf rows by resolved ScorecardDevice.
MODEL = "llama_v3_2_3b_instruct"
DEV_OK = "Snapdragon X Elite CRD"
DEV_FAILED = "Samsung Galaxy S25"


def _metric(device_name: str, path: ScorecardProfilePath) -> dict:
    return {
        "kind": "metric",
        "model_id": MODEL,
        "device_name": device_name,
        "precision": "w4a16",
        "context_length": 4096,
        "tps": 12.5,
        "prefill_tps": 900.0,
        "profile_path": path.value,
        "desired_compute_unit": "npu",
    }


def _scope(device_name: str, path: ScorecardProfilePath) -> dict:
    return {
        "kind": "scope",
        "model_id": MODEL,
        "device_name": device_name,
        "precision": "w4a16",
        "profile_path": path.value,
    }


@pytest.fixture
def perf_dir(tmp_path: Path) -> Iterator[Path]:
    """Point the writers at a throwaway models root holding one perf.yaml."""
    (tmp_path / MODEL).mkdir()
    with (
        mock.patch.object(mod, "QAIHM_MODELS_ROOT", tmp_path),
        mock.patch("qai_hub_models.scorecard.perf_yaml.QAIHM_MODELS_ROOT", tmp_path),
        mock.patch(
            "qai_hub_models.models.templates.llm.perf_collection.QAIHM_MODELS_ROOT",
            tmp_path,
        ),
        mock.patch(
            "qai_hub_models.scorecard.release_assets_yaml.QAIHM_MODELS_ROOT", tmp_path
        ),
        mock.patch.object(mod, "load_similar_devices", return_value={}),
    ):
        yield tmp_path


def _seed(both_devices: bool = True) -> None:
    """Write a committed perf.yaml carrying genie rows for one or two devices."""
    updates = [_metric(DEV_OK, ScorecardProfilePath.GENIE)]
    if both_devices:
        updates.append(_metric(DEV_FAILED, ScorecardProfilePath.GENIE))
    mod.apply_updates(updates)


def _key(reference_device_name: str) -> str:
    """perf.yaml keys by resolved ScorecardDevice, whose name differs from the
    reference device name a perf update carries.
    """
    return ScorecardDevice.get(reference_device_name, return_unregistered=True).name


def _devices_with(path: ScorecardProfilePath) -> set[str]:
    perf = QAIHMModelPerf.from_model(MODEL, not_exists_ok=True)
    out: set[str] = set()
    for precision_details in perf.precisions.values():
        for component in precision_details.components.values():
            for device, metrics in component.performance_metrics.items():
                entry = metrics.get(path)
                if entry is not None and entry.llm_metrics:
                    out.add(device.name)
    return out


def test_failed_device_row_is_removed(perf_dir: Path) -> None:
    """A device in scope that reports no metric loses its committed row."""
    _seed()
    assert _devices_with(ScorecardProfilePath.GENIE) == {_key(DEV_OK), _key(DEV_FAILED)}

    # Re-run measuring both, but only DEV_OK reports back.
    mod.apply_updates(
        [
            _scope(DEV_OK, ScorecardProfilePath.GENIE),
            _scope(DEV_FAILED, ScorecardProfilePath.GENIE),
            _metric(DEV_OK, ScorecardProfilePath.GENIE),
        ]
    )

    assert _devices_with(ScorecardProfilePath.GENIE) == {_key(DEV_OK)}


def test_out_of_scope_device_survives(perf_dir: Path) -> None:
    """A device this run never intended to measure keeps its numbers."""
    _seed()

    mod.apply_updates(
        [
            _scope(DEV_OK, ScorecardProfilePath.GENIE),
            _metric(DEV_OK, ScorecardProfilePath.GENIE),
        ]
    )

    assert _devices_with(ScorecardProfilePath.GENIE) == {_key(DEV_OK), _key(DEV_FAILED)}


def test_other_runtime_is_untouched(perf_dir: Path) -> None:
    """Scope is per profile_path, so a geniex run cannot drop genie rows."""
    _seed(both_devices=False)
    mod.apply_updates([_metric(DEV_OK, ScorecardProfilePath.GENIEX_QAIRT)])
    assert _devices_with(ScorecardProfilePath.GENIE) == {_key(DEV_OK)}
    assert _devices_with(ScorecardProfilePath.GENIEX_QAIRT) == {_key(DEV_OK)}

    # A geniex run where the device fails must leave genie alone.
    mod.apply_updates([_scope(DEV_OK, ScorecardProfilePath.GENIEX_QAIRT)])
    assert _devices_with(ScorecardProfilePath.GENIE) == {_key(DEV_OK)}
    assert _devices_with(ScorecardProfilePath.GENIEX_QAIRT) == set()


def _assets_with(model_id: str, path: ScorecardProfilePath) -> set[str]:
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    out: set[str] = set()
    for prec_details in assets.precisions.values():
        if path in prec_details.universal_assets:
            out.add("<universal>")
        for chipset, paths in prec_details.chipset_assets.items():
            if path in paths:
                out.add(chipset)
    return out


CHIPSETS = {"qualcomm-snapdragon-x-elite", "qualcomm-qcs8275"}


def _write_assets(root: Path, model_id: str) -> None:
    """Seed genie + geniex_qairt + llamacpp assets at w4a16 / q4_0."""
    assets = QAIHMModelReleaseAssets()
    for chipset in sorted(CHIPSETS):
        for path in (
            ScorecardProfilePath.GENIE,
            ScorecardProfilePath.GENIEX_QAIRT,
        ):
            assets.add_asset(
                QAIHMModelReleaseAssets.AssetDetails(s3_key=f"k/{chipset}-{path.name}"),
                precision=Precision.w4a16,
                chipset=chipset,
                path=path,
            )
    assets.add_asset(
        QAIHMModelReleaseAssets.AssetDetails(download_url="https://hf/m.gguf"),
        precision=Precision.q4_0,
        chipset=None,
        path=ScorecardProfilePath.GENIEX_LLAMACPP,
    )
    (root / model_id).mkdir(exist_ok=True)
    assets.to_model_yaml(model_id)


def test_genie_failure_on_default_device_removes_every_asset(perf_dir: Path) -> None:
    """The one X Elite genie job decides all of a precision's release assets."""
    _write_assets(perf_dir, MODEL)
    assert _assets_with(MODEL, ScorecardProfilePath.GENIE) == CHIPSETS
    assert _assets_with(MODEL, ScorecardProfilePath.GENIEX_QAIRT) == CHIPSETS

    mod.apply_updates([_scope(DEV_OK, ScorecardProfilePath.GENIE)])

    assert _assets_with(MODEL, ScorecardProfilePath.GENIE) == set()
    assert _assets_with(MODEL, ScorecardProfilePath.GENIEX_QAIRT) == set()
    # The hand-authored GGUF url is never collateral.
    assert _assets_with(MODEL, ScorecardProfilePath.GENIEX_LLAMACPP) == {"<universal>"}


def test_genie_failure_on_other_device_keeps_assets(perf_dir: Path) -> None:
    """Only the default QDC device triggers the rule."""
    _write_assets(perf_dir, MODEL)
    mod.apply_updates([_scope(DEV_FAILED, ScorecardProfilePath.GENIE)])
    assert _assets_with(MODEL, ScorecardProfilePath.GENIE) == CHIPSETS
    assert _assets_with(MODEL, ScorecardProfilePath.GENIEX_QAIRT) == CHIPSETS


def test_exempt_model_keeps_all_assets(perf_dir: Path) -> None:
    """gemma_4_e4b_it is measured beyond X Elite, so it is exempt."""
    exempt = next(iter(mod.ASSET_REMOVAL_EXEMPT_MODELS))
    _write_assets(perf_dir, exempt)

    update = dict(_scope(DEV_OK, ScorecardProfilePath.GENIE), model_id=exempt)
    mod.apply_updates([update])

    assert _assets_with(exempt, ScorecardProfilePath.GENIE) == CHIPSETS
    assert _assets_with(exempt, ScorecardProfilePath.GENIEX_QAIRT) == CHIPSETS


def _supported(model_id: str) -> tuple[list[str], list[str]]:
    perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
    return perf.supported_chipsets, [str(d) for d in perf.supported_devices]


def test_x_elite_metric_credits_x_plus(perf_dir: Path) -> None:
    """X Plus is the same NPU as X Elite, so measuring X Elite advertises both."""
    mod.apply_updates([_metric(DEV_OK, ScorecardProfilePath.GENIE)])

    chipsets, devices = _supported(MODEL)
    assert "qualcomm-snapdragon-x-elite" in chipsets
    assert "qualcomm-snapdragon-x-plus-8-core" in chipsets
    assert "Snapdragon X Plus 8-Core CRD" in devices


def test_mobile_metric_does_not_backfill_older_chipsets(perf_dir: Path) -> None:
    """Only the compute pairing applies: an 8B LLM must not claim Snapdragon 888."""
    mod.apply_updates([_metric(DEV_FAILED, ScorecardProfilePath.GENIE)])

    chipsets, _ = _supported(MODEL)
    assert "qualcomm-snapdragon-8-elite-for-galaxy" in chipsets
    assert not {
        "qualcomm-snapdragon-888",
        "qualcomm-snapdragon-8gen1",
        "qualcomm-snapdragon-8gen2",
        "qualcomm-snapdragon-8gen3",
    } & set(chipsets)
