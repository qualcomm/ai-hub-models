# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import pytest
from qai_hub_models_cli.proto import perf_pb2

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import ScorecardDevice, cs_x_elite
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf

# whisper_tiny ships per-chipset context binaries, so the filter applies.
AOT_MODEL = "whisper_tiny"
# act ships universal assets, so the filter must leave its card alone.
JIT_MODEL = "act"


def test_drop_chipsets_without_assets_drops_unbacked() -> None:
    """A claim no published asset backs must not survive."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    perf.supported_chipsets.append("qualcomm-snapdragon-888")
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert "qualcomm-snapdragon-888" not in perf.supported_chipsets


def test_drop_chipsets_without_assets_keeps_backed() -> None:
    """A chipset the model ships an asset for is kept even if unprofiled."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    before = set(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert "qualcomm-snapdragon-8gen1" in perf.supported_chipsets
    assert set(perf.supported_chipsets) <= before


def test_drop_chipsets_without_assets_does_not_add() -> None:
    """The filter only subtracts; it never introduces a new claim."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    before = set(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert not set(perf.supported_chipsets) - before


def test_drop_chipsets_without_assets_rewrites_devices() -> None:
    """supported_devices is derived from the kept chipsets, not left stale."""
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    perf.supported_chipsets.append("qualcomm-snapdragon-888")
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert not any("Galaxy S21" in d for d in perf.supported_devices)


def test_drop_chipsets_without_assets_skips_universal_assets() -> None:
    """A universal asset runs anywhere, so its card must be untouched."""
    perf = QAIHMModelPerf.from_model(JIT_MODEL)
    before = list(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(JIT_MODEL)
    assert list(perf.supported_chipsets) == before


def test_drop_chipsets_without_assets_is_idempotent() -> None:
    perf = QAIHMModelPerf.from_model(AOT_MODEL)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    once = list(perf.supported_chipsets)
    perf.drop_chipsets_without_assets(AOT_MODEL)
    assert list(perf.supported_chipsets) == once


MIXED_PATHS = [
    ScorecardProfilePath.TFLITE,
    ScorecardProfilePath.QNN_DLC,
    ScorecardProfilePath.QNN_CONTEXT_BINARY,
    ScorecardProfilePath.PRECOMPILED_QNN_ONNX,
    ScorecardProfilePath.VOICE_AI,
]


def _mixed_perf(*devices: ScorecardDevice) -> QAIHMModelPerf:
    details = perf_pb2.ModelPerf.PerformanceDetails(
        profile_job=perf_pb2.ModelPerf.ProfileJob(id="jabcd1234")
    )
    perf = QAIHMModelPerf("mixed")
    for name in ("A", "B"):
        for device in devices or (cs_x_elite,):
            for path in MIXED_PATHS:
                key = QAIHMModelPerf.Key(Precision.float, name, device, path)
                perf.update_entry(key.stamp(details))
    return perf


def _paths(perf: QAIHMModelPerf, component: str = "A") -> set[ScorecardProfilePath]:
    return {key.path for key in perf.entries if key.component == component}


@pytest.mark.parametrize(
    ("requires_aot_prepare", "kept"),
    [
        (
            False,
            {
                ScorecardProfilePath.TFLITE,
                ScorecardProfilePath.QNN_DLC,
                ScorecardProfilePath.VOICE_AI,
            },
        ),
        (
            True,
            {
                ScorecardProfilePath.QNN_CONTEXT_BINARY,
                ScorecardProfilePath.PRECOMPILED_QNN_ONNX,
                ScorecardProfilePath.VOICE_AI,
            },
        ),
    ],
)
def test_drop_entries_with_mismatched_aot(
    requires_aot_prepare: bool, kept: set[ScorecardProfilePath]
) -> None:
    """Entries the scorecard can no longer test are dropped; orchestrators are kept."""
    other = ScorecardDevice.get("Samsung Galaxy S25")
    perf = _mixed_perf(cs_x_elite, other)
    perf.drop_entries_with_mismatched_aot(requires_aot_prepare)
    for component in ("A", "B"):
        assert _paths(perf, component) == kept
        devices = {key.device for key in perf.entries if key.component == component}
        assert devices == {cs_x_elite, other}


def test_drop_entries_with_mismatched_aot_respects_component_filter() -> None:
    perf = _mixed_perf()
    perf.drop_entries_with_mismatched_aot(False, only_components={"A"})
    assert ScorecardProfilePath.QNN_CONTEXT_BINARY not in _paths(perf, "A")
    assert _paths(perf, "B") == set(MIXED_PATHS)


def test_drop_entries_with_mismatched_aot_removes_emptied_precision() -> None:
    perf = QAIHMModelPerf("mixed")
    key = QAIHMModelPerf.Key(
        Precision.float, "A", cs_x_elite, ScorecardProfilePath.QNN_CONTEXT_BINARY
    )
    perf.update_entry(key.stamp(perf_pb2.ModelPerf.PerformanceDetails()))
    perf.drop_entries_with_mismatched_aot(False)
    assert not any(key.precision == Precision.float for key in perf.entries)
