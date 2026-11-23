# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    compute_peer_chipsets,
    cs_universal,
)
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath


def test_exactly_one_default() -> None:
    num_defaults = 0
    for device in ScorecardDevice._registry.values():
        if device.is_default:
            num_defaults += 1
    assert num_defaults == 1, (
        f"Must be exactly one default device, found {num_defaults}"
    )


def test_get_universal_reference_returns_default() -> None:
    """
    Test that querying by cs_universal's reference device name returns
    the default device, not the universal device itself.
    """
    result = ScorecardDevice.get(cs_universal.reference_device_name)
    assert result.is_default, (
        f"Expected default device, got {result.name} (is_default={result.is_default})"
    )
    assert result.name != "universal", (
        f"Expected default device, not universal. Got {result.name}"
    )


def test_compute_peer_chipsets_pairs_x_elite_and_x_plus() -> None:
    """X Plus 8-Core is the same NPU as X Elite; we only ever measure X Elite."""
    pair = {"qualcomm-snapdragon-x-elite", "qualcomm-snapdragon-x-plus-8-core"}
    assert compute_peer_chipsets("qualcomm-snapdragon-x-elite") == pair
    assert compute_peer_chipsets("qualcomm-snapdragon-x-plus-8-core") == pair


def test_compute_peer_chipsets_leaves_others_alone() -> None:
    """X2 Elite is a different NPU (htp 81), and mobile chipsets never pair here."""
    for chipset in (
        "qualcomm-snapdragon-x2-elite",
        "qualcomm-snapdragon-8-elite-for-galaxy",
    ):
        assert compute_peer_chipsets(chipset) == {chipset}


def test_extended_supported_chipsets_unchanged_for_mobile() -> None:
    """The compute pairing must not leak into the mobile proxy list."""
    assert ScorecardDevice.get("cs_8_elite").extended_supported_chipsets == {
        "qualcomm-snapdragon-8-elite-for-galaxy",
        "qualcomm-snapdragon-8gen3",
        "qualcomm-snapdragon-8gen2",
        "qualcomm-snapdragon-8gen1",
        "qualcomm-snapdragon-888",
    }


def test_extended_supported_chipsets_pairs_compute() -> None:
    pair = {"qualcomm-snapdragon-x-elite", "qualcomm-snapdragon-x-plus-8-core"}
    assert ScorecardDevice.get("cs_x_elite").extended_supported_chipsets == pair
    assert ScorecardDevice.get("cs_x_plus_8_core").extended_supported_chipsets == pair
    assert ScorecardDevice.get("cs_x2_elite").extended_supported_chipsets == {
        "qualcomm-snapdragon-x2-elite"
    }


def test_supported_chipsets_for_paths_drops_proxies_when_aot() -> None:
    """A context binary is built per chipset, so older chipsets can't be proxied."""
    device = ScorecardDevice.get("cs_8_elite")
    assert device.supported_chipsets_for_paths(
        [ScorecardInferencePath.QNN_CONTEXT_BINARY]
    ) == {"qualcomm-snapdragon-8-elite-for-galaxy"}


def test_supported_chipsets_for_paths_keeps_proxies_when_jit() -> None:
    device = ScorecardDevice.get("cs_8_elite")
    assert (
        device.supported_chipsets_for_paths([ScorecardInferencePath.TFLITE])
        == device.extended_supported_chipsets
    )


def test_supported_chipsets_for_paths_keeps_proxies_when_any_path_is_jit() -> None:
    """One JIT asset is enough to justify the older chipsets."""
    device = ScorecardDevice.get("cs_8_elite")
    assert (
        device.supported_chipsets_for_paths(
            [
                ScorecardInferencePath.QNN_CONTEXT_BINARY,
                ScorecardInferencePath.TFLITE,
            ]
        )
        == device.extended_supported_chipsets
    )


def test_supported_chipsets_for_paths_keeps_compute_peers_when_aot() -> None:
    """X Elite and X Plus share one uploaded artifact, so the pairing always holds."""
    pair = {"qualcomm-snapdragon-x-elite", "qualcomm-snapdragon-x-plus-8-core"}
    for name in ("cs_x_elite", "cs_x_plus_8_core"):
        assert (
            ScorecardDevice.get(name).supported_chipsets_for_paths(
                [ScorecardInferencePath.QNN_CONTEXT_BINARY]
            )
            == pair
        )


def test_supported_chipsets_for_paths_exact_for_non_mobile() -> None:
    """Automotive and IoT devices never proxy, so AOT changes nothing for them."""
    for name in ("cs_auto_lemans_8775", "cs_9075"):
        device = ScorecardDevice.get(name)
        assert device.supported_chipsets_for_paths(
            [ScorecardInferencePath.QNN_CONTEXT_BINARY]
        ) == {device.chipset}


def test_geniex_qairt_excluded_for_automotive() -> None:
    """GenieX (QAIRT) assets are not published for automotive devices, but Genie still is."""
    for device_name in (
        "cs_auto_lemans_8775",
        "cs_auto_monaco_7255",
        "cs_auto_makena_8295",
    ):
        device = ScorecardDevice.get(device_name)
        assert ScorecardInferencePath.GENIEX_QAIRT not in device.profile_paths
        assert ScorecardInferencePath.GENIE in device.profile_paths
        assert (
            ScorecardInferencePath.GENIEX_QAIRT.compile_path not in device.compile_paths
        )
        assert ScorecardInferencePath.GENIE.compile_path in device.compile_paths
