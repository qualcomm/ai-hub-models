# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.cli import _run_find, add_find_parser
from qai_hub_models_cli.find import (
    default_search_versions,
    find_in_version,
    find_matching_releases,
)
from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    FormFactor,
    PlatformInfo,
    RuntimeInfo,
)
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.versions import UnsupportedVersionError

_TFLITE = Runtime.RUNTIME_TFLITE
_QNN = Runtime.RUNTIME_QNN_CONTEXT_BINARY
_FLOAT = Precision.PRECISION_FLOAT
_W8A8 = Precision.PRECISION_W8A8
_CHIP = "qualcomm-snapdragon-8-gen-3"
_SIMILAR_CHIP = "qualcomm-sa8255p"


def _platform(*_args: object) -> PlatformInfo:
    return PlatformInfo(runtimes=[RuntimeInfo(runtime=_TFLITE, is_aot_compiled=False)])


def _platform_with_similar(*_args: object) -> PlatformInfo:
    """Platform where ``SA8255P ADP``/``_SIMILAR_CHIP`` borrow assets from ``_CHIP``."""
    return PlatformInfo(
        runtimes=[
            RuntimeInfo(runtime=_TFLITE, is_aot_compiled=False),
            RuntimeInfo(runtime=_QNN, is_aot_compiled=True),
        ],
        devices=[
            DeviceInfo(name="Samsung Galaxy S24", chipset=_CHIP),
            DeviceInfo(
                name="SA8255P ADP",
                chipset=_SIMILAR_CHIP,
                form_factor=FormFactor.FORM_FACTOR_AUTO,
                reference_chipset=_CHIP,
            ),
        ],
        chipsets=[
            ChipsetInfo(name=_CHIP, marketing_name="Snapdragon 8 Gen 3"),
            ChipsetInfo(name=_SIMILAR_CHIP, marketing_name="Snapdragon SA8255P"),
        ],
    )


def _assets(*runtimes_precisions: tuple) -> ModelReleaseAssets:
    return ModelReleaseAssets(
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=p, runtime=r, download_url=f"https://example.com/{i}.zip"
            )
            for i, (p, r) in enumerate(runtimes_precisions)
        ],
    )


def _chipset_assets(*specs: tuple) -> ModelReleaseAssets:
    """Build assets from ``(precision, runtime, chipset_or_None)`` specs."""
    return ModelReleaseAssets(
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=p,
                runtime=r,
                download_url=f"https://example.com/{i}.zip",
                **({"chipset": c} if c else {}),
            )
            for i, (p, r, c) in enumerate(specs)
        ],
    )


# ── find_in_version ──────────────────────────────────────────────────


@patch("qai_hub_models_cli.find.get_platform", _platform)
@patch("qai_hub_models_cli.find.get_model_release_assets")
def test_find_in_version_filters_assets(mock_assets: MagicMock) -> None:
    mock_assets.return_value = _assets((_FLOAT, _TFLITE), (_W8A8, _TFLITE))
    # Matching filter returns the asset; a non-matching one is a plain miss.
    matched, reason, similar = find_in_version(
        "mobilenet_v2", Version("0.52.0"), runtime="tflite", precision="float"
    )
    assert reason is None
    assert similar is None
    assert matched is not None
    assert [a.precision for a in matched.assets] == [_FLOAT]
    assert find_in_version("mobilenet_v2", Version("0.52.0"), precision="w8a16") == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("kwargs", "side_effect", "reason_substr"),
    [
        ({}, FileNotFoundError("x"), "not in this release"),
        ({}, UnsupportedVersionError("x"), "predates the asset manifest"),
        ({"chipset": "not-a-chip"}, None, "not-a-chip"),
    ],
)
@patch("qai_hub_models_cli.find.get_platform", _platform)
@patch("qai_hub_models_cli.find.get_model_release_assets")
def test_find_in_version_skip_reasons(
    mock_assets: MagicMock,
    kwargs: dict,
    side_effect: Exception | None,
    reason_substr: str,
) -> None:
    if side_effect is not None:
        mock_assets.side_effect = side_effect
    else:
        mock_assets.return_value = _assets((_FLOAT, _TFLITE))
    matched, reason, _ = find_in_version("mobilenet_v2", Version("0.52.0"), **kwargs)
    assert matched is None
    assert reason is not None and reason_substr in reason


@pytest.mark.parametrize("by", ["device", "chipset"])
@patch("qai_hub_models_cli.find.get_platform", _platform_with_similar)
@patch("qai_hub_models_cli.find.get_model_release_assets")
def test_find_in_version_similar_chipset(mock_assets: MagicMock, by: str) -> None:
    """A similar chipset/device is not a match; it reports the reference chipset."""
    mock_assets.return_value = _chipset_assets((_W8A8, _QNN, _CHIP))
    matched, reason, similar = find_in_version(
        "mobilenet_v2",
        Version("0.52.0"),
        runtime="qnn_context_binary",
        precision="w8a8",
        device="SA8255P ADP" if by == "device" else None,
        chipset=None if by == "device" else _SIMILAR_CHIP,
    )
    # Not treated as a match: no assets, a skip reason, and the reference chipset.
    assert matched is None
    assert reason is not None and "similar to 'Snapdragon 8 Gen 3'" in reason
    assert similar is not None and similar.marketing_name == "Snapdragon 8 Gen 3"


# ── default_search_versions ──────────────────────────────────────────


@patch("qai_hub_models_cli.find.get_supported_versions")
def test_default_search_versions(mock_supported: MagicMock) -> None:
    mock_supported.return_value = [Version(v) for v in ("0.56.0", "0.52.0", "0.45.0")]
    # 0.45.0 is below MIN_MANIFEST_VERSION and is dropped.
    assert default_search_versions() == [Version("0.56.0"), Version("0.52.0")]
    # min/max clamp the range.
    assert default_search_versions(
        min_version=Version("0.53.0"), max_version=Version("0.56.0")
    ) == [Version("0.56.0")]


# ── find_matching_releases ───────────────────────────────────────────


@patch("qai_hub_models_cli.find.find_in_version")
def test_find_matching_releases_first_only_vs_all(mock_find: MagicMock) -> None:
    versions = [Version("0.54.0"), Version("0.53.0"), Version("0.52.0")]
    # 0.54.0 and 0.52.0 match; 0.53.0 does not.
    mock_find.side_effect = lambda m, v, *a: (
        (None, None, None) if v == versions[1] else (_assets(), None, None)
    )
    first, _ = find_matching_releases("m", versions=versions, first_only=True)
    assert [v for v, _ in first] == [Version("0.54.0")]
    all_results, _ = find_matching_releases("m", versions=versions, first_only=False)
    assert [v for v, _ in all_results] == [Version("0.54.0"), Version("0.52.0")]


@patch("qai_hub_models_cli.find.find_in_version")
def test_find_matching_releases_reports_similar_when_no_match(
    mock_find: MagicMock,
) -> None:
    """With no real match, the similar reference chipset is surfaced to the caller."""
    ref = ChipsetInfo(name=_CHIP, marketing_name="Snapdragon 8 Gen 3")
    mock_find.return_value = (None, "similar to 'Snapdragon 8 Gen 3'", ref)
    results, similar = find_matching_releases("m", versions=[Version("0.54.0")])
    assert results == []
    assert similar is ref

    # A real match wins over the substitute: similar is cleared.
    mock_find.side_effect = lambda m, v, *a: (
        (_assets(), None, None) if v == Version("0.54.0") else (None, "x", ref)
    )
    mock_find.return_value = None
    results, similar = find_matching_releases(
        "m", versions=[Version("0.54.0"), Version("0.52.0")], first_only=False
    )
    assert results and similar is None


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"chipset": "c", "device": "d"}, ValueError),
        ({"runtime": "not-a-runtime"}, KeyError),
        ({"precision": "not-a-precision"}, KeyError),
        ({"sdk_versions": {"notatool": "1.0"}}, ValueError),
    ],
)
def test_find_matching_releases_validates_filters(
    kwargs: dict, exc: type[Exception]
) -> None:
    with pytest.raises(exc):
        find_matching_releases("m", versions=[], **kwargs)


# ── _run_find (CLI) ──────────────────────────────────────────────────


def _make_args(**overrides: object) -> argparse.Namespace:
    defaults = dict(
        model="mobilenet_v2",
        runtime="tflite",
        precision="float",
        chipset=None,
        device=None,
        sdk_version=None,
        min_version=None,
        max_version=None,
        all=False,
        quiet=False,
    )
    return argparse.Namespace(**{**defaults, **overrides})


@patch("qai_hub_models_cli.cli.format_fetch_commands", return_value="CMDS")
@patch("qai_hub_models_cli.cli.format_release_assets_table", return_value="TABLE")
@patch("qai_hub_models_cli.cli.get_platform", _platform)
@patch("qai_hub_models_cli.cli.find_matching_releases")
def test_run_find_forwards_flags_and_quiet(
    mock_find: MagicMock,
    mock_table: MagicMock,
    mock_cmds: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_find.return_value = (
        [
            (Version("0.54.0"), _assets((_FLOAT, _TFLITE))),
            (Version("0.52.0"), _assets((_FLOAT, _TFLITE))),
        ],
        None,
    )
    # --all and the info hint are forwarded.
    _run_find(_make_args(all=True))
    assert mock_find.call_args.kwargs["first_only"] is False
    assert mock_cmds.call_args.kwargs["include_info"] is True
    capsys.readouterr()  # discard
    # Quiet prints only versions and suppresses progress.
    _run_find(_make_args(quiet=True))
    assert capsys.readouterr().out.split() == ["0.54.0", "0.52.0"]
    assert mock_find.call_args.kwargs["progress"] is None


@patch("qai_hub_models_cli.cli.get_platform", _platform_with_similar)
@patch("qai_hub_models_cli.cli.find_matching_releases")
def test_run_find_similar_chipset_suggests_reference_search(
    mock_find: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    similar = ChipsetInfo(name=_CHIP, marketing_name="Snapdragon 8 Gen 3")
    mock_find.return_value = ([], similar)
    _run_find(
        _make_args(runtime="qnn_context_binary", precision="w8a8", device="SA8255P ADP")
    )
    out = capsys.readouterr().out
    assert "Could not find a release" in out
    assert "device 'SA8255P ADP' is 'similar' to 'Snapdragon 8 Gen 3'" in out
    # A sample command searching for the reference chipset is shown.
    assert (
        "find mobilenet_v2 -r 'qnn_context_binary' -p 'w8a8' "
        "-c 'Snapdragon 8 Gen 3'" in out
    )


def test_add_find_parser() -> None:
    parser = argparse.ArgumentParser()
    add_find_parser(parser.add_subparsers())
    args = parser.parse_args(
        ["find", "mobilenet_v2", "-r", "tflite", "--all", "--min-version", "v0.52.0"]
    )
    assert (args.model, args.all, args.min_version) == (
        "mobilenet_v2",
        True,
        Version("0.52.0"),
    )
    # chipset and device are mutually exclusive.
    with pytest.raises(SystemExit):
        parser.parse_args(["find", "m", "-c", "c", "-d", "d"])
