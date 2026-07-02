# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.cli import _run_fetch, add_fetch_parser
from qai_hub_models_cli.fetch import _asset_url, fetch, get_asset_url
from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    FormFactor,
    FormFactorInfo,
    PlatformInfo,
    RuntimeInfo,
)
from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.release_assets import AssetNotFoundError

# Legacy (pre-manifest) version vs a manifest-era version (>= MIN_MANIFEST_VERSION).
_LEGACY_VERSION = Version("0.45.0")
_VERSION = Version("0.52.0")
_TFLITE = Runtime.RUNTIME_TFLITE
_QNN = Runtime.RUNTIME_QNN_CONTEXT_BINARY
_FLOAT = Precision.PRECISION_FLOAT
_W8A8 = Precision.PRECISION_W8A8
_CHIP = "qualcomm-snapdragon-8-gen-3"


# ── _asset_url ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("chipset", "expected_filename"),
    [
        (None, "mobilenet_v2-tflite-float.zip"),
        (_CHIP, "mobilenet_v2-tflite-float-qualcomm_snapdragon_8_gen_3.zip"),
    ],
)
def test_asset_url(chipset: str | None, expected_filename: str) -> None:
    url, filename = _asset_url(
        "mobilenet_v2", "tflite", "float", _LEGACY_VERSION, chipset
    )
    assert filename == expected_filename
    assert url.endswith(expected_filename)


# ── get_asset_url (legacy head-request path) ─────────────────────────


def _mock_head(status_map: dict[str, int]) -> object:
    """Mock requests.head returning a status based on URL substrings (else 404)."""

    def _head(url: str, timeout: int = 10) -> MagicMock:
        resp = MagicMock()
        resp.status_code = next((s for pat, s in status_map.items() if pat in url), 404)
        return resp

    return _head


def test_get_asset_url_legacy_found() -> None:
    with patch("qai_hub_models_cli.fetch.requests.head", _mock_head({"tflite": 200})):
        url = get_asset_url(
            model="mobilenet_v2",
            runtime="tflite",
            precision="float",
            version=_LEGACY_VERSION,
        )
    assert "mobilenet_v2-tflite-float.zip" in url


def test_get_asset_url_legacy_chipset_fallback() -> None:
    """A missing chipset asset falls back to the generic URL."""
    with patch(
        "qai_hub_models_cli.fetch.requests.head",
        _mock_head({"qualcomm_snapdragon": 404, "tflite-float.zip": 200}),
    ):
        url = get_asset_url(
            model="mobilenet_v2",
            runtime="tflite",
            precision="float",
            version=_LEGACY_VERSION,
            chipset=_CHIP,
        )
    assert "qualcomm_snapdragon" not in url


def test_get_asset_url_legacy_not_found() -> None:
    with (
        patch("qai_hub_models_cli.fetch.requests.head", _mock_head({})),
        pytest.raises(FileNotFoundError, match="No asset found"),
    ):
        get_asset_url(
            model="fake_model",
            runtime="tflite",
            precision="float",
            version=_LEGACY_VERSION,
        )


def test_get_asset_url_legacy_unexpected_status() -> None:
    with (
        patch("qai_hub_models_cli.fetch.requests.head", _mock_head({"tflite": 500})),
        pytest.raises(ConnectionError, match="Unexpected response"),
    ):
        get_asset_url(
            model="mobilenet_v2",
            runtime="tflite",
            precision="float",
            version=_LEGACY_VERSION,
        )


# ── get_asset_url (manifest path) ────────────────────────────────────


def _platform(*_args: object) -> PlatformInfo:
    """Platform with a non-AOT tflite runtime and an AOT qnn runtime.

    Accepts (and ignores) a version arg so it can stand in for ``get_platform``.
    """
    return PlatformInfo(
        aihm_version="0.45.0",
        runtimes=[
            RuntimeInfo(runtime=_TFLITE, is_aot_compiled=False),
            RuntimeInfo(runtime=_QNN, is_aot_compiled=True),
        ],
        form_factors=[FormFactorInfo(form_factor=FormFactor.FORM_FACTOR_PHONE)],
        devices=[
            DeviceInfo(
                name="Samsung Galaxy S24",
                chipset=_CHIP,
                form_factor=FormFactor.FORM_FACTOR_PHONE,
            )
        ],
        chipsets=[ChipsetInfo(name=_CHIP, marketing_name="Snapdragon 8 Gen 3")],
    )


def _assets(*specs: tuple) -> ModelReleaseAssets:
    """Build release assets from ``(precision, runtime, chipset_or_None)`` specs."""
    return ModelReleaseAssets(
        aihm_version="0.45.0",
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


@patch("qai_hub_models_cli.fetch.get_platform", _platform)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_single_match_returns_url(mock_assets: MagicMock) -> None:
    mock_assets.return_value = _assets((_FLOAT, _TFLITE, None), (_W8A8, _QNN, _CHIP))
    assert (
        get_asset_url(
            model="mobilenet_v2", runtime="tflite", precision="float", version=_VERSION
        )
        == "https://example.com/0.zip"
    )


@patch("qai_hub_models_cli.fetch.get_platform", _platform)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_sdk_version_disambiguates(mock_assets: MagicMock) -> None:
    """sdk_versions reaches the download resolution and picks the matching asset."""
    from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions

    # Two assets identical except for QAIRT version; only sdk_versions can pick one.
    mock_assets.return_value = ModelReleaseAssets(
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=_FLOAT,
                runtime=_TFLITE,
                download_url="https://example.com/old.zip",
                tool_versions=ToolVersions(qairt="2.20"),
            ),
            ModelReleaseAssets.AssetDetails(
                precision=_FLOAT,
                runtime=_TFLITE,
                download_url="https://example.com/new.zip",
                tool_versions=ToolVersions(qairt="2.31"),
            ),
        ],
    )
    url = get_asset_url(
        model="mobilenet_v2",
        runtime="tflite",
        precision="float",
        version=_VERSION,
        sdk_versions={"qairt": "2.31"},
    )
    assert url == "https://example.com/new.zip"


@pytest.mark.parametrize(
    ("runtime", "precision", "assets", "reason"),
    [
        # No runtime, everything matches -> ask for a runtime.
        (
            None,
            None,
            ((_FLOAT, _TFLITE, None), (_W8A8, _QNN, _CHIP)),
            "runtime is required",
        ),
        # Runtime alone leaves two precisions -> ask for a precision.
        (
            "tflite",
            None,
            ((_FLOAT, _TFLITE, None), (_W8A8, _TFLITE, None)),
            "precision is required",
        ),
        # AOT runtime with two chipsets and no chipset arg -> ask for a chipset.
        (
            "qnn_context_binary",
            "w8a8",
            ((_W8A8, _QNN, _CHIP), (_W8A8, _QNN, "qualcomm-snapdragon-8-elite")),
            "chipset or device is required",
        ),
        # Fully specified but still two matches -> ask to narrow filters.
        (
            "tflite",
            "float",
            ((_FLOAT, _TFLITE, None), (_FLOAT, _TFLITE, None)),
            "2 assets match",
        ),
    ],
)
@patch("qai_hub_models_cli.fetch.get_platform", _platform)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_ambiguous(
    mock_assets: MagicMock,
    runtime: str | None,
    precision: str | None,
    assets: tuple,
    reason: str,
) -> None:
    """Ambiguous requests raise with the reason and the matching-assets table."""
    mock_assets.return_value = _assets(*assets)
    with pytest.raises(AssetNotFoundError, match=reason) as exc:
        get_asset_url(
            model="mobilenet_v2",
            runtime=runtime,
            precision=precision,
            version=_VERSION,
        )
    assert "current selection(s)" in str(exc.value)  # table shown


@patch("qai_hub_models_cli.fetch.get_platform", _platform)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_no_match(mock_assets: MagicMock) -> None:
    """Zero matches gives a terse message with the list-all hint and no table."""
    mock_assets.return_value = _assets((_FLOAT, _TFLITE, None))
    with pytest.raises(AssetNotFoundError) as exc:
        get_asset_url(
            model="mobilenet_v2", runtime="tflite", precision="w8a8", version=_VERSION
        )
    msg = str(exc.value)
    assert "No asset found" in msg
    # The list-all hint carries the browsed version so it stays pinned.
    assert "fetch mobilenet_v2 -v 0.52.0 -i" in msg
    assert "Precision" not in msg  # no table


_SIMILAR_CHIP = "qualcomm-sa8255p"


def _platform_with_similar(*_args: object) -> PlatformInfo:
    """Platform where a "similar" device/chipset borrows assets from ``_CHIP``.

    ``SA8255P ADP`` is a similar device (it carries a ``reference_chipset``) on
    the ``_SIMILAR_CHIP`` chipset, which is only reachable through that device.
    Neither has assets of its own; both borrow from ``_CHIP``.
    """
    base = _platform()
    base.devices.append(
        DeviceInfo(
            name="SA8255P ADP",
            chipset=_SIMILAR_CHIP,
            form_factor=FormFactor.FORM_FACTOR_AUTO,
            reference_chipset=_CHIP,
        )
    )
    base.chipsets.append(
        ChipsetInfo(name=_SIMILAR_CHIP, marketing_name="Snapdragon SA8255P")
    )
    return base


@pytest.mark.parametrize("by", ["device", "chipset"])
@patch("qai_hub_models_cli.fetch.get_platform", _platform_with_similar)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_similar_suggests_reference(
    mock_assets: MagicMock, by: str
) -> None:
    """A similar chipset/device with no assets points at its reference chipset."""
    mock_assets.return_value = _assets((_W8A8, _QNN, _CHIP))
    with pytest.raises(AssetNotFoundError) as exc:
        get_asset_url(
            model="mobilenet_v2",
            runtime="qnn_context_binary",
            precision="w8a8",
            version=_VERSION,
            device="SA8255P ADP" if by == "device" else None,
            chipset=None if by == "device" else _SIMILAR_CHIP,
        )
    msg = str(exc.value)
    assert "'similar' to 'Snapdragon 8 Gen 3'" in msg
    # A ready-to-run fetch command targeting the reference chipset is offered.
    assert "-c 'Snapdragon 8 Gen 3'" in msg


@patch("qai_hub_models_cli.fetch.get_platform", _platform_with_similar)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_similar_no_reference_asset(mock_assets: MagicMock) -> None:
    """If the reference chipset also lacks the asset, fall back to the plain error."""
    # Reference chipset has a QNN asset, but the user asked for tflite.
    mock_assets.return_value = _assets((_W8A8, _QNN, _CHIP))
    with pytest.raises(AssetNotFoundError) as exc:
        get_asset_url(
            model="mobilenet_v2",
            runtime="tflite",
            precision="w8a8",
            version=_VERSION,
            device="SA8255P ADP",
        )
    msg = str(exc.value)
    assert "No asset found" in msg
    assert "similar" not in msg


@patch("qai_hub_models_cli.fetch.get_platform", _platform)
@patch("qai_hub_models_cli.fetch.get_model_release_assets")
def test_get_asset_url_quiet_omits_table(mock_assets: MagicMock) -> None:
    mock_assets.return_value = _assets((_FLOAT, _TFLITE, None), (_FLOAT, _TFLITE, None))
    with pytest.raises(AssetNotFoundError) as exc:
        get_asset_url(
            model="mobilenet_v2",
            runtime="tflite",
            precision="float",
            version=_VERSION,
            quiet=True,
        )
    msg = str(exc.value)
    assert "2 assets match" in msg
    assert "Precision" not in msg  # no table
    assert "fetch mobilenet_v2 -i" not in msg  # no command hints


# ── fetch ────────────────────────────────────────────────────────────


@patch("qai_hub_models_cli.fetch.get_asset_url", return_value="s3://bucket/m.zip")
@patch("qai_hub_models_cli.fetch.download")
def test_fetch_forwards_args_and_downloads(
    mock_download: MagicMock, mock_get_url: MagicMock, tmp_path: Path
) -> None:
    mock_download.return_value = tmp_path / "m.zip"
    result = fetch(
        model="model",
        runtime="qnn_context_binary",
        output_dir=tmp_path,
        device="S24",
        version=_VERSION,
    )
    assert result == tmp_path / "m.zip"
    # The resolved URL flows into download; the device flows into get_asset_url.
    assert mock_download.call_args.args[0] == "s3://bucket/m.zip"
    assert mock_get_url.call_args.kwargs["device"] == "S24"


@patch("qai_hub_models_cli.fetch.get_asset_url", return_value="https://x/m-tflite.zip")
@patch("qai_hub_models_cli.fetch.download")
def test_fetch_extract_increments_dir(
    mock_download: MagicMock, mock_get_url: MagicMock, tmp_path: Path
) -> None:
    (tmp_path / "m-tflite").mkdir()  # collide with the extraction dir
    fetch(
        model="model",
        runtime="tflite",
        output_dir=tmp_path,
        extract=True,
        version=_VERSION,
    )
    assert "m-tflite-1" in str(mock_download.call_args)


# ── _run_fetch (CLI) ─────────────────────────────────────────────────


def _make_args(**overrides: object) -> MagicMock:
    args = MagicMock()
    defaults: dict[str, object] = dict(
        model="mobilenet_v2",
        runtime="tflite",
        precision="float",
        chipset=None,
        device=None,
        qaihm_version=_VERSION,
        extract=True,
        output_dir=".",
        url_only=False,
        info=False,
        sdk_version=None,
        quiet=False,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


@patch("qai_hub_models_cli.cli.get_asset_url", return_value="https://x/asset.zip")
def test_run_fetch_url_only_prints(
    mock_url: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    _run_fetch(_make_args(url_only=True))
    assert "https://x/asset.zip" in capsys.readouterr().out


@patch("qai_hub_models_cli.cli.format_fetch_commands", return_value="CMDS")
@patch("qai_hub_models_cli.cli.format_release_assets_table", return_value="TABLE")
@patch("qai_hub_models_cli.cli.get_platform")
@patch("qai_hub_models_cli.cli.match_release_assets")
@patch("qai_hub_models_cli.cli.get_model_release_assets")
@patch("qai_hub_models_cli.cli.fetch")
def test_run_fetch_info_filters_and_skips_download(
    mock_fetch: MagicMock,
    mock_get_assets: MagicMock,
    mock_match: MagicMock,
    mock_platform: MagicMock,
    mock_table: MagicMock,
    mock_cmds: MagicMock,
) -> None:
    mock_match.return_value.matches.assets = [MagicMock()]
    _run_fetch(_make_args(info=True, runtime="tflite", precision=None))
    mock_fetch.assert_not_called()
    # All filter args (incl. parsed sdk_versions) are forwarded.
    assert "tflite" in mock_match.call_args.args
    assert mock_match.call_args.args[-1] == {}  # no sdk_version filters


@patch("qai_hub_models_cli.cli.format_release_assets_table")
@patch("qai_hub_models_cli.cli.get_platform")
@patch("qai_hub_models_cli.cli.match_release_assets")
@patch("qai_hub_models_cli.cli.get_model_release_assets")
@patch("qai_hub_models_cli.cli.fetch")
def test_run_fetch_info_no_match(
    mock_fetch: MagicMock,
    mock_get_assets: MagicMock,
    mock_match: MagicMock,
    mock_platform: MagicMock,
    mock_table: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_match.return_value.matches.assets = []
    mock_match.return_value.similar_chipset = None
    _run_fetch(_make_args(info=True))
    mock_fetch.assert_not_called()
    mock_table.assert_not_called()
    assert "No release assets match" in capsys.readouterr().out


@patch("qai_hub_models_cli.cli.print_upgrade_notice")
@patch("qai_hub_models_cli.cli.get_model_asset_details")
@patch("qai_hub_models_cli.cli.fetch", return_value=Path("/out/model"))
def test_run_fetch_smoke(
    mock_fetch: MagicMock, mock_asset: MagicMock, mock_notice: MagicMock
) -> None:
    """Drive _run_fetch with real parsed args to catch attribute mismatches."""
    parser = argparse.ArgumentParser()
    add_fetch_parser(parser.add_subparsers())
    _run_fetch(parser.parse_args(["fetch", "mobilenet_v2", "-r", "tflite"]))
    mock_fetch.assert_called_once()
    mock_notice.assert_called_once()


@patch("qai_hub_models_cli.cli.print_upgrade_notice")
@patch("qai_hub_models_cli.cli.fetch", return_value=Path("/out/model"))
def test_run_fetch_quiet_skips_upgrade_notice(
    mock_fetch: MagicMock, mock_notice: MagicMock
) -> None:
    _run_fetch(_make_args(quiet=True))
    mock_notice.assert_not_called()


# ── add_fetch_parser ─────────────────────────────────────────────────


def test_add_fetch_parser() -> None:
    parser = argparse.ArgumentParser()
    add_fetch_parser(parser.add_subparsers())

    args = parser.parse_args(
        [
            "fetch",
            "mobilenet_v2",
            "-r",
            "tflite",
            "-i",
            "--sdk-version",
            "QAIRT=2.20",
            "LiteRT=1.4.4",
        ]
    )
    assert args.model == "mobilenet_v2"
    assert args.runtime == "tflite"
    assert args.precision is None  # no implicit default
    assert args.info is True
    # nargs="+" collects multiple values, lowercased.
    assert args.sdk_version == ["qairt=2.20", "litert=1.4.4"]

    # chipset and device are mutually exclusive.
    with pytest.raises(SystemExit):
        parser.parse_args(["fetch", "m", "-r", "tflite", "-c", "c", "-d", "d"])
