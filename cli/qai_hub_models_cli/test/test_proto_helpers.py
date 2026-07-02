# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from qai_hub_models_cli.envvars import FORCE_MANIFEST_ROOT_ENVVAR
from qai_hub_models_cli.proto.info_pb2 import (
    ModelDomain,
    ModelInfo,
    ModelLicense,
    ModelTag,
    ModelUseCase,
)
from qai_hub_models_cli.proto.manifest_pb2 import (
    ManifestModelEntry,
    ModelManifestUrls,
    ReleaseManifest,
)
from qai_hub_models_cli.proto.numerics_pb2 import ModelNumerics
from qai_hub_models_cli.proto.perf_pb2 import ModelPerf
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

# ── Test fixtures ─────────────────────────────────────────────────────

_RELEASE_VERSION = Version("0.52.0")
_DEV_VERSION = Version("0.45.0.dev1")


def _manifest() -> ReleaseManifest:
    return ReleaseManifest(
        version="0.45.0",
        platform_url="https://example.com/platform.pb",
        models=[
            ManifestModelEntry(
                id="mobilenet_v2",
                display_name="MobileNet V2",
                domain=ModelDomain.MODEL_DOMAIN_COMPUTER_VISION,
                manifest_urls=ModelManifestUrls(
                    info="https://example.com/mobilenet_v2/info.pb",
                    perf="https://example.com/mobilenet_v2/perf.pb",
                    numerics="https://example.com/mobilenet_v2/numerics.pb",
                    release_assets="https://example.com/mobilenet_v2/release_assets.pb",
                ),
            ),
            ManifestModelEntry(
                id="whisper_small",
                display_name="Whisper Small",
                domain=ModelDomain.MODEL_DOMAIN_AUDIO,
                manifest_urls=ModelManifestUrls(
                    info="https://example.com/whisper_small/info.pb",
                    perf="https://example.com/whisper_small/perf.pb",
                    numerics="https://example.com/whisper_small/numerics.pb",
                    release_assets="https://example.com/whisper_small/release_assets.pb",
                ),
            ),
        ],
    )


def _model_info() -> ModelInfo:
    return ModelInfo(
        id="mobilenet_v2",
        name="MobileNet V2",
        domain=ModelDomain.MODEL_DOMAIN_COMPUTER_VISION,
        use_case=ModelUseCase.MODEL_USE_CASE_IMAGE_CLASSIFICATION,
        tags=[ModelTag.MODEL_TAG_REAL_TIME],
        license_type=ModelLicense.MODEL_LICENSE_APACHE_2_0,
    )


def _model_perf() -> ModelPerf:
    return ModelPerf(
        aihm_version="0.45.0",
        model_id="mobilenet_v2",
        supported_devices=["Samsung Galaxy S24"],
        supported_chipsets=["qualcomm-snapdragon-8-gen-3"],
        performance_metrics=[
            ModelPerf.PerformanceDetails(
                precision=Precision.PRECISION_FLOAT,
                device="Samsung Galaxy S24",
                runtime=Runtime.RUNTIME_TFLITE,
                metrics=ModelPerf.PerfMetrics(
                    inference_time_milliseconds=2.5,
                    primary_compute_unit="NPU",
                ),
            ),
        ],
    )


def _model_numerics() -> ModelNumerics:
    return ModelNumerics(
        aihm_version="0.45.0",
        model_id="mobilenet_v2",
        metrics=[
            ModelNumerics.NumericsMetric(
                dataset_name="imagenet",
                metric_name="Top-1 Accuracy",
                metric_unit="%",
                partial_torch_metric=71.8,
            ),
        ],
    )


def _model_release_assets() -> ModelReleaseAssets:
    return ModelReleaseAssets(
        aihm_version="0.45.0",
        model_id="mobilenet_v2",
        assets=[
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_TFLITE,
                download_url="https://example.com/mobilenet_v2-tflite-float.zip",
            ),
            ModelReleaseAssets.AssetDetails(
                precision=Precision.PRECISION_FLOAT,
                runtime=Runtime.RUNTIME_QNN_CONTEXT_BINARY,
                chipset="qualcomm-snapdragon-8-gen-3",
                download_url="https://example.com/mobilenet_v2-qnn-float-8g3.zip",
            ),
        ],
    )


def _platform_info() -> PlatformInfo:
    return PlatformInfo(
        aihm_version="0.45.0",
        runtimes=[
            RuntimeInfo(
                runtime=Runtime.RUNTIME_TFLITE,
                website_runtime="TFLite",
                file_extension=".tflite",
                is_aot_compiled=False,
            ),
            RuntimeInfo(
                runtime=Runtime.RUNTIME_QNN_CONTEXT_BINARY,
                website_runtime="QNN",
                file_extension=".bin",
                is_aot_compiled=True,
            ),
        ],
        form_factors=[
            FormFactorInfo(
                form_factor=FormFactor.FORM_FACTOR_PHONE,
                display_name="Phone",
            ),
        ],
        devices=[
            DeviceInfo(
                name="Samsung Galaxy S24",
                chipset="qualcomm-snapdragon-8-gen-3",
                form_factor=FormFactor.FORM_FACTOR_PHONE,
            ),
        ],
        chipsets=[
            ChipsetInfo(
                name="qualcomm-snapdragon-8-gen-3",
                marketing_name="Snapdragon 8 Gen 3",
                aliases=["sd8gen3"],
            ),
        ],
    )


@pytest.fixture(autouse=True)
def _clear_lru_caches() -> Generator[None, None, None]:
    from qai_hub_models_cli.proto_helpers.info import get_model_info
    from qai_hub_models_cli.proto_helpers.manifest import get_manifest
    from qai_hub_models_cli.proto_helpers.numerics import get_model_numerics
    from qai_hub_models_cli.proto_helpers.perf import get_model_perf
    from qai_hub_models_cli.proto_helpers.platform import get_platform
    from qai_hub_models_cli.proto_helpers.release_assets import (
        get_model_release_assets,
    )

    fns = (
        get_manifest,
        get_model_info,
        get_model_perf,
        get_model_numerics,
        get_model_release_assets,
        get_platform,
    )
    for fn in fns:
        fn.cache_clear()
    yield
    for fn in fns:
        fn.cache_clear()


_skip_version_check = patch(
    "qai_hub_models_cli.proto_helpers._common.verify_version_supported"
)
_not_dev_release = patch(
    "qai_hub_models_cli.proto_helpers._common.use_aihm_source", return_value=False
)


# ── _common.py ────────────────────────────────────────────────────────


class TestCommon:
    def test_read_write_proto_roundtrip(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers._common import read_proto, write_proto

        proto = _manifest()
        path = tmp_path / "manifest.pb"
        write_proto(path, proto)
        loaded = read_proto(path, ReleaseManifest)
        assert loaded.version == "0.45.0"
        assert len(loaded.models) == 2

    def test_get_release_cache_dir_envvar_override(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers._common import get_release_cache_dir

        with patch.dict("os.environ", {FORCE_MANIFEST_ROOT_ENVVAR: str(tmp_path)}):
            assert get_release_cache_dir(Version("0.45.0")) == tmp_path

    def test_use_aihm_source_true_for_dev(self) -> None:
        from qai_hub_models_cli.proto_helpers._common import use_aihm_source

        with (
            patch(
                "qai_hub_models_cli.proto_helpers._common.CURRENT_VERSION",
                Version("0.45.0.dev1"),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.is_heavy_package_installed",
                return_value=True,
            ),
        ):
            assert use_aihm_source(Version("0.45.0.dev1")) is True

    def test_use_aihm_source_false_for_release(self) -> None:
        from qai_hub_models_cli.proto_helpers._common import use_aihm_source

        with patch(
            "qai_hub_models_cli.proto_helpers._common.CURRENT_VERSION",
            Version("0.45.0"),
        ):
            assert use_aihm_source(Version("0.45.0")) is False

    def test_use_aihm_source_false_package_not_installed(self) -> None:
        from qai_hub_models_cli.proto_helpers._common import use_aihm_source

        with (
            patch(
                "qai_hub_models_cli.proto_helpers._common.CURRENT_VERSION",
                Version("0.45.0.dev1"),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.is_heavy_package_installed",
                return_value=False,
            ),
        ):
            assert use_aihm_source(Version("0.45.0.dev1")) is False

    def test_fetch_proto_uses_cache(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers._common import fetch_proto

        cache_path = tmp_path / "manifest.pb"
        cache_path.write_bytes(_manifest().SerializeToString())
        with patch("qai_hub_models_cli.proto_helpers._common.download") as mock_dl:
            result = fetch_proto(
                "https://example.com/manifest.pb", cache_path, ReleaseManifest
            )
        mock_dl.assert_not_called()
        assert result.version == "0.45.0"


# ── manifest.py ───────────────────────────────────────────────────────


class TestGetManifest:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        path = tmp_path / "manifest.pb"
        path.write_bytes(_manifest().SerializeToString())
        result = get_manifest(local_path=path)
        assert result.version == "0.45.0"
        assert len(result.models) == 2

    def test_fetches_from_s3(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_path = tmp_path / "releases" / "v0.45.0" / "manifest.pb"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_bytes(_manifest().SerializeToString())

        with (
            _skip_version_check,
            _not_dev_release,
            patch(
                "qai_hub_models_cli.proto_helpers._common.get_release_cache_dir",
                return_value=tmp_path / "releases" / "v0.45.0",
            ),
        ):
            result = get_manifest(_RELEASE_VERSION)
        assert result.version == "0.45.0"


class TestGetManifestEntry:
    def test_lookup_by_id(self) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry

        with patch(
            "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
            return_value=_manifest(),
        ):
            entry = get_manifest_entry("mobilenet_v2", _RELEASE_VERSION)
        assert entry.id == "mobilenet_v2"
        assert entry.display_name == "MobileNet V2"

    def test_not_found(self) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry

        with (
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            pytest.raises(KeyError, match="No model exists"),
        ):
            get_manifest_entry("nonexistent_model", _RELEASE_VERSION)


# ── info.py ───────────────────────────────────────────────────────────


class TestGetModelInfo:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        path = tmp_path / "info.pb"
        path.write_bytes(_model_info().SerializeToString())
        result = get_model_info("mobilenet_v2", local_path=path)
        assert result.id == "mobilenet_v2"
        assert result.name == "MobileNet V2"


# perf / numerics / release_assets are thin wrappers over the same
# fetch_model_proto path; one local_path read each is enough to confirm wiring.
# (The unknown-model KeyError is covered once by TestGetManifestEntry.)


class TestGetModelPerf:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.perf import get_model_perf

        path = tmp_path / "perf.pb"
        path.write_bytes(_model_perf().SerializeToString())
        result = get_model_perf("mobilenet_v2", local_path=path)
        assert result.model_id == "mobilenet_v2"
        assert result.performance_metrics[0].metrics.inference_time_milliseconds == 2.5


class TestGetModelNumerics:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.numerics import get_model_numerics

        path = tmp_path / "numerics.pb"
        path.write_bytes(_model_numerics().SerializeToString())
        result = get_model_numerics("mobilenet_v2", local_path=path)
        assert result.model_id == "mobilenet_v2"
        assert result.metrics[0].dataset_name == "imagenet"


class TestGetModelReleaseAssets:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_release_assets,
        )

        path = tmp_path / "release_assets.pb"
        path.write_bytes(_model_release_assets().SerializeToString())
        result = get_model_release_assets("mobilenet_v2", local_path=path)
        assert result.model_id == "mobilenet_v2"
        assert len(result.assets) == 2


class TestGetModelAssetDetails:
    def test_universal_lookup(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_asset_details,
        )

        asset = get_model_asset_details(
            _model_release_assets(), _platform_info(), "tflite", "float"
        )
        assert asset.runtime == Runtime.RUNTIME_TFLITE

    def test_chipset_required_for_aot(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_asset_details,
        )

        with pytest.raises(FileNotFoundError, match="Chipset is required"):
            get_model_asset_details(
                _model_release_assets(), _platform_info(), "qnn_context_binary", "float"
            )

    @pytest.mark.parametrize(
        "target",
        [
            {"chipset": "qualcomm-snapdragon-8-gen-3"},  # canonical id
            {"chipset": "sd8gen3"},  # alias
            {"device": "Samsung Galaxy S24"},  # device name
        ],
    )
    def test_aot_lookup_by_chipset_or_device(self, target: dict[str, str]) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            get_model_asset_details,
        )

        asset = get_model_asset_details(
            _model_release_assets(),
            _platform_info(),
            "qnn_context_binary",
            "float",
            **target,
        )
        assert asset.chipset == "qualcomm-snapdragon-8-gen-3"


class TestFilterReleaseAssets:
    def test_filters_by_each_field(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            filter_release_assets,
        )

        assets, platform = _model_release_assets(), _platform_info()

        # No filters: all assets, metadata preserved.
        all_filtered = filter_release_assets(assets, platform)
        assert len(all_filtered.assets) == 2
        assert all_filtered.model_id == "mobilenet_v2"

        # runtime narrows to the single tflite asset.
        assert (
            len(filter_release_assets(assets, platform, runtime="tflite").assets) == 1
        )
        # precision="float" matches both assets.
        assert (
            len(filter_release_assets(assets, platform, precision="float").assets) == 2
        )

    def test_chipset_and_device_keep_universal_assets(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            filter_release_assets,
        )

        assets, platform = _model_release_assets(), _platform_info()

        # A universal asset runs on any chipset, so both it and the matching
        # chipset-specific asset are kept (chipset id and device both resolve to
        # the same chipset).
        expected = {"", "qualcomm-snapdragon-8-gen-3"}
        by_chipset = filter_release_assets(
            assets, platform, chipset="qualcomm-snapdragon-8-gen-3"
        )
        by_device = filter_release_assets(assets, platform, device="Samsung Galaxy S24")
        assert {a.chipset for a in by_chipset.assets} == expected
        assert {a.chipset for a in by_device.assets} == expected

        with pytest.raises(ValueError, match="at most one"):
            filter_release_assets(
                assets, platform, chipset="sd8gen3", device="Samsung Galaxy S24"
            )

    def test_sdk_version_filter(self) -> None:
        from qai_hub_models_cli.common import parse_sdk_version_filters
        from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions
        from qai_hub_models_cli.proto_helpers.release_assets import (
            filter_release_assets,
        )

        assets = ModelReleaseAssets(
            aihm_version="0.45.0",
            model_id="mobilenet_v2",
            assets=[
                ModelReleaseAssets.AssetDetails(
                    precision=Precision.PRECISION_FLOAT,
                    runtime=Runtime.RUNTIME_TFLITE,
                    download_url="https://example.com/a.zip",
                    tool_versions=ToolVersions(qairt="2.20", litert="1.4.4"),
                ),
                # No tool versions: never matches an SDK filter.
                ModelReleaseAssets.AssetDetails(
                    precision=Precision.PRECISION_W8A8,
                    runtime=Runtime.RUNTIME_TFLITE,
                    download_url="https://example.com/c.zip",
                ),
            ],
        )
        platform = _platform_info()

        # tool=version syntax, substring match on the named tool's version.
        match = filter_release_assets(
            assets, platform, sdk_versions=parse_sdk_version_filters(["qairt=2.20"])
        )
        assert [a.download_url for a in match.assets] == ["https://example.com/a.zip"]
        assert not filter_release_assets(
            assets, platform, sdk_versions=parse_sdk_version_filters(["qairt=9.99"])
        ).assets

        # Multiple filters are ANDed: both must match the same asset.
        both = parse_sdk_version_filters(["qairt=2.20", "litert=1.4.4"])
        assert [
            a.download_url
            for a in filter_release_assets(assets, platform, sdk_versions=both).assets
        ] == ["https://example.com/a.zip"]
        assert not filter_release_assets(
            assets,
            platform,
            sdk_versions=parse_sdk_version_filters(["qairt=2.20", "litert=9.9"]),
        ).assets

        # Bad syntax is rejected at parse time; an empty version and an unknown
        # tool at match time.
        with pytest.raises(ValueError, match="tool=version"):
            parse_sdk_version_filters(["2.20"])
        with pytest.raises(ValueError, match="is empty"):
            filter_release_assets(
                assets, platform, sdk_versions=parse_sdk_version_filters(["litert="])
            )
        with pytest.raises(ValueError, match="Unknown SDK tool"):
            filter_release_assets(
                assets, platform, sdk_versions=parse_sdk_version_filters(["foo=1.0"])
            )


class TestFormatFetchCommands:
    def test_prefills_known_values_and_chipset_hints(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            format_fetch_commands,
        )

        # Known runtime is filled in; unset precision stays a placeholder. The
        # fixture has a chipset-specific asset, so the device hint/flag appear.
        out = format_fetch_commands(
            _model_release_assets(), "mobilenet_v2", runtime="tflite"
        )
        assert "-r 'tflite' -p <precision>" in out
        assert "See devices per chipset" in out
        assert "[ -c '<chipset>' || -d '<device>' ]" in out

        # An explicit chipset replaces the placeholder flag (quoted for spaces).
        out = format_fetch_commands(
            _model_release_assets(), "mobilenet_v2", chipset="Snapdragon 8 Gen 3"
        )
        assert "-c 'Snapdragon 8 Gen 3'" in out

    def test_echoes_sdk_version_filters(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            format_fetch_commands,
        )

        out = format_fetch_commands(
            _model_release_assets(),
            "mobilenet_v2",
            sdk_versions={"qairt": "2.20", "litert": "1.4.4"},
        )
        assert "-s 'qairt=2.20'" in out
        assert "-s 'litert=1.4.4'" in out

    def test_subset_adds_see_all_hint(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            format_fetch_commands,
        )

        assert "See all assets" not in format_fetch_commands(
            _model_release_assets(), "mobilenet_v2"
        )
        assert "See all assets" in format_fetch_commands(
            _model_release_assets(), "mobilenet_v2", subset=True
        )

    def test_url_only_relabels_and_adds_flag(self) -> None:
        from qai_hub_models_cli.proto_helpers.release_assets import (
            format_fetch_commands,
        )

        out = format_fetch_commands(
            _model_release_assets(), "mobilenet_v2", url_only=True
        )
        assert "Get an asset URL" in out
        assert "Download an asset" not in out
        assert "--url-only" in out


# ── platform.py ───────────────────────────────────────────────────────


class TestGetPlatform:
    def test_local_path(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.platform import get_platform

        path = tmp_path / "platform.pb"
        path.write_bytes(_platform_info().SerializeToString())
        result = get_platform(local_path=path)
        assert result.aihm_version == "0.45.0"
        assert len(result.runtimes) == 2

    def test_fetches_from_s3(self) -> None:
        from qai_hub_models_cli.proto_helpers.platform import get_platform

        with (
            _not_dev_release,
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=False,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.platform.get_manifest",
                return_value=_manifest(),
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.fetch_proto",
                return_value=_platform_info(),
            ),
        ):
            result = get_platform(_RELEASE_VERSION)
        assert result.aihm_version == "0.45.0"


class TestResolveChipset:
    def test_resolves_all_reference_forms(self) -> None:
        from qai_hub_models_cli.proto_helpers.platform import resolve_chipset

        platform = _platform_info()
        cs, dv = platform.chipsets, platform.devices
        expected = "qualcomm-snapdragon-8-gen-3"
        # device name (case-insensitive)
        assert resolve_chipset(cs, dv, device="samsung galaxy s24").name == expected
        # canonical id, marketing name, and alias all resolve.
        assert (
            resolve_chipset(cs, dv, chipset="qualcomm-snapdragon-8-gen-3").name
            == expected
        )
        assert resolve_chipset(cs, dv, chipset="Snapdragon 8 Gen 3").name == expected
        assert resolve_chipset(cs, dv, chipset="sd8gen3").name == expected

        with pytest.raises(ValueError, match="exactly one"):
            resolve_chipset(cs, dv)
        with pytest.raises(KeyError, match="qai-hub-models devices"):
            resolve_chipset(cs, dv, device="nope")
        with pytest.raises(KeyError, match="qai-hub-models chipsets"):
            resolve_chipset(cs, dv, chipset="nope")


def test_form_factor_and_world_display_names() -> None:
    from qai_hub_models_cli.proto.platform_pb2 import FormFactor, WebsiteWorld
    from qai_hub_models_cli.proto_helpers.platform_enums import (
        form_factor_proto_to_str,
        world_proto_to_str,
    )

    assert form_factor_proto_to_str(FormFactor.FORM_FACTOR_XR) == "XR"
    assert form_factor_proto_to_str(FormFactor.FORM_FACTOR_IOT) == "IoT"
    assert world_proto_to_str(WebsiteWorld.WEBSITE_WORLD_AUTOMOTIVE) == "Auto"


# ══════════════════════════════════════════════════════════════════════════
# Disk cache tests (dev-release path)
# Tests manifest (release-level) and info (per-model) as representatives.
# ══════════════════════════════════════════════════════════════════════════


def _enter_mocks(stack: contextlib.ExitStack, mocks: tuple) -> None:
    for m in mocks:
        stack.enter_context(m)


def _aihm_sys_modules(mock_cli: MagicMock) -> dict:
    mock_version_mod = MagicMock()
    mock_version_mod.__version__ = "0.45.0.dev1"
    mock_pkg = MagicMock()
    mock_pkg._version = mock_version_mod
    mock_pkg.cli = mock_cli
    return {
        "qai_hub_models": mock_pkg,
        "qai_hub_models._version": mock_version_mod,
        "qai_hub_models.cli": mock_cli,
    }


class TestManifestDiskCache:
    def _mocks(self, cache_dir: Path, mock_cli: MagicMock) -> tuple:
        return (
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=True,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.get_release_cache_dir",
                return_value=cache_dir,
            ),
            patch.dict("sys.modules", _aihm_sys_modules(mock_cli)),
        )

    def test_cache_miss_writes_file(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_dir = tmp_path / "releases" / "v0.45.0.dev1"
        mock_cli = MagicMock()
        mock_cli.get_manifest_proto = MagicMock(return_value=_manifest())

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_dir, mock_cli))
            result = get_manifest(_DEV_VERSION)

        mock_cli.get_manifest_proto.assert_called_once()
        assert (cache_dir / "manifest.pb").exists()
        assert result.version == "0.45.0"

    def test_cache_hit_skips_getter(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.manifest import get_manifest

        cache_dir = tmp_path / "releases" / "v0.45.0.dev1"
        cache_dir.mkdir(parents=True)
        (cache_dir / "manifest.pb").write_bytes(_manifest().SerializeToString())

        mock_cli = MagicMock()
        mock_cli.get_manifest_proto = MagicMock()

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_dir, mock_cli))
            result = get_manifest(_DEV_VERSION)

        mock_cli.get_manifest_proto.assert_not_called()
        assert result.version == "0.45.0"


class TestInfoDiskCache:
    def _mocks(self, cache_path: Path, mock_cli: MagicMock) -> tuple:
        return (
            patch(
                "qai_hub_models_cli.proto_helpers._common.use_aihm_source",
                return_value=True,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers._common.model_cache_path",
                return_value=cache_path,
            ),
            patch(
                "qai_hub_models_cli.proto_helpers.manifest.get_manifest",
                return_value=_manifest(),
            ),
            patch.dict("sys.modules", _aihm_sys_modules(mock_cli)),
        )

    def test_cache_miss_writes_file(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        cache_path = tmp_path / "info.pb"
        mock_cli = MagicMock()
        mock_cli.get_info_proto = MagicMock(return_value=_model_info())

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_path, mock_cli))
            result = get_model_info("mobilenet_v2", _DEV_VERSION)

        mock_cli.get_info_proto.assert_called_once_with("mobilenet_v2")
        assert cache_path.exists()
        assert result.id == "mobilenet_v2"

    def test_cache_hit_skips_getter(self, tmp_path: Path) -> None:
        from qai_hub_models_cli.proto_helpers.info import get_model_info

        cache_path = tmp_path / "info.pb"
        cache_path.write_bytes(_model_info().SerializeToString())
        mock_cli = MagicMock()
        mock_cli.get_info_proto = MagicMock()

        with contextlib.ExitStack() as stack:
            _enter_mocks(stack, self._mocks(cache_path, mock_cli))
            result = get_model_info("mobilenet_v2", _DEV_VERSION)

        mock_cli.get_info_proto.assert_not_called()
        assert result.id == "mobilenet_v2"


class TestToolVersionLabels:
    def test_labels_match_proto_fields(self) -> None:
        """_TOOL_VERSION_LABELS must stay 1:1 with the ToolVersions proto fields."""
        from qai_hub_models_cli.proto.shared.tool_versions_pb2 import ToolVersions
        from qai_hub_models_cli.proto_helpers.tool_versions import _TOOL_VERSION_LABELS

        labeled_fields = {field for field, _ in _TOOL_VERSION_LABELS}
        proto_fields = {f.name for f in ToolVersions.DESCRIPTOR.fields}
        assert labeled_fields == proto_fields
