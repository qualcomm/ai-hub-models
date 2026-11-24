# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from typing import TypeVar

from google.protobuf.message import Message
from packaging.version import Version
from qai_hub_models_cli.proto.model_manifest_pb2 import ModelManifest
from qai_hub_models_cli.proto.model_numerics_pb2 import ModelNumerics
from qai_hub_models_cli.proto.model_perf_pb2 import ModelPerf
from qai_hub_models_cli.proto.model_release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.platform_pb2 import PlatformInfo
from qai_hub_models_cli.proto.release_manifest_pb2 import (
    ReleaseManifest,
    ReleaseManifestModelEntry,
)
from qai_hub_models_cli.proto.shared.precision_pb2 import PRECISION_FLOAT
from qai_hub_models_cli.proto_helpers._common import read_proto_yaml
from qai_hub_models_cli.versions import CURRENT_VERSION
from tqdm import tqdm

from qai_hub_models._version import __version__
from qai_hub_models.configs.proto_helpers import (
    precision_from_proto,
    runtime_from_proto,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.aws import QAIHM_PRIVATE_S3_BUCKET, get_s3_url
from qai_hub_models.utils.path_helpers import (
    MODEL_IDS,
    PLATFORM_YAML_PATH,
    QAIHM_MODELS_ROOT,
    SCORECARD_MODELS_ROOT,
    SIMILAR_DEVICES_YAML_PATH,
    is_internal_repo,
)

_M = TypeVar("_M", bound=Message)


def _build_release_manifest_entry(
    model_id: str, similar_chipsets: frozenset[str], include_unpublished: bool
) -> ReleaseManifestModelEntry | None:
    """
    Serialized ``ReleaseManifestModelEntry`` for one model, or None if filtered out.

    Module-level and returns bytes so it can run in a worker process.
    """
    manifest = get_model_manifest(model_id, include_internal=True)
    published = (
        manifest._internal.status == ModelManifest.Internal.MODEL_STATUS_PUBLISHED
    )
    if not include_unpublished and not published:
        return None

    assets = _read_model_proto(model_id, "release-assets.yaml", ModelReleaseAssets)
    perf = get_model_perf(model_id)
    precisions = {a.precision for a in assets.assets}
    precisions.update(e.precision for e in perf.performance_metrics)
    runtimes = {a.runtime for a in assets.assets}
    runtimes.update(e.runtime for e in perf.performance_metrics)
    return ReleaseManifestModelEntry(
        id=model_id,
        display_name=manifest.name,
        domain=manifest.domain,
        is_quantized=any(p != PRECISION_FLOAT for p in precisions),
        supported_runtimes=sorted(runtimes),
        supported_chipsets=[
            c for c in perf.supported_chipsets if c not in similar_chipsets
        ],
        tags=manifest.tags,
        use_case=manifest.use_case,
    )


def _build_release_manifest_entry_bytes(
    model_id: str, similar_chipsets: frozenset[str], include_unpublished: bool
) -> bytes | None:
    if entry := _build_release_manifest_entry(
        model_id, similar_chipsets, include_unpublished
    ):
        return entry.SerializeToString()
    return None


def get_release_manifest() -> ReleaseManifest:
    """Build a ReleaseManifest from the local model manifest files."""
    # The scripts package is not present in release builds, so import it here.
    from qai_hub_models.scripts.build_release_proto import _similar_chipsets

    build_entry = partial(
        _build_release_manifest_entry_bytes,
        similar_chipsets=_similar_chipsets(),
        include_unpublished=is_internal_repo(),
    )
    with ProcessPoolExecutor() as pool:
        entries = list(
            tqdm(
                pool.map(build_entry, MODEL_IDS, chunksize=8),
                total=len(MODEL_IDS),
                desc="Building CLI model manifest from current repository ref",
                unit="model",
            )
        )

    return ReleaseManifest(
        version=__version__,
        models=[
            ReleaseManifestModelEntry.FromString(e) for e in entries if e is not None
        ],
    )


def _read_model_proto(model_id: str, filename: str, proto_type: type[_M]) -> _M:
    """Read ``<model dir>/<filename>`` as a proto; empty message if the file is absent."""
    path = QAIHM_MODELS_ROOT / model_id / filename
    if not path.exists():
        return proto_type()
    return read_proto_yaml(path, proto_type)


def get_model_manifest(model_id: str, include_internal: bool = True) -> ModelManifest:
    """
    Read a ModelManifest proto from the model's manifest.yaml.

    With ``include_internal``, the model's internal-config.yaml (if any) is loaded
    into the manifest's ``_internal`` field.
    """
    manifest = _read_model_proto(model_id, "manifest.yaml", ModelManifest)
    manifest.aihm_version = __version__
    if include_internal:
        internal_path = SCORECARD_MODELS_ROOT / model_id / "internal-config.yaml"
        if internal_path.exists():
            manifest._internal.CopyFrom(
                read_proto_yaml(internal_path, ModelManifest.Internal)
            )
    return manifest


def get_model_perf(model_id: str) -> ModelPerf:
    """Read a ModelPerf proto from the model's perf.yaml."""
    return _read_model_proto(model_id, "perf.yaml", ModelPerf)


def get_model_numerics(model_id: str) -> ModelNumerics:
    """Read a ModelNumerics proto from the model's numerics.yaml."""
    numerics = _read_model_proto(model_id, "numerics.yaml", ModelNumerics)
    if not numerics.metrics:
        return ModelNumerics()

    # Numerics carries no tool versions; borrow them from the matching perf entry.
    perf = get_model_perf(model_id)
    tool_versions = {
        (e.device, e.precision, e.runtime): e.tool_versions
        for e in perf.performance_metrics
    }
    numerics.aihm_version = __version__
    numerics.model_id = model_id
    for metric in numerics.metrics:
        for dm in metric.device_metrics:
            versions = tool_versions.get((dm.device, dm.precision, dm.runtime))
            if versions is not None:
                dm.tool_versions.CopyFrom(versions)
    return numerics


@cache
def get_platform(*, _include_similar_devices: bool = False) -> PlatformInfo:
    """
    Read the platform registry from platform.yaml (dev installs).

    The result is cached and shared, so callers must not mutate it.

    Parameters
    ----------
    _include_similar_devices
        If True, also add the devices and chipsets from similar_devices.yaml.
        These are never written to platform.yaml or published.

    Returns
    -------
    platform : PlatformInfo
        The platform registry, stamped with the installed version.
    """
    platform = read_proto_yaml(PLATFORM_YAML_PATH, PlatformInfo)
    platform.aihm_version = __version__
    if _include_similar_devices:
        similar = read_proto_yaml(SIMILAR_DEVICES_YAML_PATH, PlatformInfo)
        platform.devices.extend(similar.devices)
        platform.chipsets.extend(similar.chipsets)
    return platform


def _resolve_download_url(
    asset: ModelReleaseAssets.AssetDetails, model_id: str, version: str, internal: bool
) -> str:
    if asset.download_url and internal:
        return asset.download_url
    if not internal:
        return asset.download_url or ASSET_CONFIG.get_release_asset_url(
            model_id,
            version,
            runtime_from_proto(asset.runtime),
            precision_from_proto(asset.precision),
            asset.chipset if asset.HasField("chipset") else None,
        )
    if asset.HasField("s3_key"):
        return get_s3_url(QAIHM_PRIVATE_S3_BUCKET, asset.s3_key)
    return ""


def get_model_release_assets(
    model_id: str, _set_url_version: Version | str | None = None
) -> ModelReleaseAssets | None:
    """
    Read a ModelReleaseAssets proto from the model's release-assets.yaml.

    Parameters
    ----------
    model_id
        Model ID.
    _set_url_version
        If set, replace s3 bucket URLs with URLs that point to this version.

    Returns
    -------
    assets : ModelReleaseAssets | None
        The assets, or None if the model has none.
    """
    # The scorecard package is not available in release builds, so import here.
    from qai_hub_models.scorecard.device import compute_peer_chipsets

    source = _read_model_proto(model_id, "release-assets.yaml", ModelReleaseAssets)
    if not source.assets:
        return None

    version = str(_set_url_version or CURRENT_VERSION)
    internal = is_internal_repo() and _set_url_version is not None

    # Resolve URLs before copying to peers, which share one uploaded artifact.
    assets = list(source.assets)
    for asset in assets:
        asset.download_url = _resolve_download_url(asset, model_id, version, internal)
    present = {(a.precision, a.chipset) for a in assets if a.HasField("chipset")}
    for asset in list(assets):
        if not asset.HasField("chipset"):
            continue
        for peer in compute_peer_chipsets(asset.chipset) - {asset.chipset}:
            if (asset.precision, peer) not in present:
                peer_asset = ModelReleaseAssets.AssetDetails()
                peer_asset.CopyFrom(asset)
                peer_asset.chipset = peer
                assets.append(peer_asset)

    out = ModelReleaseAssets(aihm_version=version, model_id=model_id)
    for asset in assets:
        if not asset.download_url:
            continue
        entry = out.assets.add()
        entry.CopyFrom(asset)
        entry.ClearField("s3_key")
        entry.tool_versions.SetInParent()
    return out
