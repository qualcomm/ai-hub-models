# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable, Iterable

from packaging.version import Version

from qai_hub_models_cli.proto.release_assets_pb2 import ModelReleaseAssets
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.platform import (
    get_platform,
    resolve_chipset,
    resolve_device,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_str_to_proto,
    runtime_str_to_proto,
)
from qai_hub_models_cli.proto_helpers.release_assets import (
    filter_release_assets,
    get_model_release_assets,
)
from qai_hub_models_cli.proto_helpers.tool_versions import validate_sdk_tools
from qai_hub_models_cli.versions import (
    MIN_MANIFEST_VERSION,
    UnsupportedVersionError,
    get_supported_versions,
)


def find_in_version(
    model: str,
    version: Version,
    runtime: Runtime.ValueType | str | None = None,
    precision: Precision.ValueType | str | None = None,
    chipset: str | None = None,
    device: str | None = None,
    sdk_versions: dict[str, str] | None = None,
) -> tuple[ModelReleaseAssets | None, str | None]:
    """
    Search a single release for assets of *model* matching the filters.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name.
    version
        AI Hub Models release version to search.
    runtime
        Runtime to filter on (e.g. ``"tflite"``).
    precision
        Precision to filter on (e.g. ``"float"``).
    chipset
        Chipset reference (ID, name, or alias). Mutually exclusive with *device*.
    device
        Device name; resolved to its chipset. Mutually exclusive with *chipset*.
    sdk_versions
        ``{tool: version}`` filter map (see
        :func:`qai_hub_models_cli.common.parse_sdk_version_filters`).

    Returns
    -------
    tuple[ModelReleaseAssets | None, str | None]
        ``(assets, skip_reason)``. On a match, *assets* is the matching assets
        and *skip_reason* is None. Otherwise *assets* is None and *skip_reason*
        is why the release was skipped (model/chipset/device not valid for it, or
        it predates the asset manifest), or None for a plain miss.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    """
    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")

    try:
        assets = get_model_release_assets(model, version)
    except UnsupportedVersionError:
        return None, "release predates the asset manifest"
    except (FileNotFoundError, KeyError):
        # Model is absent from this release (or has no published assets).
        return None, f"model {model!r} is not in this release"
    platform = get_platform(version)

    # Check the chipset/device against this release explicitly, so an unknown
    # one is reported as a skip reason rather than surfacing as a filter error.
    if chipset is not None:
        try:
            resolve_chipset(platform.chipsets, platform.devices, chipset=chipset)
        except KeyError:
            return None, f"chipset {chipset!r} is not in this release"
    if device is not None:
        try:
            resolve_device(platform.devices, device)
        except KeyError:
            return None, f"device {device!r} is not in this release"

    matched = filter_release_assets(
        assets,
        platform,
        runtime=runtime,
        precision=precision,
        chipset=chipset,
        device=device,
        sdk_versions=sdk_versions,
    )
    if matched.assets:
        return matched, None
    return None, None


def default_search_versions(
    min_version: Version | None = None,
    max_version: Version | None = None,
) -> list[Version]:
    """Return the releases ``find`` searches by default, newest-first."""
    if (
        min_version is not None
        and max_version is not None
        and min_version > max_version
    ):
        raise ValueError(
            f"--min-version ({min_version}) must be <= --max-version ({max_version})."
        )

    def _in_bounds(v: Version) -> bool:
        return (
            v >= MIN_MANIFEST_VERSION
            and (min_version is None or v >= min_version)
            and (max_version is None or v <= max_version)
        )

    return [v for v in get_supported_versions() if _in_bounds(v)]


def find_matching_releases(
    model: str,
    runtime: Runtime.ValueType | str | None = None,
    precision: Precision.ValueType | str | None = None,
    chipset: str | None = None,
    device: str | None = None,
    sdk_versions: dict[str, str] | None = None,
    versions: Iterable[Version] | None = None,
    min_version: Version | None = None,
    max_version: Version | None = None,
    first_only: bool = False,
    progress: Callable[[Version, bool, str | None], None] | None = None,
) -> list[tuple[Version, ModelReleaseAssets]]:
    """
    Search releases for assets of *model* matching the given fetch filters.

    Releases are searched newest-first. The filters mirror
    :func:`qai_hub_models_cli.fetch.fetch`; any left as ``None`` is not applied.

    Parameters
    ----------
    model
        Model ID or display name.
    runtime
        Runtime to filter on (see :func:`find_in_version`).
    precision
        Precision to filter on (see :func:`find_in_version`).
    chipset
        Chipset reference to filter on. Mutually exclusive with *device*.
    device
        Device name to filter on. Mutually exclusive with *chipset*.
    sdk_versions
        ``{tool: version}`` filter map (see :func:`find_in_version`).
    versions
        Explicit releases to search, newest-first. When given, *min_version* and
        *max_version* are ignored. Defaults to :func:`default_search_versions`.
    min_version
        If given (and *versions* is not), exclude releases older than this.
    max_version
        If given (and *versions* is not), exclude releases newer than this.
    first_only
        If True, stop and return after the first (newest) matching release.
    progress
        Optional callback ``(version, found, skip_reason)`` invoked per release.

    Returns
    -------
    list[tuple[Version, ModelReleaseAssets]]
        ``(version, matching_assets)`` pairs, newest-first.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided, or an SDK tool name is
        unknown.
    KeyError
        If *runtime* or *precision* is not a known value.
    """
    # find_in_version re-checks this per release, but validate up front so the
    # error is raised even when no release ends up being searched (empty
    # versions), and before any per-release network work begins.
    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")

    # find_in_version swallows lookup errors per release, so validate the
    # release-agnostic filters (runtime/precision tokens, SDK tool names) once up
    # front — otherwise a typo would silently match nothing in every release.
    # Chipset/device membership can vary by release and is checked per release.
    if runtime is not None:
        runtime_str_to_proto(runtime)
    if precision is not None:
        precision_str_to_proto(precision)
    if sdk_versions:
        validate_sdk_tools(sdk_versions)

    if versions is None:
        versions = default_search_versions(min_version, max_version)

    results: list[tuple[Version, ModelReleaseAssets]] = []
    for version in versions:
        matched, skip_reason = find_in_version(
            model, version, runtime, precision, chipset, device, sdk_versions
        )
        if progress is not None:
            progress(version, matched is not None, skip_reason)
        if matched is not None:
            results.append((version, matched))
            if first_only:
                break
    return results
