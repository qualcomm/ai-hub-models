# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from packaging.version import Version

from qai_hub_models_cli.proto.platform_pb2 import (
    ChipsetInfo,
    DeviceInfo,
    FormFactor,
    OperatingSystem,
    PlatformInfo,
    RuntimeInfo,
    WebsiteWorld,
)
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers._common import fetch_release_proto
from qai_hub_models_cli.proto_helpers.manifest import get_manifest
from qai_hub_models_cli.proto_helpers.platform_enums import (
    form_factor_proto_to_str,
    normalize_label,
    os_proto_to_str,
    runtime_proto_to_str,
    runtime_str_to_proto,
    world_proto_to_str,
)
from qai_hub_models_cli.utils import build_table
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    MIN_MODEL_FILTER_VERSION,
    feature_supported,
)

# Column headers for the chipset attributes shared by the `chipsets` and
# `devices` CLI tables, in display order. Kept here so both tables stay aligned.
CHIPSET_ATTRIBUTE_HEADERS: list[str] = [
    "FP16",
    "HTP Version",
    "SoC Model",
]


_T = TypeVar("_T")


def _as_list(value: _T | list[_T]) -> list[_T]:
    """Wrap a single value in a list, leaving an existing list unchanged."""
    return value if isinstance(value, list) else [value]


def _os_matches(device_os: OperatingSystem, query: OperatingSystem) -> bool:
    """
    True if *device_os* satisfies the *query* OS filter.

    The OS type must match. A query with no ``version`` matches any version;
    a query with a ``version`` must match exactly.
    """
    if device_os.ostype != query.ostype:
        return False
    return not query.version or device_os.version == query.version


@functools.lru_cache(maxsize=1)
def get_platform(
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> PlatformInfo:
    """
    Fetch and cache the platform info protobuf for a given version.

    Contains the registry of devices, chipsets, form factors, and
    runtimes supported by AI Hub.

    Parameters
    ----------
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local platform protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    PlatformInfo
        Parsed platform info protobuf.

    Raises
    ------
    UnsupportedVersionError
        If *version* is not a supported release (when *local_path* is None).
    """
    if local_path is not None:
        url = None
    else:
        manifest = get_manifest(version)
        url = manifest.platform_url
    return fetch_release_proto(
        version,
        PlatformInfo,
        cache_filename="platform.pb",
        source_getter="get_platform_proto",
        url=url,
        local_path=local_path,
    )


def resolve_runtime(
    runtimes: Iterable[RuntimeInfo],
    runtime: Runtime.ValueType | str,
) -> RuntimeInfo:
    """
    Look up the ``RuntimeInfo`` for a given runtime.

    Parameters
    ----------
    runtimes
        Runtime registry (``platform.runtimes``) to look the runtime up in.
    runtime
        Runtime enum value (e.g. ``RUNTIME_TFLITE``), token (e.g. ``"tflite"``),
        or display name (e.g. ``"TensorFlow Lite"``).

    Returns
    -------
    RuntimeInfo
        Platform runtime entry containing ``is_aot_compiled``,
        ``file_extension``, etc.

    Raises
    ------
    KeyError
        If *runtime* is not a known runtime.
    """
    runtimes = list(runtimes)
    runtime_val = runtime_str_to_proto(runtime, runtimes)
    for rt in runtimes:
        if rt.runtime == runtime_val:
            return rt
    runtime_name = (
        runtime if isinstance(runtime, str) else runtime_proto_to_str(runtime)
    )
    raise KeyError(f"Unknown runtime {runtime_name!r}.")


def normalize_hw_name(name: str) -> str:
    """Normalize a device or chipset name for lenient matching.

    Case-folds and unifies separators (via :func:`normalize_label`), drops
    trademark symbols (``®``/``™``/``©``), and collapses whitespace. So
    ``"Snapdragon® 8 Gen 1 Mobile"``, ``"snapdragon 8 gen 1 mobile"``, and
    ``"snapdragon_8-gen-1_mobile"`` all normalize to the same key.
    """
    for symbol in ("®", "™", "©"):
        name = name.replace(symbol, " ")
    return " ".join(normalize_label(name).split())


def resolve_device(devices: Iterable[DeviceInfo], device: str) -> DeviceInfo:
    """
    Look up a device by name, matched leniently (see :func:`normalize_hw_name`).

    Parameters
    ----------
    devices
        Device registry (``platform.devices``) to resolve against.
    device
        A device name (e.g. ``"Samsung Galaxy S24"``).

    Returns
    -------
    DeviceInfo
        The matching device.

    Raises
    ------
    KeyError
        If *device* does not match any known device.
    """
    device_key = normalize_hw_name(device)
    for d in devices:
        if normalize_hw_name(d.name) == device_key:
            return d
    raise KeyError(
        f"Unknown device {device!r}. "
        "Run `qai-hub-models devices` to see supported devices."
    )


def resolve_chipset(
    chipsets: Iterable[ChipsetInfo],
    devices: Iterable[DeviceInfo],
    device: str | None = None,
    chipset: str | None = None,
) -> ChipsetInfo:
    """
    Resolve a device or chipset reference to its ``ChipsetInfo``.

    Exactly one of *device* or *chipset* must be provided.

    Parameters
    ----------
    chipsets
        Chipset registry (``platform.chipsets``) to resolve against.
    devices
        Device registry (``platform.devices``), used when resolving by *device*.
    device
        A device name (e.g. ``"Samsung Galaxy S24"``). Resolved to the chipset
        of that device. Matched case-insensitively.
    chipset
        A chipset reference: its canonical ID (e.g.
        ``"qualcomm-snapdragon-8-gen-3"``), name (e.g.
        ``"Snapdragon 8 Gen 3"``), or one of its aliases. Matched
        case-insensitively.

    Returns
    -------
    ChipsetInfo
        The matching chipset proto.

    Raises
    ------
    ValueError
        If neither or both of *device* and *chipset* are provided.
    KeyError
        If the reference does not match any known device or chipset.
    """
    if (device is None) == (chipset is None):
        raise ValueError("Provide exactly one of 'device' or 'chipset'.")

    chipsets_by_name = {c.name: c for c in chipsets}

    if device is not None:
        d = resolve_device(devices, device)  # raises if the device is unknown
        if d.chipset in chipsets_by_name:
            return chipsets_by_name[d.chipset]
        raise KeyError(f"Device {d.name!r} references unknown chipset {d.chipset!r}.")

    assert chipset is not None  # exactly one of device/chipset, checked above
    chipset_key = normalize_hw_name(chipset)
    for c in chipsets_by_name.values():
        candidates = [c.name, c.marketing_name, *c.aliases]
        if any(normalize_hw_name(candidate) == chipset_key for candidate in candidates):
            return c
    raise KeyError(
        f"Unknown chipset {chipset!r}. "
        "Run `qai-hub-models chipsets` to see supported chipsets."
    )


def device_names_for_filter(
    chipsets: Iterable[ChipsetInfo],
    devices: Iterable[DeviceInfo],
    chipset: str | list[str] | None,
    device: str | list[str] | None,
) -> set[str] | None:
    """Expand a ``--chipset``/``--device`` filter into the device names it matches.

    Pass one of *chipset* or *device* (each a name or list of names); the other
    must be ``None``. Returns lowercased device names to match records against:

    - ``device``: the named device(s) themselves.
    - ``chipset``: every device that runs on the named chipset(s).
    - neither: ``None``, meaning "no device filter".

    Every name is validated against *chipsets*/*devices*; an unknown one raises
    ``KeyError`` pointing at the ``devices``/``chipsets`` commands.
    """
    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")
    if device is not None:
        names = [device] if isinstance(device, str) else device
        for name in names:
            resolve_chipset(chipsets, devices, device=name)  # validate; raises
        return {name.lower() for name in names}
    if chipset is not None:
        devices = list(devices)
        refs = [chipset] if isinstance(chipset, str) else chipset
        chipset_ids = {
            resolve_chipset(chipsets, devices, chipset=ref).name for ref in refs
        }
        return {d.name.lower() for d in devices if d.chipset in chipset_ids}
    return None


def filter_devices(
    devices: Iterable[DeviceInfo],
    chipsets: Iterable[ChipsetInfo],
    *,
    form_factor: FormFactor.ValueType | list[FormFactor.ValueType] | None = None,
    os: OperatingSystem | list[OperatingSystem] | None = None,
    fp16: bool | None = None,
    htp_version: int | list[int] | None = None,
    soc_model: int | list[int] | None = None,
) -> list[DeviceInfo]:
    """
    Filter *devices* by the given criteria, preserving input order.

    Chipset-derived filters (*fp16*, *htp_version*, *soc_model*) are resolved
    through each device's chipset, looked up in *chipsets*.

    Parameters
    ----------
    devices
        Devices to filter.
    chipsets
        Chipsets used to resolve chipset-derived filters.
    form_factor
        Keep only devices of this form factor (or one of several, if a list).
    os
        Keep only devices matching this operating system (or one of several, if
        a list). An ``OperatingSystem`` with no ``version`` matches any version
        of that OS type; with a ``version`` it must match exactly.
    fp16
        If True, keep only devices whose chipset supports fp16.
    htp_version
        Keep only devices whose chipset has this HTP version.
    soc_model
        Keep only devices whose chipset has this SoC model.

    Returns
    -------
    list[DeviceInfo]
        Matching device protos.
    """
    chipsets_by_name = {c.name: c for c in chipsets}

    # Normalize each filter to its comparison form once, up front.
    form_factors = _as_list(form_factor) if form_factor is not None else None
    oses = _as_list(os) if os is not None else None
    htps = _as_list(htp_version) if htp_version is not None else None
    socs = _as_list(soc_model) if soc_model is not None else None
    chipset_filtered = fp16 is not None or htps is not None or socs is not None

    def matches(device: DeviceInfo) -> bool:
        if form_factors is not None and device.form_factor not in form_factors:
            return False
        if oses is not None and not any(_os_matches(device.os, o) for o in oses):
            return False
        if chipset_filtered:
            chipset = chipsets_by_name.get(device.chipset)
            if chipset is None:
                return False
            if fp16 is not None and chipset.supports_fp16 != fp16:
                return False
            if htps is not None and chipset.htp_version not in htps:
                return False
            if socs is not None and chipset.soc_model not in socs:
                return False
        return True

    return [d for d in devices if matches(d)]


def filter_chipsets(
    chipsets: Iterable[ChipsetInfo],
    *,
    world: WebsiteWorld.ValueType | list[WebsiteWorld.ValueType] | None = None,
    fp16: bool | None = None,
    htp_version: int | list[int] | None = None,
    soc_model: int | list[int] | None = None,
) -> list[ChipsetInfo]:
    """
    Filter *chipsets* by the given criteria, preserving input order.

    Parameters
    ----------
    chipsets
        Chipsets to filter.
    world
        Keep only chipsets of this type (``WebsiteWorld``), or one of several
        if a list.
    fp16
        If True, keep only chipsets that support fp16.
    htp_version
        Keep only chipsets with this HTP version.
    soc_model
        Keep only chipsets with this SoC model.

    Returns
    -------
    list[ChipsetInfo]
        Matching chipset protos.
    """
    worlds = _as_list(world) if world is not None else None
    htps = _as_list(htp_version) if htp_version is not None else None
    socs = _as_list(soc_model) if soc_model is not None else None

    def matches(chipset: ChipsetInfo) -> bool:
        if worlds is not None and chipset.world not in worlds:
            return False
        if fp16 is not None and chipset.supports_fp16 != fp16:
            return False
        if htps is not None and chipset.htp_version not in htps:
            return False
        return not (socs is not None and chipset.soc_model not in socs)

    return [c for c in chipsets if matches(c)]


def chipset_attribute_row(chipset: ChipsetInfo | None) -> list[str]:
    """
    Render the shared chipset attribute columns for one chipset.

    Returns cells in the same order as ``CHIPSET_ATTRIBUTE_HEADERS``. A ``None``
    chipset yields empty cells so table rows stay aligned.
    """
    if chipset is None:
        return [""] * len(CHIPSET_ATTRIBUTE_HEADERS)
    return [
        "Yes" if chipset.supports_fp16 else "No",
        str(chipset.htp_version) if chipset.htp_version else "",
        str(chipset.soc_model) if chipset.soc_model else "",
    ]


def format_devices_table(
    devices: Iterable[DeviceInfo],
    chipsets: Iterable[ChipsetInfo],
    title: str | None = "Devices",
) -> str:
    """Format a table of devices with their chipset attributes."""
    chipsets_by_name = {c.name: c for c in chipsets}

    def chipset_name(chipset_id: str) -> str:
        chipset = chipsets_by_name.get(chipset_id)
        return chipset.marketing_name if chipset else chipset_id

    def device_type(device: DeviceInfo) -> str:
        # A device's type is the "world" of its chipset, falling back to the
        # device's own form factor when the chipset is not in the registry.
        chipset = chipsets_by_name.get(device.chipset)
        if chipset is not None:
            return world_proto_to_str(chipset.world)
        return form_factor_proto_to_str(device.form_factor)

    return build_table(
        ["Type", "Name", "OS", "Chipset", *CHIPSET_ATTRIBUTE_HEADERS],
        [
            [
                device_type(d),
                d.name,
                os_proto_to_str(d.os),
                chipset_name(d.chipset),
                *chipset_attribute_row(chipsets_by_name.get(d.chipset)),
            ]
            for d in devices
        ],
        wrap_column="Name",
        title=title,
    )


def format_chipsets_table(
    chipsets: Iterable[ChipsetInfo], title: str | None = "Chipsets"
) -> str:
    """Format a table of chipsets with their attributes."""
    return build_table(
        ["Type", "Name", "Aliases", *CHIPSET_ATTRIBUTE_HEADERS],
        [
            [
                world_proto_to_str(c.world),
                c.marketing_name,
                ", ".join(c.aliases),
                *chipset_attribute_row(c),
            ]
            for c in chipsets
        ],
        wrap_column="Name",
        title=title,
    )


def format_runtimes_table(
    runtimes: Iterable[RuntimeInfo],
    version: Version,
    title: str | None = "Runtimes (Compilation Targets)",
) -> str:
    """Format a table of runtimes with their display details.

    The display-metadata columns (Name, Description) are sourced from
    ``RuntimeInfo`` fields only populated as of ``MIN_MODEL_FILTER_VERSION``;
    for older platforms they are omitted rather than shown blank.
    """
    # Docs URLs are long, unwrappable tokens; keeping them in the table would
    # squeeze every other column. List them via format_runtime_links instead.
    has_metadata = feature_supported(version, MIN_MODEL_FILTER_VERSION)
    columns = ["ID"]
    if has_metadata:
        columns += ["Name", "Description"]
    columns += ["Ext", "Compiled"]

    rows = []
    for rt in runtimes:
        row = [runtime_proto_to_str(rt.runtime)]
        if has_metadata:
            row += [rt.display_name, rt.description]
        row += [
            rt.file_extension,
            "Ahead-of-Time" if rt.is_aot_compiled else "On-Device",
        ]
        rows.append(row)

    return build_table(
        columns,
        rows,
        wrap_column="Description" if has_metadata else "ID",
        title=title,
    )


def format_runtime_links(runtimes: Iterable[RuntimeInfo]) -> str:
    """Format a ``Learn more`` footnote of per-runtime docs URLs (or "" if none)."""
    links = [
        (rt.display_name or runtime_proto_to_str(rt.runtime), rt.documentation_url)
        for rt in runtimes
        if rt.documentation_url
    ]
    if not links:
        return ""
    width = max(len(name) for name, _ in links)
    body = "\n".join(f"  {name:<{width}}  {url}" for name, url in links)
    return f"Learn more:\n{body}"
