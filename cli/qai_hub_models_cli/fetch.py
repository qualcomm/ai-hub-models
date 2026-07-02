# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import requests
from packaging.version import Version

from qai_hub_models_cli.common import (
    AIHUB_MODELS_URL,
    ASSET_FOLDER,
    STORE_URL,
    sample_command,
)
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers.platform import (
    get_platform,
    resolve_runtime,
)
from qai_hub_models_cli.proto_helpers.platform_enums import (
    precision_proto_to_str,
    runtime_proto_to_str,
)
from qai_hub_models_cli.proto_helpers.release_assets import (
    AssetNotFoundError,
    filter_release_assets,
    format_fetch_commands,
    format_release_assets_table,
    get_model_release_assets,
)
from qai_hub_models_cli.utils import download, get_next_free_path
from qai_hub_models_cli.versions import (
    CURRENT_VERSION,
    MIN_MANIFEST_VERSION,
    UnsupportedVersionError,
    version_flag,
)

ASSET_FILENAME = "{model_id}-{runtime}-{precision}.zip"
ASSET_CHIPSET_FILENAME = (
    "{model_id}-{runtime}-{precision}-{chipset_with_underscores}.zip"
)


def _normalize_runtime(runtime: Runtime.ValueType | str) -> str:
    if isinstance(runtime, int):
        return runtime_proto_to_str(runtime)
    return runtime


def _normalize_precision(precision: Precision.ValueType | str) -> str:
    if isinstance(precision, int):
        return precision_proto_to_str(precision)
    return precision


def _asset_url(
    model_id: str,
    runtime: str,
    precision: str,
    version: Version,
    chipset: str | None = None,
) -> tuple[str, str]:
    """Return (url, filename) for the asset."""
    model_id = model_id.lower()
    runtime_str = runtime.lower()
    precision_str = precision.lower()
    ver = str(version)
    if chipset is not None:
        filename = ASSET_CHIPSET_FILENAME.format(
            model_id=model_id,
            runtime=runtime_str,
            precision=precision_str,
            chipset_with_underscores=chipset.lower().replace("-", "_"),
        )
    else:
        filename = ASSET_FILENAME.format(
            model_id=model_id,
            runtime=runtime_str,
            precision=precision_str,
        )
    folder = ASSET_FOLDER.format(model_id=model_id, version=ver)
    url = f"{STORE_URL}/{folder}/{filename}"
    return url, filename


def get_asset_url(
    *,
    model: str,
    runtime: Runtime.ValueType | str | None,
    precision: Precision.ValueType | str | None,
    version: Version = CURRENT_VERSION,
    chipset: str | None = None,
    device: str | None = None,
    quiet: bool = False,
    url_only: bool = False,
    sdk_versions: dict[str, str] | None = None,
) -> str:
    """
    Resolve the download URL for a model asset.

    The asset is selected by filtering the model's release assets by the
    provided fields. A download must resolve to exactly one asset, so enough
    fields must be given to uniquely identify it: *runtime* and *precision* are
    always required, and AOT-compiled runtimes additionally require a *chipset*
    or *device*.

    Parameters
    ----------
    model
        Model Name or ID (e.g. ``"mobilenet_v2"``).
    runtime
        Target runtime (e.g. ``RUNTIME_TFLITE`` or ``"tflite"``). Required.
    precision
        Model precision (e.g. ``PRECISION_FLOAT`` or ``"float"``). Required.
    version
        AI Hub Models version.
    chipset
        Optional chipset reference: canonical ID, name, or alias.
        Resolved to the canonical chipset ID. Required for AOT-compiled runtimes.
    device
        Optional device name to select the asset by; resolved to its chipset.
        Mutually exclusive with *chipset*. Required for AOT-compiled runtimes.
    quiet
        Only affects error output. If True, a no-single-match error omits the
        assets table and command hints, leaving only a terse reason. Has no
        effect on the returned URL on success.
    url_only
        Only affects error output. If True, the suggested download command in
        a no-single-match error keeps the ``--url-only`` flag.
    sdk_versions
        Optional ``{tool: version}`` filter map (see
        :func:`parse_sdk_version_filters`) used to disambiguate assets that
        differ only by tool version.

    Returns
    -------
    str
        URL for the asset that exists.

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided, or if a required field
        (*runtime*, *precision*, or a *chipset*/*device* for AOT runtimes) is
        missing.
    KeyError
        If *runtime*, *precision*, *chipset*, or *device* is not known.
    FileNotFoundError
        If the asset does not exist on the server.
    """
    # Hint shown on every failure: how to list all available assets.
    show_all = (
        f"Run `{sample_command('fetch', model, version_flag(version), '-i')}` "
        "to see all available assets."
    )

    if chipset is not None and device is not None:
        raise ValueError("Provide at most one of 'chipset' or 'device'.")

    if version >= MIN_MANIFEST_VERSION:
        release_assets = get_model_release_assets(model, version)
        platform = get_platform(version)

        # Filter the assets down to the user's args.
        matches = filter_release_assets(
            release_assets, platform, runtime, precision, chipset, device, sdk_versions
        )

        # A download must resolve to exactly one asset.
        if len(matches.assets) == 1:
            return matches.assets[0].download_url

        # Nothing matched: the user's filters don't correspond to any asset.
        if not matches.assets:
            raise AssetNotFoundError(
                f"No asset found for model={model!r} with runtime={runtime!r}, "
                f"precision={precision!r}, chipset={chipset!r}, device={device!r}.\n"
                f"{show_all}"
            )

        # Several assets match, so the request is ambiguous. Explain why, then
        # show the matching options. The table is a subset when the filters
        # excluded some assets.
        if runtime is None:
            reason = "A runtime is required to fetch a model asset."
        elif precision is None:
            reason = "A precision is required to fetch a model asset."
        elif (
            resolve_runtime(platform.runtimes, runtime).is_aot_compiled
            and chipset is None
            and device is None
        ):
            # AOT-compiled runtimes produce chipset-specific assets, so a
            # chipset or device is needed to pick exactly one.
            reason = (
                f"A chipset or device is required for AOT-compiled runtime {runtime!r}."
            )
        else:
            reason = (
                f"{len(matches.assets)} assets match your filters. "
                "Narrow them down so exactly one matches."
            )

        # In quiet mode, surface only the terse reason.
        if quiet:
            raise AssetNotFoundError(reason)

        table = format_release_assets_table(
            matches, platform.chipsets, runtimes=platform.runtimes
        )
        commands = format_fetch_commands(
            release_assets,
            model,
            subset=len(matches.assets) < len(release_assets.assets),
            runtime=runtime if isinstance(runtime, str) else None,
            precision=precision if isinstance(precision, str) else None,
            chipset=chipset,
            device=device,
            sdk_versions=sdk_versions,
            url_only=url_only,
            version=version,
        )
        raise AssetNotFoundError(
            f"{reason}\n\nAssets that match your current selection(s):\n{table}"
            f"\n\n{commands}"
        )
    if device is not None:
        raise UnsupportedVersionError(
            f"Device requires version {MIN_MANIFEST_VERSION} or later; provide a chipset instead."
        )

    # Legacy: No manifest was published for these releases, so we can't list
    # assets to filter; runtime and precision must be specified directly.
    if runtime is None:
        raise ValueError(f"A runtime is required to fetch a model asset.\n{show_all}")
    if precision is None:
        raise ValueError(f"A precision is required to fetch a model asset.\n{show_all}")

    def _head(url: str) -> int:
        resp = requests.head(url, timeout=10)
        if resp.status_code not in (200, 403, 404):
            raise ConnectionError(
                f"Unexpected response checking asset availability "
                f"(status {resp.status_code})."
            )
        return resp.status_code

    runtime_s = _normalize_runtime(runtime)
    precision_s = _normalize_precision(precision)
    if chipset is not None:
        url, _ = _asset_url(model, runtime_s, precision_s, version, chipset)
        if _head(url) == 200:
            return url

    url, _ = _asset_url(model, runtime_s, precision_s, version)
    if _head(url) == 200:
        return url

    chipset_msg = f", chipset={chipset!r}" if chipset else ""
    raise FileNotFoundError(
        f"No asset found for model={model!r}, runtime={runtime!r}, "
        f"precision={precision!r}, version={version}{chipset_msg}.\n"
        f"  - Browse available models: {AIHUB_MODELS_URL}\n"
        "  - List valid devices/chipsets: qai-hub list-devices (from the qai_hub package)"
    )


def fetch(
    *,
    model: str,
    runtime: Runtime.ValueType | str | None,
    output_dir: str | os.PathLike,
    precision: Precision.ValueType | str | None = None,
    chipset: str | None = None,
    device: str | None = None,
    version: Version = CURRENT_VERSION,
    extract: bool = False,
    quiet: bool = False,
    sdk_versions: dict[str, str] | None = None,
) -> Path:
    """
    Download a pre-compiled model asset from AI Hub Models.

    If a chipset is provided, the chipset-specific asset is tried first.
    If that does not exist, falls back to the generic asset.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``).
    runtime
        Target runtime (e.g. ``RUNTIME_TFLITE`` or ``"tflite"``).
    output_dir
        Output directory.
    precision
        Model precision (e.g. ``PRECISION_FLOAT`` or ``"float"``). When None,
        it is treated as an unset filter (see :func:`get_asset_url`).
    chipset
        Chipset name for device-specific (AOT compiled) runtimes.
    device
        Device name (e.g. ``"Samsung Galaxy S24"``) for device-specific (AOT compiled) runtimes.
        Mutually exclusive with *chipset*.
    version
        AI Hub Models version. Defaults to the installed CLI version.
    extract
        If True, extract the downloaded zip archive.
    quiet
        If True, suppress all output (progress bar, warnings, retry messages).
    sdk_versions
        Optional ``{tool: version}`` filter map (see
        :func:`parse_sdk_version_filters`) to disambiguate assets that differ
        only by tool version.

    Returns
    -------
    Path
        Path to the downloaded file (or extraction directory if *extract* is True).

    Raises
    ------
    ValueError
        If both *chipset* and *device* are provided.
    FileNotFoundError
        If the asset does not exist on the server.
    """
    url = get_asset_url(
        model=model,
        runtime=runtime,
        precision=precision,
        version=version,
        chipset=chipset,
        device=device,
        quiet=quiet,
        sdk_versions=sdk_versions,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = url.removeprefix("s3://").rsplit("/", 1)[-1]
    if extract:
        dst = get_next_free_path(out / Path(filename).stem)
    else:
        dst = get_next_free_path(out / filename)
    return download(url, dst, extract=extract, quiet=quiet)
