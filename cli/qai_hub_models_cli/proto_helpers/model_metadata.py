# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Helpers for the ``ModelMetadata`` proto published next to each release asset.

The export test uploads ``<zipname>_metadata.proto`` (binary) and
``<zipname>_metadata.json`` beside each ``<zipname>.zip``, so the metadata for
an asset lives at the asset's URL with ``.zip`` swapped for ``_metadata.proto``.
"""

from __future__ import annotations

import posixpath
import shutil
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli.errors import (
    AssetNotFoundError,
    CLIUserError,
    UnsupportedVersionError,
)
from qai_hub_models_cli.fetch import get_asset_url
from qai_hub_models_cli.proto.model_metadata_pb2 import ModelMetadata
from qai_hub_models_cli.proto.shared.precision_pb2 import Precision
from qai_hub_models_cli.proto.shared.runtime_pb2 import Runtime
from qai_hub_models_cli.proto_helpers._common import (
    fetch_proto,
    model_cache_path,
    read_proto,
    read_proto_json,
    read_proto_yaml,
)
from qai_hub_models_cli.proto_helpers.manifest import get_manifest_entry
from qai_hub_models_cli.versions import CURRENT_VERSION

# Longest first, so ".tar.gz" wins over ".gz".
_ARCHIVE_EXTENSIONS = sorted(
    (ext for _, exts, _ in shutil.get_unpack_formats() for ext in exts),
    key=len,
    reverse=True,
)
LAST_VERSION_WITHOUT_METADATA = Version("0.66.0")
METADATA_PROTO_SUFFIX = "_metadata.proto"
METADATA_JSON_SUFFIX = "_metadata.json"


def metadata_url_for_asset(asset_url: str, suffix: str = METADATA_PROTO_SUFFIX) -> str:
    """
    Return the URL of the metadata file published beside an asset zip.

    Parameters
    ----------
    asset_url
        URL (``https://``) or key of the asset.
    suffix
        ``_metadata.proto`` (binary, default) or ``_metadata.json``.

    Returns
    -------
    str
        The asset URL with its file extension replaced by *suffix*.
    """
    for extension in _ARCHIVE_EXTENSIONS:
        if asset_url.endswith(extension):
            return asset_url.removesuffix(extension) + suffix
    return posixpath.splitext(asset_url)[0] + suffix


def get_model_metadata(
    model: str,
    runtime: Runtime.ValueType | str,
    precision: Precision.ValueType | str,
    chipset: str | None = None,
    device: str | None = None,
    version: Version = CURRENT_VERSION,
    sdk_versions: dict[str, str] | None = None,
) -> ModelMetadata:
    """
    Fetch and cache the ``ModelMetadata`` proto of a released model asset.

    Parameters
    ----------
    model
        Model ID (e.g. ``"mobilenet_v2"``) or display name.
    runtime
        Target runtime (e.g. ``RUNTIME_TFLITE`` or ``"tflite"``).
    precision
        Model precision (e.g. ``PRECISION_FLOAT`` or ``"float"``).
    chipset
        Chipset reference. Required for AOT-compiled runtimes unless *device* is given.
    device
        Device name; resolved to its chipset. Mutually exclusive with *chipset*.
    version
        AI Hub Models release version. Defaults to the installed CLI version.
    sdk_versions
        Optional ``{tool: version}`` filter map to disambiguate assets.

    Returns
    -------
    ModelMetadata
        Export configuration, tool versions, and per-file input/output tensor
        specs of the asset.

    Raises
    ------
    AssetNotFoundError
        If the asset has no published metadata.
    CLIUserError
        If the asset selection is invalid or ambiguous.
    UnsupportedVersionError
        If *version* is 0.66.0 or older, which did not publish metadata
        separately from the zips.
    """
    if version <= LAST_VERSION_WITHOUT_METADATA:
        raise UnsupportedVersionError(
            f"Metadata files were not uploaded separately from the zips in version "
            f"{version}. Use a version newer than {LAST_VERSION_WITHOUT_METADATA}."
        )

    asset_url = get_asset_url(
        model=model,
        runtime=runtime,
        precision=precision,
        version=version,
        chipset=chipset,
        device=device,
        sdk_versions=sdk_versions,
    )
    metadata_url = metadata_url_for_asset(asset_url)
    entry = get_manifest_entry(model, version)
    cache_path = model_cache_path(version, entry.id, metadata_url.rsplit("/", 1)[-1])
    try:
        return fetch_proto(metadata_url, cache_path, ModelMetadata)
    except CLIUserError:
        raise
    except Exception as e:
        raise AssetNotFoundError(
            f"No metadata is published for this asset (looked for {metadata_url})."
        ) from e


def read_model_metadata(path: str | Path) -> ModelMetadata:
    """
    Read a ``ModelMetadata`` proto from a local file.

    The format is chosen by file extension: ``.json``, ``.yaml``/``.yml``, or
    binary protobuf otherwise (``.proto``/``.pb``).

    Parameters
    ----------
    path
        Path to the metadata file.

    Returns
    -------
    ModelMetadata
        The parsed metadata.

    Raises
    ------
    CLIUserError
        If *path* is not a file, or it cannot be parsed.
    """
    file = Path(path)
    if not file.is_file():
        raise CLIUserError(f"{file} is not a file.")
    suffix = file.suffix.lower()
    try:
        if suffix == ".json":
            return read_proto_json(file, ModelMetadata)
        if suffix in {".yaml", ".yml"}:
            return read_proto_yaml(file, ModelMetadata)
        return read_proto(file, ModelMetadata)
    except Exception as e:
        raise CLIUserError(f"Could not parse {file} as ModelMetadata: {e}") from e
