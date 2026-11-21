# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools
import posixpath
from pathlib import Path

from packaging.version import Version

from qai_hub_models_cli._internal.utils import use_internal_releases
from qai_hub_models_cli.common import STORE_URL
from qai_hub_models_cli.errors import CLIUnknownNameError
from qai_hub_models_cli.proto.release_manifest_pb2 import (
    ReleaseManifest,
    ReleaseManifestModelEntry,
)
from qai_hub_models_cli.proto_helpers._common import (
    fetch_release_proto,
    use_aihm_source,
)
from qai_hub_models_cli.proto_helpers.platform_enums import normalize_label
from qai_hub_models_cli.versions import CURRENT_VERSION


def resolve_model_id(model: str, version: Version = CURRENT_VERSION) -> str:
    """Converts the model (name or ID) to the model's unique ID."""
    if use_aihm_source(version):
        # In-source builds must rebuild the entire release manifest to do correct
        # model name -> ID lookups. To construct each entry in the release manifest,
        # several model YAML files need to be read. This is very expensive.
        #
        # Instead, we load the current MODEL_IDs and attempt to do a fast lookup first.
        from qai_hub_models.utils.path_helpers import MODEL_IDS

        # Maximize our chance at a "ID cache hit" by applying common name -> ID mappings.
        model_sanitized = model.replace("-", "_").replace(".", "_").lower()
        if model_sanitized in MODEL_IDS:
            return model_sanitized

    # Standard lookup.
    return get_release_manifest_entry(model, version, _check_source_model_ids=False).id


def get_release_manifest_entry(
    model: str, version: Version = CURRENT_VERSION, _check_source_model_ids: bool = True
) -> ReleaseManifestModelEntry:
    if _check_source_model_ids and use_aihm_source(version):
        # In-source builds must rebuild the entire release manifest to do currect
        # model name -> ID lookups. To construct each entry in the release manifest,
        # several model YAML files need to be read. This is very expensive.
        #
        # Instead, we load the current MODEL_IDs and attempt to do a fast lookup first.
        from qai_hub_models.utils.path_helpers import MODEL_IDS

        # Maximize our chance at a "ID cache hit" by applying common name -> ID mappings.
        model_sanitized = model.replace("-", "_").replace(".", "_").lower()
        if model_sanitized in MODEL_IDS:
            from qai_hub_models.cli.proto_api import _build_release_manifest_entry
            from qai_hub_models.scripts.build_release_proto import _similar_chipsets

            if entry := _build_release_manifest_entry(
                model_sanitized, _similar_chipsets(), True
            ):
                return entry

    manifest = get_release_manifest(version)
    model_key = normalize_label(model)
    for entry in manifest.models:
        if model_key in (
            normalize_label(entry.id),
            normalize_label(entry.display_name),
        ):
            return entry
    raise CLIUnknownNameError(
        f"No model exists with the name or ID: {model!r}. "
        "Run qai-hub-models models to see all Model IDs."
    )


@functools.lru_cache(maxsize=1)
def get_release_manifest(
    version: Version = CURRENT_VERSION,
    local_path: Path | None = None,
) -> ReleaseManifest:
    """
    Fetch and cache the release manifest for a given version.

    The manifest lists every model in the release along with URLs
    for its info, perf, numerics, and release-assets protobufs.

    Parameters
    ----------
    version
        AI Hub Models release version. Defaults to the installed CLI version.
        Ignored when *local_path* is provided.
    local_path
        Path to a local manifest protobuf file. When provided, reads
        directly from disk instead of fetching from S3.

    Returns
    -------
    ReleaseManifest
        Parsed manifest protobuf.

    Raises
    ------
    UnsupportedVersionError
        If *version* is not a supported release (when *local_path* is None).
    """
    if use_internal_releases():
        from qai_hub_models_cli._internal.aws import QAIHM_PRIVATE_S3_BUCKET

        url_prefix = f"s3://{QAIHM_PRIVATE_S3_BUCKET}/"
    else:
        url_prefix = STORE_URL

    url = posixpath.join(
        url_prefix, "qai-hub-models", "releases", f"v{version}", "manifest.pb"
    )
    return fetch_release_proto(
        version,
        ReleaseManifest,
        cache_filename="manifest.pb",
        source_getter="get_release_manifest_proto",
        url=url,
        local_path=local_path,
    )
