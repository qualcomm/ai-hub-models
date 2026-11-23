# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.scorecard import ScorecardInferencePath
from qai_hub_models.scorecard.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    ModelZooAssetConfig,
)
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_qaihm_s3,
    s3_download,
    s3_file_exists,
)
from qai_hub_models.utils.path_helpers import get_next_free_path


def fetch_prerelease_assets(
    model_id: str,
    runtime_or_path: TargetRuntime | ScorecardInferencePath,
    precision: Precision = Precision.float,
    device_or_chipset: hub.Device | str | None = None,
    output_folder: str | os.PathLike | None = None,
    asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    verbose: bool = True,
) -> Path:
    """
    Fetch pre-release assets for this model to disk.

    Parameters
    ----------
    model_id
        Model ID to fetch
    runtime_or_path
        Target Runtime or ScorecardInferencePath to fetch
    precision
        Precision to fetch
    device_or_chipset
        Device or chipset for which assets should be fetched. Ignored if runtime is not compiled for a specific device.
    output_folder
        If set, downloads all assets to this folder. If not set, only file URLs are returned. The paths list will be empty.
    asset_config
        QAIHM asset config.
    verbose
        If True, prints additional information during the fetch process.

    Returns
    -------
    path: Path
        File path of the downloaded asset.
    """
    if verbose:
        print(
            "Attempting to fetch pre-release assets for the currently installed AI Hub Models package."
        )
        print("This requires access to AI Hub Models' private AWS S3 bucket.")
        # This line is printed for users of fetch_static_assets, which calls fetch_prerelease_assets.
        # The default behavior when no specific version is provided to fetch_static_assets is to fetch unpublished assets.
        print(
            "If you wish to fetch published (released) assets, provide a specific version tag.\n"
        )

    sc_path = (
        ScorecardInferencePath.from_runtime(runtime_or_path)
        if isinstance(runtime_or_path, TargetRuntime)
        else runtime_or_path
    )
    runtime = sc_path.runtime
    if runtime.is_aot_compiled:
        if device_or_chipset is None:
            raise ValueError(
                "You must specify a device or chipset to fetch an asset that is device-specific."
            )
        if isinstance(device_or_chipset, hub.Device):
            chipsets = DevicesAndChipsetsYaml.load()
            _, device_details = chipsets.get_device_details_without_aihub(
                device_or_chipset
            )
            chipset = device_details.chipset
        else:
            chipset = device_or_chipset
    else:
        chipset = None

    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if asset := assets.get_asset(precision, chipset, sc_path):
        return download_prerelease_asset(
            asset,
            model_id=model_id,
            runtime=runtime,
            precision=precision,
            chipset=chipset,
            output_folder=output_folder,
            asset_config=asset_config,
            verbose=verbose,
        )
    raise ValueError("No pre-release assets found for the specified configuration.")


def download_prerelease_asset(
    asset: QAIHMModelReleaseAssets.AssetDetails,
    model_id: str,
    runtime: TargetRuntime,
    precision: Precision,
    chipset: str | None,
    output_folder: str | os.PathLike | None = None,
    asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    verbose: bool = True,
) -> Path:
    """Download an already-resolved pre-release asset from S3.

    Use when the caller has already chosen a release-assets entry — for
    example, after consulting an in-flight workflow override yaml — and just
    needs the S3 download.
    """
    assert asset.s3_key is not None, "s3_key must be present for pre-release assets"
    bucket = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
    if not s3_file_exists(bucket, asset.s3_key):
        raise ValueError("No pre-release assets found for the specified configuration.")
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    asset_name = asset_config.get_release_asset_filename(
        model_id, runtime, precision, chipset
    )
    dst_path = Path(
        os.path.join(output_folder, asset_name) if output_folder else asset_name
    )
    dst_path = get_next_free_path(dst_path)
    s3_download(bucket, asset.s3_key, dst_path, verbose)
    return dst_path
