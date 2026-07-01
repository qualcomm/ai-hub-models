# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import qai_hub as hub
import requests
from packaging.version import Version
from packaging.version import parse as packaging_parse_version
from pydantic import ValidationError
from qai_hub_models_cli.fetch import get_asset_url

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models._version import __version__
from qai_hub_models.configs.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    ModelZooAssetConfig,
    download_file,
)
from qai_hub_models.utils.fetch_prerelease_assets import (
    fetch_prerelease_assets,
)
from qai_hub_models.utils.version_helpers import QAIHMVersion


def fetch_static_assets(
    model_id: str,
    runtime: TargetRuntime,
    precision: Precision = Precision.float,
    device_or_chipset: hub.Device | str | None = None,
    qaihm_version_tag: str | None = None,
    output_folder: str | os.PathLike | None = None,
    asset_config: ModelZooAssetConfig = ASSET_CONFIG,
    skip_download: bool = False,
    verbose: bool = True,
) -> tuple[Path | None, str]:
    """
    Fetch previously released assets for a model to disk, and place them in the output folder.

    Parameters
    ----------
    model_id
        Model ID to fetch.
    runtime
        Target Runtime to fetch.
    precision
        Precision to fetch.
    device_or_chipset
        Device or chipset for which assets should be fetched. Ignored if runtime is not compiled for a specific device.
    qaihm_version_tag
        QAIHM version tag to fetch (ex. "v0.33.0"). If None, gets the assets from the currently installed QAIHM version.
    output_folder
        If set, downloads all assets to this folder. If not set, only file URLs are returned. The paths list will be empty.
    asset_config
        QAIHM asset config.
    skip_download
        If True, only returns the URL of the asset without downloading it.
    verbose
        If True, prints additional information during the fetch process.

    Returns
    -------
    path : Path | None
        Downloaded file path. None if skip_download is True.
    url : str
        Model file URL.

    Notes
    -----
    This function will attempt to fetch assets that match the currently installed version of AI Hub Models.
    For users with access to private pre-release assets:
        * if the version tag provided is "current" or None, we will **only** attempt to fetch pre-released assets
        * if the version tag is specific (eg. v0.45.0), we will fetch released assets for that tag
    """
    qaihm_version_tag = qaihm_version_tag or __version__
    if "dev" in __version__:
        # We always try to fetch assets that correspond with the **installed** version of QAIHM.
        # If that version is not a release build, the most up-to-date assets are pre-release assets.
        return fetch_prerelease_assets(
            model_id,
            runtime,
            precision,
            device_or_chipset,
            output_folder,
            asset_config,
            verbose,
        ), ""

    parsed_qaihm_version_tag = (
        QAIHMVersion.tag_from_string(qaihm_version_tag)
        if qaihm_version_tag
        else QAIHMVersion.current_tag
    )

    if runtime.is_aot_compiled:
        if device_or_chipset is None:
            raise ValueError(
                "You must specify a device or chipset to fetch an asset that is device-specific."
            )
        if isinstance(device_or_chipset, hub.Device):
            # load devices_and_chipsets.yaml
            if parsed_qaihm_version_tag != QAIHMVersion.current_tag:
                chipsets = _load_chipsets_from_previous_release(
                    parsed_qaihm_version_tag, asset_config
                )
            else:
                chipsets = DevicesAndChipsetsYaml.load()
            _, device_details = chipsets.get_device_details_without_aihub(
                device_or_chipset
            )
            chipset = device_details.chipset
        else:
            chipset = device_or_chipset
    else:
        chipset = None

    version = packaging_parse_version(parsed_qaihm_version_tag[1:])
    if version < packaging_parse_version("0.44.0"):
        raise ValueError(
            "Fetching device-specific assets is not supported for QAIHM versions < v0.44.0. Please downgrade your AI Hub Models version to v0.43.0 or earlier to fetch assets for v0.43.0 and earlier."
        )

    asset_url = get_asset_url(
        model=model_id,
        runtime=runtime.value,
        precision=str(precision),
        chipset=chipset,
        version=version,
        quiet=True,
    )
    response = requests.head(asset_url, timeout=30, allow_redirects=True)
    if response.status_code != 200:
        raise ValueError("No release found.")

    asset_name = asset_config.get_release_asset_filename(
        model_id, runtime, precision, chipset
    )

    if skip_download:
        return None, asset_url

    return Path(
        download_file(
            asset_url,
            asset_name
            if output_folder is None
            else os.path.join(output_folder, asset_name),
        )
    ), asset_url


def _load_chipsets_from_previous_release(
    qaihm_version_tag: str, asset_config: ModelZooAssetConfig
) -> DevicesAndChipsetsYaml:
    """
    Fetch older devices_and_chipsets.yaml from GitHub.com, attempt to load & return it.

    Parameters
    ----------
    qaihm_version_tag:
        QAIHM version tag to fetch (ex. "v0.33.0")
    asset_config:
        QAIHM asset config.

    Returns
    -------
    DevicesAndChipsetsYaml
        Chipsets object.

    Raises
    ------
    FileNotFoundError
      - the yaml is unavailable on GitHub
      - the yaml schema has changed and can't be read by the current AI Hub Models version
    """
    chipsets_url = asset_config.get_qaihm_repo_download_url(
        None, "devices_and_chipsets.yaml", Version(qaihm_version_tag)
    )
    with TemporaryDirectory() as tmpdir:
        try:
            chipsets_path = download_file(
                chipsets_url, os.path.join(tmpdir, "devices_and_chipsets.yaml"), False
            )
        except ValueError:
            raise FileNotFoundError(
                f"Unable to find supported devices and chipsets for QAIHM {qaihm_version_tag}. Verify {qaihm_version_tag} is a valid tag at {asset_config.repo_url}, and verify that the model is available in that version."
            ) from None
        try:
            return DevicesAndChipsetsYaml.from_yaml(chipsets_path)
        except (ValueError, ValidationError):
            # The schema may have changed if we can't load the old yamls.
            raise FileNotFoundError(
                f"Unable to parse supported devices and chipsets for QAIHM {qaihm_version_tag}. Downgrade your AI Hub Models install to {qaihm_version_tag} and try again."
            ) from None
