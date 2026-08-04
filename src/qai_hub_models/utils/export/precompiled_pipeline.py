# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Export pipeline for precompiled models — models that ship a QNN context binary
and skip the PyTorch → ONNX → compile path entirely.

Both single-model and collection-model precompiled flows live here; they share
~80% of the recipe and the asymmetry is small enough to keep in one file.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import qai_hub as hub

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_collection_model import PrecompiledCollectionModel
from qai_hub_models.utils.base_model import PrecompiledWorkbenchModel
from qai_hub_models.utils.export.context import resolve_model_cls
from qai_hub_models.utils.export.download import (
    save_precompiled_collection_models,
    save_precompiled_model,
)
from qai_hub_models.utils.export.profile import (
    run_collection_profile,
    run_profile,
)
from qai_hub_models.utils.export.result import (
    CollectionExportResult,
    ComponentGroup,
    ExportResult,
)
from qai_hub_models.utils.export.summary import (
    extract_tool_versions,
    print_profile_summary,
)
from qai_hub_models.utils.printing import print_tool_versions
from qai_hub_models.utils.qai_hub_helpers import _AIHUB_URL, get_device_and_chipset_name

_GENIE_BLURB = (
    "These models can be deployed on-device using the Genie SDK. "
    "For a full tutorial, please follow the instructions here: "
    "https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
)


def _export_precompiled_single(
    model_cls: type[PrecompiledWorkbenchModel],
    model_id: str,
    manifest: QAIHMModelManifest,
    device: hub.Device,
    skip_profiling: bool,
    skip_downloading: bool,
    skip_summary: bool,
    download_dir: Path,
    profile_options: str,
    zip_assets: bool,
    target_runtime: TargetRuntime,
) -> ExportResult:
    print("Initializing model class")
    model = model_cls.from_pretrained()

    uploaded = (
        hub.upload_model(model.get_target_model_path()) if not skip_profiling else None
    )

    profile_opts = model.get_hub_profile_options(target_runtime, profile_options)
    profile_job = (
        run_profile(model_id, device, profile_opts, uploaded)
        if uploaded is not None
        else None
    )

    if not skip_summary or not skip_downloading:
        tool_versions, tool_versions_from_device = extract_tool_versions(
            profile_job,
            None,
            None,
        )
    else:
        tool_versions, tool_versions_from_device = None, False

    download_path = (
        save_precompiled_model(download_dir, model, zip_assets)
        if not skip_downloading
        else None
    )

    if not skip_summary:
        if profile_job is not None:
            print_profile_summary(profile_job)
        print_tool_versions(tool_versions, tool_versions_from_device)
        if download_path is not None:
            print(f"{model_id} was saved to {download_path}\n")
        if target_runtime in (TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT):
            print(_GENIE_BLURB)

    return ExportResult(
        profile_job=profile_job,
        download_path=download_path,
        tool_versions=tool_versions,
    )


def _export_precompiled_collection(
    model_cls: type[PrecompiledCollectionModel],
    model_id: str,
    manifest: QAIHMModelManifest,
    device: hub.Device,
    components: list[str] | None,
    skip_profiling: bool,
    skip_downloading: bool,
    skip_summary: bool,
    download_dir: Path,
    profile_options: str,
    zip_assets: bool,
    target_runtime: TargetRuntime,
) -> CollectionExportResult:
    print("Initializing model class")
    model = model_cls.from_pretrained()
    components = components or model.component_names
    for name in components:
        if name not in model.component_names:
            raise ValueError(f"Invalid component {name}.")

    uploaded = ComponentGroup[hub.Model]()
    if not skip_profiling:
        print("Uploading model assets on hub")
        for name in components:
            uploaded[name] = hub.upload_model(
                model.get_component_target_model_path(name)
            )

    profile_opts = model.get_hub_profile_options(target_runtime, profile_options)
    profile_jobs: ComponentGroup[hub.client.ProfileJob] | None = (
        run_collection_profile(model_id, device, profile_opts, uploaded, components)
        if not skip_profiling
        else None
    )

    if not skip_summary or not skip_downloading:
        first_pj = next(iter(profile_jobs.values()), None) if profile_jobs else None
        tool_versions, tool_versions_from_device = extract_tool_versions(
            first_pj,
            None,
            None,
        )
    else:
        tool_versions, tool_versions_from_device = None, False

    download_path = (
        save_precompiled_collection_models(download_dir, model, components, zip_assets)
        if not skip_downloading
        else None
    )

    if not skip_summary:
        if profile_jobs:
            for job in profile_jobs.values():
                print_profile_summary(job)
        print_tool_versions(tool_versions, tool_versions_from_device)
        if download_path is not None:
            print(f"{model_id} was saved to {download_path}\n")
        if target_runtime in (TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT):
            print(_GENIE_BLURB)

    return CollectionExportResult(
        profile_jobs=profile_jobs,
        download_path=download_path,
        tool_versions=tool_versions,
    )


def export_model(
    model_id: str,
    device: hub.Device,
    components: list[str] | None = None,
    skip_profiling: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    profile_options: str = "",
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> ExportResult | CollectionExportResult:
    """
    Five-step recipe for precompiled assets:

        1. Initialize the precompiled model and upload its asset(s) to hub
        2. Profile on a real device
        3. (skipped — precompiled models do not run torch inference comparisons)
        4. Extract SDK / runtime versions
        5. Save the precompiled asset(s) to the local output directory

    Parameters
    ----------
    model_id
        Model folder name.
    device
        Hub device to export for.
    components
        Optional subset of components for collection precompiled models.
    skip_profiling
        If set, skips profiling on real devices.
    skip_downloading
        If set, skips saving the precompiled asset(s) locally.
    skip_summary
        If set, skips waiting for and summarizing results.
    output_dir
        Directory to store generated assets. Defaults to ``<cwd>/export_assets``.
    profile_options
        Extra options for the profile job.
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Any additional keyword arguments passed to ``Model.from_pretrained``.

    Returns
    -------
    ExportResult | CollectionExportResult
        Profile job(s), downloaded asset path, and tool versions.
    """
    warnings.filterwarnings("ignore")

    if not can_access_qualcomm_ai_hub():
        raise RuntimeError(
            "Could not find AI Hub credentials. Sign up at "
            f"{_AIHUB_URL} for free access. To use pre-released assets "
            "without AI Hub credentials, run "
            f"`qai-hub-models fetch {model_id}` instead."
        )

    manifest = QAIHMModelManifest.from_model(model_id)
    model_cls = resolve_model_cls(model_id)
    precision = manifest.default_precision
    target_runtime = TargetRuntime.QNN_CONTEXT_BINARY
    output_path = Path(output_dir or Path.cwd() / "export_assets")
    is_collection = issubclass(model_cls, PrecompiledCollectionModel)

    _, chipset = get_device_and_chipset_name(device)
    download_dir = output_path / ASSET_CONFIG.get_release_asset_name(
        model_id, target_runtime, precision, chipset
    )

    if is_collection:
        return _export_precompiled_collection(
            model_cls,
            model_id,
            manifest,
            device,
            components,
            skip_profiling,
            skip_downloading,
            skip_summary,
            download_dir,
            profile_options,
            zip_assets,
            target_runtime,
        )
    return _export_precompiled_single(
        model_cls,
        model_id,
        manifest,
        device,
        skip_profiling,
        skip_downloading,
        skip_summary,
        download_dir,
        profile_options,
        zip_assets,
        target_runtime,
    )
