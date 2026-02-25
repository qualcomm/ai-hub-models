# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, cast

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.qwen2_7b_instruct import MODEL_ID, Model
from qai_hub_models.utils.args import (
    export_parser,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_model import (
    BasePrecompiledModel,
    PrecompiledCollectionModel,
)
from qai_hub_models.utils.export_result import CollectionExportResult, ExportResult
from qai_hub_models.utils.export_without_hub_access import export_without_hub_access
from qai_hub_models.utils.path_helpers import get_next_free_path
from qai_hub_models.utils.printing import (
    print_profile_metrics_from_job,
    print_tool_versions,
)
from qai_hub_models.utils.qai_hub_helpers import (
    can_access_qualcomm_ai_hub,
)


def profile_model(
    model_name: str,
    device: hub.Device,
    options: dict[str, str],
    uploaded_models: dict[str, hub.Model],
    components: list[str] | None = None,
) -> dict[str, hub.client.ProfileJob]:
    profile_jobs: dict[str, hub.client.ProfileJob] = {}
    for component_name in components or Model.component_class_names:
        print(f"Profiling model {component_name} on a hosted device.")
        submitted_profile_job = hub.submit_profile_job(
            model=uploaded_models[component_name],
            device=device,
            name=f"{model_name}_{component_name}",
            options=options.get(component_name, ""),
        )
        profile_jobs[component_name] = cast(
            hub.client.ProfileJob, submitted_profile_job
        )
    return profile_jobs


def save_model(
    output_dir: os.PathLike | str,
    tool_versions: ToolVersions | None,
    model: PrecompiledCollectionModel,
    components: list[str],
    zip_assets: bool,
) -> Path:
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()
        for component_name in components:
            path = cast(
                BasePrecompiledModel, model.components[component_name]
            ).get_target_model_path()
            shutil.copyfile(src=path, dst=dst_path / os.path.basename(path))

        if zip_assets:
            output_path = Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        else:
            shutil.move(dst_path, output_path)

    return output_path


def export_model(
    device: hub.Device,
    components: list[str] | None = None,
    skip_profiling: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    profile_options: str = "",
    fetch_static_assets: str | None = None,
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> CollectionExportResult:
    """
    This function executes the following recipe:

        1. Initialize model
        2. Upload model assets to hub
        3. Profiles the model performance on a real device
        4. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
        5. Saves the model asset to the local directory
        6. Summarizes the results from profiling

    Each of the last 4 steps can be optionally skipped using the input options.

    Parameters
    ----------
    device
        Device for which to export the model (e.g., hub.Device("Samsung Galaxy S25")).
        Full list of available devices can be found by running `hub.get_devices()`.
    components
        List of sub-components of the model that will be exported.
        Each component is compiled and profiled separately.
        Defaults to all components of the CollectionModel if not specified.
    skip_profiling
        If set, skips profiling of compiled model on real devices.
    skip_downloading
        If set, skips downloading of compiled model.
    skip_summary
        If set, skips waiting for and summarizing results
        from profiling.
    output_dir
        Directory to store generated assets (e.g. compiled model).
        Defaults to `<cwd>/export_assets`.
    profile_options
        Additional options to pass when submitting the profile job.
    fetch_static_assets
        If set, known assets are fetched from the given version rather than re-computing them. Can be passed as "latest" or "v<version>".
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Additional optional kwargs used to customize
        `model_cls.from_precompiled`

    Returns
    -------
    CollectionExportResult
        A Mapping from component_name to:
            * A ProfileJob containing metadata about the profile job (None if profiling skipped).
        * The path to the downloaded model folder (or zip), or None if one or more of: skip_downloading is True, fetch_static_assets is set, or AI Hub Workbench is not accessible
    """
    model_name = MODEL_ID

    output_path = Path(output_dir or Path.cwd() / "export_assets")
    precision = Precision.w4a16
    target_runtime = TargetRuntime.QNN_CONTEXT_BINARY

    component_arg = components
    components = components or Model.component_class_names
    for component_name in components:
        if component_name not in Model.component_class_names:
            raise ValueError(f"Invalid component {component_name}.")
    if fetch_static_assets or not can_access_qualcomm_ai_hub():
        static_model_path = export_without_hub_access(
            MODEL_ID,
            device,
            skip_profiling,
            True,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            precision,
            profile_options,
            component_arg,
            qaihm_version_tag=fetch_static_assets,
        )
        return CollectionExportResult(
            components={
                component_name: ExportResult() for component_name in components
            },
            download_path=static_model_path,
        )

    hub_device = hub.get_devices(
        name=device.name, attributes=device.attributes, os=device.os
    )[-1]
    chipset_attr = next(
        (attr for attr in hub_device.attributes if "chipset" in attr), None
    )
    chipset = chipset_attr.split(":")[-1] if chipset_attr else None

    # 1. Initialize model
    print("Initializing model class")
    model = Model.from_precompiled()

    # 2. Upload model assets to hub
    uploaded_models: dict[str, hub.Model] = {}
    if not skip_profiling:
        print("Uploading model assets on hub")
        for component_name in components:
            path = cast(
                BasePrecompiledModel, model.components[component_name]
            ).get_target_model_path()
            uploaded_models[component_name] = hub.upload_model(path)

    # 3. Profiles the model performance on a real device
    profile_jobs: dict[str, hub.client.ProfileJob] = {}
    if not skip_profiling:
        profile_jobs = profile_model(
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            uploaded_models,
            components,
        )

    # 4. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
    tool_versions: ToolVersions | None = None
    tool_versions_are_from_device_job = False
    if not skip_summary:
        profile_job = next(iter(profile_jobs.values())) if profile_jobs else None
        if profile_job is not None and profile_job.wait():
            tool_versions = ToolVersions.from_job(profile_job)
            tool_versions_are_from_device_job = True

    # 5. Saves the model asset to the local directory
    downloaded_model_path: Path | None = None
    if not skip_downloading:
        model_directory = output_path / ASSET_CONFIG.get_release_asset_name(
            MODEL_ID, target_runtime, precision, chipset
        )
        downloaded_model_path = save_model(
            model_directory, tool_versions, model, components, zip_assets
        )

    # 6. Summarizes the results from profiling
    if not skip_summary and not skip_profiling:
        for component_name in components:
            profile_job = profile_jobs[component_name]
            assert profile_job.wait().success, "Job failed: " + profile_job.url
            profile_data: dict[str, Any] = profile_job.download_profile()
            print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary:
        print_tool_versions(tool_versions, tool_versions_are_from_device_job)

    if downloaded_model_path:
        print(f"{model_name} was saved to {downloaded_model_path}\n")

    print(
        "These models can be deployed on-device using the Genie SDK. For a full tutorial, please follow the instructions here: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
    )

    return CollectionExportResult(
        components={
            component_name: ExportResult(
                profile_job=profile_jobs.get(component_name, None),
            )
            for component_name in components
        },
        download_path=downloaded_model_path,
        tool_versions=tool_versions,
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.w4a16: [
            TargetRuntime.QNN_CONTEXT_BINARY,
        ],
    }

    parser = export_parser(
        model_cls=Model,
        export_fn=export_model,
        supported_precision_runtimes=supported_precision_runtimes,
        default_export_device="Samsung Galaxy S25 (Family)",
    )
    args = parser.parse_args()
    export_model(**vars(args))


if __name__ == "__main__":
    main()
