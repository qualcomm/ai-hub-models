# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Download compiled hub models locally and stamp metadata alongside them."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs._info_yaml_enums import MODEL_USE_CASE
from qai_hub_models.configs.model_metadata import (
    ChipsetAttributes,
    ModelFileMetadata,
    ModelMetadata,
    merge_input_metadata,
    merge_output_metadata,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_collection_model import (
    CollectionModel,
    PrecompiledCollectionModel,
)
from qai_hub_models.utils.base_model import BaseModel, PrecompiledWorkbenchModel
from qai_hub_models.utils.base_multi_graph_collection_model import (
    MultiGraphCollectionModel,
)
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.onnx.helpers import (
    download_and_unzip_workbench_onnx_model,
    get_onnx_model_bundle_metadata,
)
from qai_hub_models.utils.path_helpers import get_next_free_path


def download_model_bundle(
    output_dir: os.PathLike | str,
    model: BaseModel,
    model_id: str,
    model_display_name: str,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_model: hub.Model,
    zip_assets: bool,
    hub_device: hub.Device | None = None,
    use_case: MODEL_USE_CASE | None = None,
) -> Path:
    """Download the compiled hub model and write semantic + file metadata alongside it."""
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)
    supplementary_files: dict[str, str]

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        if target_model.model_type == hub.SourceModelType.ONNX:
            onnx_result = download_and_unzip_workbench_onnx_model(
                target_model, dst_path, model_id
            )
            model_file_name, supplementary_files = get_onnx_model_bundle_metadata(
                onnx_result, dst_path, model_id
            )
        else:
            downloaded_path = target_model.download(os.path.join(dst_path, model_id))
            model_file_name = os.path.basename(downloaded_path)
            supplementary_files = {}

        file_metadata = ModelFileMetadata.from_hub_model(target_model)
        merge_input_metadata(file_metadata, model.get_input_spec(), runtime)
        merge_output_metadata(file_metadata, model.get_output_spec(), runtime)
        model_metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_display_name,
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files={model_file_name: file_metadata},
            chipset_attributes=(
                ChipsetAttributes.from_hub_device(hub_device)
                if runtime.is_aot_compiled
                else None
            ),
            use_case=use_case,
            supplementary_files=supplementary_files,
        )

        model.write_supplementary_files(dst_path, model_metadata)
        model_metadata.to_json(dst_path / "metadata.json")

        if zip_assets:
            return Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        shutil.move(dst_path, output_path)
    return output_path


def save_precompiled_model(
    output_dir: os.PathLike | str,
    model: PrecompiledWorkbenchModel,
    zip_assets: bool,
) -> Path:
    """Copy a precompiled QNN context binary to ``output_dir``."""
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        path = model.get_target_model_path()
        shutil.copyfile(src=path, dst=dst_path / os.path.basename(path))

        if zip_assets:
            return Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        shutil.move(dst_path, output_path)
    return output_path


def download_collection_model_bundle(
    output_dir: os.PathLike | str,
    model: CollectionModel,
    model_id: str,
    model_display_name: str,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_models: ComponentGroup[hub.Model],
    zip_assets: bool,
    hub_device: hub.Device | None = None,
    use_case: MODEL_USE_CASE | None = None,
) -> Path:
    """Download each component's hub model and write combined semantic metadata."""
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)
    supplementary_files: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        file_metadata_by_name: dict[str, ModelFileMetadata] = {}
        for component_name, target_model in target_models.items():
            if target_model.model_type == hub.SourceModelType.ONNX:
                onnx_result = download_and_unzip_workbench_onnx_model(
                    target_model, dst_path, component_name
                )
                model_file_name, supplementary_files_component = (
                    get_onnx_model_bundle_metadata(
                        onnx_result, dst_path, component_name
                    )
                )
                supplementary_files.update(supplementary_files_component)
            else:
                downloaded_path = target_model.download(
                    os.path.join(dst_path, component_name)
                )
                model_file_name = os.path.basename(downloaded_path)

            file_metadata = ModelFileMetadata.from_hub_model(target_model)
            merge_input_metadata(
                file_metadata,
                model.get_component_input_spec(component_name),
                runtime,
            )
            merge_output_metadata(
                file_metadata,
                model.get_component_output_spec(component_name),
                runtime,
            )
            file_metadata_by_name[model_file_name] = file_metadata

        model_metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_display_name,
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files=file_metadata_by_name,
            chipset_attributes=(
                ChipsetAttributes.from_hub_device(hub_device)
                if runtime.is_aot_compiled
                else None
            ),
            use_case=use_case,
            supplementary_files=supplementary_files,
        )
        model.write_supplementary_files(dst_path, model_metadata)
        model_metadata.to_json(dst_path / "metadata.json")

        if zip_assets:
            return Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        shutil.move(dst_path, output_path)
    return output_path


def download_multi_graph_collection_model_bundle(
    output_dir: os.PathLike | str,
    model: MultiGraphCollectionModel,
    model_id: str,
    model_display_name: str,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_models: ComponentGroup[hub.Model],
    zip_assets: bool,
    hub_device: hub.Device | None = None,
    use_case: MODEL_USE_CASE | None = None,
) -> Path:
    """
    Download each component's hub model and merge metadata across every graph
    of that component into a single ``ModelFileMetadata``.
    """
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)
    all_input_specs = model.get_input_spec()

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        file_metadata_by_name: dict[str, ModelFileMetadata] = {}
        for component_name, target_model in target_models.items():
            if target_model.model_type != hub.SourceModelType.QNN_CONTEXT_BINARY:
                raise ValueError(
                    "Multi graph models may only consist of context binaries."
                )

            downloaded_path = target_model.download(
                os.path.join(dst_path, component_name)
            )
            model_file_name = os.path.basename(downloaded_path)

            file_metadata = ModelFileMetadata.from_hub_model(target_model)
            for graph_spec in all_input_specs.by_component(component_name).values():
                if hasattr(model, "remap_metadata_input_spec"):
                    remapped_spec = model.remap_metadata_input_spec(
                        component_name=component_name,
                        graph_input_spec=graph_spec,
                        compiled_input_names=set(file_metadata.inputs.keys()),
                    )
                else:
                    remapped_spec = graph_spec
                merge_input_metadata(file_metadata, remapped_spec, runtime)
            file_metadata_by_name[model_file_name] = file_metadata

        model_metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_display_name,
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files=file_metadata_by_name,
            chipset_attributes=(
                ChipsetAttributes.from_hub_device(hub_device)
                if runtime.is_aot_compiled
                else None
            ),
            use_case=use_case,
        )
        model.write_supplementary_files(dst_path, model_metadata)
        model_metadata.to_json(dst_path / "metadata.json")

        if zip_assets:
            return Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        shutil.move(dst_path, output_path)
    return output_path


def download_multi_graph_model_bundle(
    output_dir: os.PathLike | str,
    model: MultiGraphWorkbenchModel,
    model_id: str,
    model_display_name: str,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_model: hub.Model,
    zip_assets: bool,
    hub_device: hub.Device | None = None,
    use_case: MODEL_USE_CASE | None = None,
) -> Path:
    """
    Download the linked context binary and merge metadata across every graph
    into a single ``ModelFileMetadata``.
    """
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)
    all_input_specs = model.get_input_spec()

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        if target_model.model_type != hub.SourceModelType.QNN_CONTEXT_BINARY:
            raise ValueError("Multi graph models may only consist of context binaries.")

        downloaded_path = target_model.download(os.path.join(dst_path, model_id))
        model_file_name = os.path.basename(downloaded_path)

        file_metadata = ModelFileMetadata.from_hub_model(target_model)
        for graph_spec in all_input_specs.values():
            merge_input_metadata(file_metadata, graph_spec, runtime)

        model_metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_display_name,
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files={model_file_name: file_metadata},
            chipset_attributes=(
                ChipsetAttributes.from_hub_device(hub_device)
                if runtime.is_aot_compiled
                else None
            ),
            use_case=use_case,
        )
        model.write_supplementary_files(dst_path, model_metadata)
        model_metadata.to_json(dst_path / "metadata.json")

        if zip_assets:
            return Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        shutil.move(dst_path, output_path)
    return output_path


def save_precompiled_collection_models(
    output_dir: os.PathLike | str,
    model: PrecompiledCollectionModel,
    components: list[str],
    zip_assets: bool,
) -> Path:
    """Copy each component's precompiled context binary to ``output_dir``."""
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        for component_name in components:
            path = model.get_component_target_model_path(component_name)
            shutil.copyfile(src=path, dst=dst_path / os.path.basename(path))

        if zip_assets:
            return Path(
                shutil.make_archive(
                    str(output_path),
                    "zip",
                    root_dir=tmpdir,
                    base_dir=output_folder_name,
                )
            )
        shutil.move(dst_path, output_path)
    return output_path
