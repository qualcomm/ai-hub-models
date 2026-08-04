# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Export pipeline for multi-graph collection models — sharded LLMs whose
components each contain multiple per-graph DLCs that get linked together into
one context binary per component.

Quantization for these models lives in a separate ``quantize.py`` script;
precision flows in from the caller (typically resolved from the checkpoint by
the model's ``export.py`` shim).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.args import get_export_model_name, get_model_kwargs
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.export.compile import run_multi_graph_collection_compile
from qai_hub_models.utils.export.context import (
    resolve_model_cls,
    resolve_model_display_name,
)
from qai_hub_models.utils.export.download import (
    download_multi_graph_collection_model_bundle,
)
from qai_hub_models.utils.export.link import run_multi_graph_collection_link
from qai_hub_models.utils.export.profile import run_multi_graph_collection_profile
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphCollectionExportResult,
    MultiGraphComponentGroup,
)
from qai_hub_models.utils.export.summary import (
    extract_tool_versions,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_multi_graph_collection_source
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.printing import print_tool_versions
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_URL,
    assert_success_and_get_target_models,
    get_device_and_chipset_name,
)


def export_model(
    model_id: str,
    device: hub.Device,
    precision: Precision,
    components: list[str] | None = None,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    skip_profiling: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    compile_options: str = "",
    profile_options: str = "",
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> MultiGraphCollectionExportResult:
    """
    Six-step recipe for sharded LLMs (``has_multi_graph + is_collection_model``):

        1. Instantiate the model and upload one source per (component, graph),
           deduping per component when the source is shared
        2. Compile each (component, graph) pair to ``target_runtime``
        3. Link each component's compiled DLCs into one context binary (AOT only)
        4. Profile each (component, graph) on a real device
        5. Extract SDK / runtime versions
        6. Download every component's bundle together with combined metadata

    Parameters
    ----------
    model_id
        Model folder name.
    device
        Hub device to export for.
    precision
        Target precision; for these LLMs typically resolved from the
        checkpoint by the caller.
    components
        Optional subset of components to export. Defaults to all.
    target_runtime
        On-device runtime to target.
    skip_profiling
        If set, skips profiling on real devices.
    skip_downloading
        If set, skips downloading the compiled bundle.
    skip_summary
        If set, skips waiting for and summarizing results.
    output_dir
        Directory to store generated assets. Defaults to ``<cwd>/export_assets``.
    compile_options
        Extra options for the compile job.
    profile_options
        Extra options for the profile job.
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Any additional keyword arguments passed to ``Model.from_pretrained``.

    Returns
    -------
    MultiGraphCollectionExportResult
        Per-component compile/link/profile jobs and the downloaded bundle path.
    """
    warnings.filterwarnings("ignore")

    if not can_access_qualcomm_ai_hub():
        raise RuntimeError(
            "Could not find AI Hub credentials. Sign up at "
            f"{_AIHUB_URL} for free access. To use pre-released assets "
            "without AI Hub credentials, run "
            f"`qai-hub-models fetch {model_id}` instead."
        )

    model_cls = resolve_model_cls(model_id)
    display_name = resolve_model_display_name(model_id)
    additional_model_kwargs["precision"] = precision
    model_name = get_export_model_name(
        model_cls, model_id, precision, additional_model_kwargs
    )
    output_path = Path(output_dir or Path.cwd() / "export_assets")

    _, chipset = get_device_and_chipset_name(device)

    # 1. Instantiate the model and upload one source per (component, graph).
    model = model_cls.from_pretrained(
        **get_model_kwargs(model_cls, additional_model_kwargs)
    )
    components = components or model.component_names
    for name in components:
        if name not in model.component_names:
            raise ValueError(f"Invalid component {name}.")
    input_specs = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )
    source_models = upload_multi_graph_collection_source(model, input_specs, components)

    # 2. Compile each (component, graph) pair.
    compile_jobs = run_multi_graph_collection_compile(
        model,
        model_name,
        device,
        target_runtime,
        precision,
        source_models,
        input_specs,
        components,
        extra_options=compile_options,
    )

    # 3. Link each component's compiled DLCs (AOT only). For JIT runtimes pick
    #    one job per component out of the compile results.
    link_jobs: ComponentGroup[hub.client.LinkJob] | None = None
    target_models: ComponentGroup[hub.Model]
    if target_runtime.uses_hub_link:
        compiled = assert_success_and_get_target_models(compile_jobs)
        link_jobs = run_multi_graph_collection_link(
            compiled,
            device,
            model_name,
            model,
            target_runtime,
        )
        target_models = assert_success_and_get_target_models(link_jobs)
    else:
        first_per_component: dict[str, hub.client.CompileJob] = {}
        for (comp_name, _graph), job in compile_jobs.items():
            first_per_component.setdefault(comp_name, job)
        target_models = assert_success_and_get_target_models(
            ComponentGroup(first_per_component)
        )

    profile_opts = model.get_hub_profile_options(target_runtime, profile_options)

    # 4. Profile each (component, graph).
    profile_jobs: MultiGraphComponentGroup[hub.client.ProfileJob] | None = None
    if not skip_profiling:
        profile_jobs = run_multi_graph_collection_profile(
            model_name,
            device,
            profile_opts,
            target_models,
            components,
        )

    # 5. Tool versions.
    if not skip_summary or not skip_downloading:
        first_profile = (
            next(iter(profile_jobs.values()), None) if profile_jobs else None
        )
        first_compile = next(iter(compile_jobs.values()), None)
        tool_versions, tool_versions_from_device = extract_tool_versions(
            first_profile,
            None,
            first_compile,
        )
    else:
        tool_versions, tool_versions_from_device = None, False

    # 6. Download every component into one bundle.
    download_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        download_path = download_multi_graph_collection_model_bundle(
            output_dir=output_path
            / ASSET_CONFIG.get_release_asset_name(
                model_id, target_runtime, precision, chipset
            ),
            model=model,
            model_id=model_id,
            model_display_name=display_name,
            runtime=target_runtime,
            precision=precision,
            tool_versions=tool_versions,
            target_models=target_models,
            zip_assets=zip_assets,
            hub_device=device,
        )

    if not skip_summary:
        if profile_jobs:
            for profile_job in profile_jobs.values():
                print_profile_summary(profile_job)
        print_tool_versions(tool_versions, tool_versions_from_device)
        if download_path is not None:
            print(f"{model_name} was saved to {download_path}\n")
        if target_runtime in (TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT):
            print(
                "These models can be deployed on-device using the Genie SDK. "
                "For a full tutorial, please follow the instructions here: "
                "https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
            )

    return MultiGraphCollectionExportResult(
        compile_jobs=compile_jobs,
        link_jobs=link_jobs,
        profile_jobs=profile_jobs,
        inference_jobs=None,
        download_path=download_path,
        tool_versions=tool_versions,
    )
