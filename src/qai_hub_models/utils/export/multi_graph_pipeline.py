# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Export pipeline for multi-graph (non-collection) models — a single
``MultiGraphWorkbenchModel`` whose multiple graphs are compiled
independently and linked into one context binary.

Quantization is not handled here; precision flows in from the caller and is
assumed to already match what the source model serializes.
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
from qai_hub_models.utils.export.compile import run_multi_graph_compile
from qai_hub_models.utils.export.context import (
    resolve_model_cls,
    resolve_model_display_name,
)
from qai_hub_models.utils.export.download import download_multi_graph_model_bundle
from qai_hub_models.utils.export.link import run_multi_graph_link
from qai_hub_models.utils.export.profile import run_multi_graph_profile
from qai_hub_models.utils.export.result import MultiGraphExportResult, MultiGraphGroup
from qai_hub_models.utils.export.summary import (
    extract_tool_versions,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_multi_graph_source
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
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.QNN_CONTEXT_BINARY,
    skip_profiling: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    compile_options: str = "",
    profile_options: str = "",
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> MultiGraphExportResult:
    """
    Six-step recipe for multi-graph non-collection models:

        1. Instantiate the model and upload one source per graph (deduped
           when ``model.shared_source_model`` is True)
        2. Compile each graph to ``target_runtime``
        3. Link the compiled DLCs into one context binary (AOT only)
        4. Profile each graph on a real device against the linked binary
        5. Extract SDK / runtime versions
        6. Download the linked bundle with merged per-graph metadata

    Parameters
    ----------
    model_id
        Model folder name.
    device
        Hub device to export for.
    precision
        Target precision.
    target_runtime
        AOT runtime to target. ``MultiGraphWorkbenchModel`` only supports
        AOT runtimes (asserted by the base class).
    skip_profiling
        If set, skips profiling on real devices.
    skip_downloading
        If set, skips downloading the linked bundle.
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
    MultiGraphExportResult
        Per-graph compile/profile jobs, the link job, and the downloaded
        bundle path.
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

    # 1. Instantiate the model and upload one source per graph.
    model = model_cls.from_pretrained(
        **get_model_kwargs(model_cls, additional_model_kwargs)
    )
    input_specs = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )
    source_models = upload_multi_graph_source(model, input_specs)

    # 2. Compile each graph.
    compile_jobs = run_multi_graph_compile(
        model,
        model_name,
        device,
        target_runtime,
        precision,
        source_models,
        input_specs,
        extra_options=compile_options,
    )

    # 3. Link all graphs into one context binary.
    compiled = assert_success_and_get_target_models(compile_jobs)
    link_job = run_multi_graph_link(
        compiled,
        device,
        model_name,
        model,
        target_runtime,
    )
    target_model = link_job.get_target_model()
    assert target_model is not None, f"Link job failed: {link_job}"

    profile_opts = model.get_hub_profile_options(target_runtime, profile_options)

    # 4. Profile each graph against the linked model.
    profile_jobs: MultiGraphGroup[hub.client.ProfileJob] | None = None
    if not skip_profiling:
        profile_jobs = run_multi_graph_profile(
            model_name,
            device,
            profile_opts,
            target_model,
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

    # 6. Download the bundle.
    download_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        download_path = download_multi_graph_model_bundle(
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
            target_model=target_model,
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

    return MultiGraphExportResult(
        compile_jobs=compile_jobs,
        link_job=link_job,
        profile_jobs=profile_jobs,
        inference_jobs=None,
        download_path=download_path,
        tool_versions=tool_versions,
    )
