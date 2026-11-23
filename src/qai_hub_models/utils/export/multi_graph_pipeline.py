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

import pandas as pd
import qai_hub as hub
from qai_hub_models_cli.errors import CLIRuntimeError

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.args import get_export_model_name, get_model_kwargs
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_multi_graph_model import MultiGraphWorkbenchModel
from qai_hub_models.utils.export.compile import run_multi_graph_compile
from qai_hub_models.utils.export.context import (
    resolve_manifest,
    resolve_model_cls,
    resolve_model_display_name,
    resolve_model_id,
)
from qai_hub_models.utils.export.download import download_multi_graph_model_bundle
from qai_hub_models.utils.export.inference import run_multi_graph_inference
from qai_hub_models.utils.export.link import run_multi_graph_link
from qai_hub_models.utils.export.result import (
    MultiGraphExportIntermediates,
    MultiGraphExportResult,
    MultiGraphGroup,
    validate_export_intermediates,
)
from qai_hub_models.utils.export.summary import (
    compute_multi_graph_inference_summary,
    extract_tool_versions,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_multi_graph_source
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.printing import print_tool_versions
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_URL,
    get_device_and_chipset_name,
    get_target_models,
)


def export_model(
    source_dir: Path,
    device: hub.Device,
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.QNN_CONTEXT_BINARY,
    export_intermediates: MultiGraphExportIntermediates | None = None,
    skip_linking: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    compile_options: str = "",
    link_options: str = "",
    inference_options: str = "",
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
    source_dir
        On-disk path to the recipe folder (contains ``manifest.yaml`` and
        ``model.py``).
    device
        Hub device to export for.
    precision
        Target precision.
    target_runtime
        AOT runtime to target. ``MultiGraphWorkbenchModel`` only supports
        AOT runtimes (asserted by the base class).
    export_intermediates
        Already-built per-graph assets to start from, in place of rebuilding
        them. If ``compiled_models``/``compile_jobs`` is set, it must cover
        every graph -- upload/compile is skipped entirely, not per-graph.
        If ``source_models`` is set, it's used as the starting point for
        compile instead of re-uploading. ``link_job``/``inference_jobs``
        likewise skip linking/inference entirely when set. A job field takes
        precedence over its sibling model field when both are set.
    skip_linking
        If set, stops after compiling -- skips linking and everything after it.
    skip_inferencing
        If set, skips running the model on a real device.
    skip_downloading
        If set, skips downloading the linked bundle.
    skip_summary
        If set, skips waiting for and summarizing results.
    output_dir
        Directory to store generated assets. Defaults to ``<cwd>/export_assets``.
    compile_options
        Extra options for the compile job.
    link_options
        Extra options for the link job.
    inference_options
        Extra options for the inference job.
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
    assert target_runtime.uses_hub_link, (
        "multi_graph_pipeline.export_model() only supports runtimes that use "
        f"hub.link(), got {target_runtime}"
    )

    model_id = resolve_model_id(source_dir)
    if not can_access_qualcomm_ai_hub():
        raise CLIRuntimeError(
            "Could not find AI Hub credentials. Sign up at "
            f"{_AIHUB_URL} for free access. To use pre-released assets "
            "without AI Hub credentials, run "
            f"`qai-hub-models fetch {model_id}` instead."
        )

    manifest = resolve_manifest(source_dir)
    model_cls = resolve_model_cls(source_dir)
    display_name = resolve_model_display_name(source_dir)
    additional_model_kwargs["precision"] = precision
    model_name = get_export_model_name(
        model_cls, model_id, precision, additional_model_kwargs
    )
    output_path = Path(output_dir or Path.cwd() / "export_assets")

    _, chipset = get_device_and_chipset_name(device)
    export_intermediates = export_intermediates or MultiGraphExportIntermediates()

    # 1. Instantiate the model and upload a graph's source if needed.
    model = model_cls.from_pretrained(
        **get_model_kwargs(model_cls, additional_model_kwargs)
    )
    input_specs = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )

    compile_jobs: MultiGraphGroup[hub.client.CompileJob] | None = (
        export_intermediates.compile_jobs
    )
    compiled_models: MultiGraphGroup[hub.Model] | None = (
        export_intermediates.compiled_models
    )
    validate_export_intermediates(
        export_intermediates, target_runtime.uses_hub_link, keys=input_specs
    )

    # 2. Compile every graph, unless already-compiled assets were supplied.
    if compile_jobs is None and compiled_models is None:
        source_models = export_intermediates.source_models or MultiGraphGroup(
            upload_multi_graph_source(model, input_specs)
        )
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
    if skip_linking:
        return MultiGraphExportResult(compile_jobs=compile_jobs)

    # 3. Link all graphs into one context binary. Chained off the compile
    # jobs without waiting for them.
    link_job: hub.client.LinkJob | None = export_intermediates.link_job
    if link_job is not None:
        target_model = get_target_models(link_job)
    else:
        if compiled_models is None:
            assert compile_jobs is not None
            compiled_models = get_target_models(compile_jobs)
        link_job = run_multi_graph_link(
            compiled_models,
            device,
            model_name,
            model,
            target_runtime,
            extra_options=link_options,
        )
        target_model = get_target_models(link_job)

    inference_opts = model.get_hub_inference_options(target_runtime, inference_options)

    # 4. Profile each graph against the linked model.
    inference_jobs: MultiGraphGroup[hub.client.InferenceJob] | None = (
        export_intermediates.inference_jobs
    )
    if inference_jobs is None and not skip_inferencing:
        assert target_model is not None
        inference_jobs = run_multi_graph_inference(
            model.sample_inputs(
                input_spec=input_specs,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            inference_opts,
            target_model,
        )

    # 5. Tool versions.
    if not skip_summary or not skip_downloading:
        first_inference = (
            next(iter(inference_jobs.values()), None) if inference_jobs else None
        )
        first_compile = (
            next(iter(compile_jobs.values()), None) if compile_jobs else None
        )
        tool_versions, tool_versions_from_inference = extract_tool_versions(
            first_inference,
            first_compile,
        )
    else:
        tool_versions, tool_versions_from_inference = None, False

    # 6. Download the bundle.
    download_path: Path | None = None
    metadata: ModelMetadata | None = None
    if not skip_downloading and tool_versions is not None:
        download_path, metadata = download_multi_graph_model_bundle(
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
            use_case=manifest.use_case,
        )

    inference_metrics: MultiGraphGroup[pd.DataFrame] | None = None
    if not skip_summary:
        has_torch_baseline = (
            type(model).local_inference is not MultiGraphWorkbenchModel.local_inference
        )
        if inference_jobs and not has_torch_baseline:
            print(
                f"{type(model).__name__} does not implement local_inference; "
                "skipping on-device numerics."
            )
        if inference_jobs:
            inference_metrics = MultiGraphGroup()
            for graph_name, inference_job in inference_jobs.items():
                print_profile_summary(inference_job)
                if not has_torch_baseline:
                    continue
                print(f"Inference results for graph {graph_name}:")
                inference_metrics[graph_name] = compute_multi_graph_inference_summary(
                    model,
                    graph_name,
                    inference_job,
                    input_specs[graph_name],
                    outputs_to_skip=list(
                        (manifest.outputs_to_skip_validation or {}).keys()
                    ),
                    metrics=manifest.inference_metrics,
                    print_result=True,
                )
            inference_metrics = inference_metrics or None
        print_tool_versions(tool_versions, tool_versions_from_inference)
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
        inference_jobs=inference_jobs,
        download_path=download_path,
        tool_versions=tool_versions,
        metadata=metadata,
        inference_metrics=inference_metrics,
    )
