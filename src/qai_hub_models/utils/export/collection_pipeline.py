# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Export pipeline for CollectionModel (single-graph components).

Each step submits one hub job per component. Multi-graph collections (sharded
LLMs) require a richer flow and are handled separately — see the design doc.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import qai_hub as hub
from qai_hub_models_cli.errors import CLIRuntimeError, CLIUserError

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.args import get_export_model_name, get_model_kwargs
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.export.compile import (
    run_collection_compile,
)
from qai_hub_models.utils.export.context import (
    resolve_manifest,
    resolve_model_app_cls,
    resolve_model_cls,
    resolve_model_display_name,
    resolve_model_id,
)
from qai_hub_models.utils.export.download import download_collection_model_bundle
from qai_hub_models.utils.export.inference import run_collection_inference
from qai_hub_models.utils.export.link import run_collection_link
from qai_hub_models.utils.export.quantize import (
    resolve_component_precisions,
    run_collection_quantize,
)
from qai_hub_models.utils.export.result import (
    CollectionExportIntermediates,
    CollectionExportResult,
    ComponentGroup,
    validate_export_intermediates,
)
from qai_hub_models.utils.export.summary import (
    compute_inference_summary,
    extract_tool_versions,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_collection_source
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.printing import print_on_target_demo_cmd, print_tool_versions
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_URL,
    get_device_and_chipset_name,
    get_target_models,
)


def export_model(
    source_dir: Path,
    device: hub.Device,
    components: list[str] | None = None,
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    num_calibration_samples: int | None = None,
    export_intermediates: CollectionExportIntermediates | None = None,
    skip_compiling: bool = False,
    skip_linking: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    compile_options: str = "",
    quantize_options: str = "",
    link_options: str = "",
    inference_options: str = "",
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> CollectionExportResult:
    """
    Run the seven-step recipe over every selected component:

        1. Instantiate the CollectionModel and upload each component's source
        2. Quantize each non-float component
        3. Compile each component to ``target_runtime``
        4. Link each compiled DLC to a context binary (AOT runtimes only)
        5. Run on-device inference per component, profiling at the same time
        6. Extract SDK / runtime versions
        7. Download all components together with a combined metadata.json

    Parameters
    ----------
    source_dir
        On-disk path to the recipe folder (contains ``manifest.yaml`` and
        ``model.py``).
    device
        Hub device to export for.
    components
        Optional subset of components to export. Defaults to all components.
    precision
        Target precision (per-component or single).
    target_runtime
        On-device runtime to target.
    num_calibration_samples
        Number of calibration data samples to use for quantization.
    export_intermediates
        Already-built per-component assets to start from, in place of
        rebuilding them. For each component, if ``compiled_model``/``compile_job``
        is set, upload/quantize/compile are all skipped for that component; if
        only ``quantized_model``/``quantize_job`` is set, upload/quantize are
        skipped; if ``source_model`` is set, it's used as the starting point
        for compile/quantize instead of re-uploading. A job field takes
        precedence over its sibling model field when both are set. Components
        missing from a field fall back to the normal pipeline.
    skip_compiling
        If set, skips compiling.
    skip_linking
        If set, stops after compiling -- skips linking (AOT runtimes only)
        and everything after it.
    skip_inferencing
        If set, skips the inference job entirely -- no outputs and no profile
        metrics.
    skip_downloading
        If set, skips downloading compiled models.
    skip_summary
        If set, skips waiting for and summarizing results.
    output_dir
        Directory to store generated assets. Defaults to ``<cwd>/export_assets``.
    compile_options
        Extra options for the compile job.
    quantize_options
        Extra options for the quantize job.
    link_options
        Extra options for the link job. Ignored for runtimes that do not link.
    inference_options
        Extra options for the inference job.
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Any additional keyword arguments passed to ``Model.from_pretrained``.

    Returns
    -------
    CollectionExportResult
        Jobs per component, downloaded bundle path, and tool versions.
    """
    warnings.filterwarnings("ignore")

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
    app = resolve_model_app_cls(source_dir)
    display_name = resolve_model_display_name(source_dir)
    model_name = get_export_model_name(
        model_cls, model_id, precision, additional_model_kwargs
    )
    output_path = Path(output_dir or Path.cwd() / "export_assets")

    _, chipset = get_device_and_chipset_name(device)
    export_intermediates = export_intermediates or CollectionExportIntermediates()

    # 1. Instantiate the CollectionModel.
    model = model_cls.from_pretrained(
        **get_model_kwargs(
            model_cls, dict(**additional_model_kwargs, precision=precision)
        )
    )
    components = components or model.component_names
    for name in components:
        if name not in model.component_names:
            raise CLIUserError(f"Invalid component {name}.")
    input_specs = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )

    quantize_jobs: ComponentGroup[hub.client.QuantizeJob] | None = (
        export_intermediates.quantize_jobs
    )
    quantized_models: ComponentGroup[hub.Model] | None = (
        export_intermediates.quantized_models
    )
    compile_jobs: ComponentGroup[hub.client.CompileJob] | None = (
        export_intermediates.compile_jobs
    )
    compiled_models: ComponentGroup[hub.Model] | None = (
        export_intermediates.compiled_models
    )

    # 2. Quantize each non-float component that needs it, unless already
    # covered by a supplied quantize_jobs/quantized_models.
    to_quantize: list[str] = []
    if (
        compile_jobs is None
        and compiled_models is None
        and precision != Precision.float
        and manifest.can_use_quantize_job
    ):
        component_precisions = resolve_component_precisions(
            model, precision, components
        )
        to_quantize = [
            c
            for c in components
            if component_precisions.get(c, precision) != Precision.float
        ]

    # Checked before anything is submitted.
    validate_export_intermediates(
        export_intermediates,
        target_runtime.uses_hub_link,
        keys=components,
        quantize_keys=to_quantize,
    )

    source_models = ComponentGroup[hub.Model](export_intermediates.source_models or {})
    if compile_jobs is None and compiled_models is None:
        if quantize_jobs is not None:
            source_models.update(
                {c: get_target_models(quantize_jobs[c]) for c in to_quantize}
            )
        elif quantized_models is not None:
            source_models.update({c: quantized_models[c] for c in to_quantize})
        to_upload = [c for c in components if c not in source_models]
        if to_upload:
            source_models.update(
                upload_collection_source(model, input_specs, to_upload)
            )

        if to_quantize and quantize_jobs is None and quantized_models is None:
            onnx_compile_jobs = run_collection_compile(
                model,
                model_name,
                device,
                TargetRuntime.ONNX,
                Precision.float,
                source_models,
                input_specs,
                to_quantize,
            )
            new_quantize_jobs = run_collection_quantize(
                {name: component_precisions[name] for name in to_quantize},
                model,
                model_name,
                get_target_models(onnx_compile_jobs),
                num_calibration_samples,
                extra_options=quantize_options,
                input_specs=input_specs,
                app=app,
                calibration_data=export_intermediates.calibration_data,
            )
            quantize_jobs = ComponentGroup(new_quantize_jobs)
            if skip_compiling:
                return CollectionExportResult(quantize_jobs=quantize_jobs)
            source_models.update(get_target_models(new_quantize_jobs))

    # 3. Compile every component, unless already-compiled assets were supplied.
    if compile_jobs is None and compiled_models is None:
        compile_jobs = run_collection_compile(
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
    if skip_linking:
        return CollectionExportResult(
            quantize_jobs=quantize_jobs, compile_jobs=compile_jobs
        )

    link_jobs: ComponentGroup[hub.client.LinkJob] | None = (
        export_intermediates.link_jobs
    )

    # 4. Link (AOT only).
    if compiled_models is None and compile_jobs is not None:
        compiled_models = get_target_models(compile_jobs)
    assert compiled_models is not None
    if target_runtime.uses_hub_link:
        if link_jobs is None:
            link_jobs = run_collection_link(
                compiled_models,
                device,
                model_name,
                model,
                target_runtime,
                extra_options=link_options,
            )
        target_models = get_target_models(link_jobs)
    else:
        target_models = compiled_models

    inference_opts = model.get_hub_inference_options(target_runtime, inference_options)

    # 5. Run on-device inference per component, which also collects profile metrics.
    inference_jobs: ComponentGroup[hub.client.InferenceJob] | None = (
        export_intermediates.inference_jobs
    )
    if inference_jobs is None and not skip_inferencing:
        inference_jobs = run_collection_inference(
            model.sample_inputs(
                input_specs=input_specs,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            inference_opts,
            target_models,
            components,
        )
    inference_jobs = inference_jobs or ComponentGroup()

    # 6. Tool versions — pick any one job from each group.
    if not skip_summary or not skip_downloading:
        tool_versions, tool_versions_from_inference = extract_tool_versions(
            next(iter(inference_jobs.values()), None) if inference_jobs else None,
            next(iter(compile_jobs.values()), None) if compile_jobs else None,
        )
    else:
        tool_versions, tool_versions_from_inference = None, False

    # 7. Download all components together.
    download_path: Path | None = None
    metadata: ModelMetadata | None = None
    if not skip_downloading and tool_versions is not None:
        download_path, metadata = download_collection_model_bundle(
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
            use_case=manifest.use_case,
        )

    inference_metrics: ComponentGroup[pd.DataFrame] | None = None
    if not skip_summary:
        if inference_jobs:
            inference_metrics = ComponentGroup()
            for component_name, inference_job in inference_jobs.items():
                print_profile_summary(inference_job)
                component = model.components[component_name]
                inference_metrics[component_name] = compute_inference_summary(
                    component,
                    inference_job,
                    input_specs[component_name],
                    target_runtime,
                    outputs_to_skip=list(
                        (manifest.outputs_to_skip_validation or {}).keys()
                    ),
                    metrics=manifest.inference_metrics,
                    print_result=True,
                )
        print_tool_versions(tool_versions, tool_versions_from_inference)
        demo_jobs = link_jobs or compile_jobs
        if manifest.has_on_target_demo and demo_jobs:
            print_on_target_demo_cmd(
                list(demo_jobs.values()),
                source_dir,
                device,
            )
        if download_path is not None:
            print(f"{model_name} was saved to {download_path}\n")
        if target_runtime in (TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT):
            print(
                "These models can be deployed on-device using the Genie SDK. "
                "For a full tutorial, please follow the instructions here: "
                "https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
            )

    return CollectionExportResult(
        quantize_jobs=quantize_jobs or None,
        compile_jobs=compile_jobs or None,
        link_jobs=link_jobs or None,
        inference_jobs=inference_jobs or None,
        download_path=download_path,
        tool_versions=tool_versions,
        metadata=metadata,
        inference_metrics=inference_metrics,
    )
