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

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.args import get_export_model_name, get_model_kwargs
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.export.compile import (
    run_collection_compile,
)
from qai_hub_models.utils.export.context import (
    resolve_model_app_cls,
    resolve_model_cls,
    resolve_model_dir,
    resolve_model_display_name,
)
from qai_hub_models.utils.export.download import download_collection_model_bundle
from qai_hub_models.utils.export.inference import run_collection_inference
from qai_hub_models.utils.export.link import run_collection_link
from qai_hub_models.utils.export.profile import run_collection_profile
from qai_hub_models.utils.export.quantize import (
    resolve_component_precisions,
    run_collection_quantize,
)
from qai_hub_models.utils.export.result import CollectionExportResult, ComponentGroup
from qai_hub_models.utils.export.summary import (
    extract_tool_versions,
    print_inference_summary,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_collection_source
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.printing import print_on_target_demo_cmd, print_tool_versions
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_URL,
    assert_success_and_get_target_models,
    get_device_and_chipset_name,
)


def export_model(
    model_id: str,
    device: hub.Device,
    components: list[str] | None = None,
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    num_calibration_samples: int | None = None,
    quantized_model_id: dict[str, str] | None = None,
    skip_compiling: bool = False,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    compile_options: str = "",
    quantize_options: str = "",
    profile_options: str = "",
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> CollectionExportResult:
    """
    Run the eight-step recipe over every selected component:

        1. Instantiate the CollectionModel and upload each component's source
        2. Quantize each non-float component
        3. Compile each component to ``target_runtime``
        4. Link each compiled DLC to a context binary (AOT runtimes only)
        5. Profile each component on a real device
        6. Run on-device inference per component on sample inputs
        7. Extract SDK / runtime versions
        8. Download all components together with a combined metadata.json

    Parameters
    ----------
    model_id
        Model folder name.
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
    quantized_model_id
        Per-component dict of quantized ONNX hub model ids; skips the quantize job.
    skip_compiling
        If set, skips compiling.
    skip_profiling
        If set, skips profiling on real devices.
    skip_inferencing
        If set, skips on-device inference comparison.
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
    profile_options
        Extra options for the profile job.
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

    if not can_access_qualcomm_ai_hub():
        raise RuntimeError(
            "Could not find AI Hub credentials. Sign up at "
            f"{_AIHUB_URL} for free access. To use pre-released assets "
            "without AI Hub credentials, run "
            f"`qai-hub-models fetch {model_id}` instead."
        )

    manifest = QAIHMModelManifest.from_model(model_id)
    model_cls = resolve_model_cls(model_id)
    app = resolve_model_app_cls(model_id)
    display_name = resolve_model_display_name(model_id)
    source_dir = resolve_model_dir(model_id)
    model_name = get_export_model_name(
        model_cls, model_id, precision, additional_model_kwargs
    )
    output_path = Path(output_dir or Path.cwd() / "export_assets")

    _, chipset = get_device_and_chipset_name(device)

    # 1. Instantiate the CollectionModel and upload each component's source.
    model = model_cls.from_pretrained(
        **get_model_kwargs(
            model_cls, dict(**additional_model_kwargs, precision=precision)
        )
    )
    components = components or model.component_names
    for name in components:
        if name not in model.component_names:
            raise ValueError(f"Invalid component {name}.")
    input_specs = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )
    source_models = upload_collection_source(model, input_specs, components)

    # 2. Quantize each non-float component.
    quantize_jobs: ComponentGroup[hub.client.QuantizeJob] | None = None
    if precision != Precision.float and manifest.can_use_quantize_job:
        component_precisions = resolve_component_precisions(
            model, precision, components
        )
        if quantized_model_id:
            prequantized = ComponentGroup(
                {
                    name: hub.get_model(quantized_model_id[name])
                    for name in components
                    if name in quantized_model_id
                }
            )
            source_models.update(prequantized)
        else:
            onnx_components = [
                c
                for c, p in component_precisions.items()
                if c in components and p != Precision.float
            ]
            onnx_compile_jobs = run_collection_compile(
                model,
                model_name,
                device,
                TargetRuntime.ONNX,
                precision,
                source_models,
                input_specs,
                onnx_components,
            )
            onnx_models = assert_success_and_get_target_models(onnx_compile_jobs)
            quantize_jobs = run_collection_quantize(
                {name: component_precisions[name] for name in onnx_components},
                model,
                model_name,
                onnx_models,
                num_calibration_samples,
                extra_options=quantize_options,
                input_specs=input_specs,
                app=app,
            )
            if skip_compiling:
                return CollectionExportResult(quantize_jobs=quantize_jobs)
            quantized = assert_success_and_get_target_models(quantize_jobs)
            source_models.update(quantized)

    # 3. Compile each component.
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

    # 4. Link (AOT only).
    link_jobs: ComponentGroup[hub.client.LinkJob] | None = None
    if target_runtime.uses_hub_link:
        compiled = assert_success_and_get_target_models(compile_jobs)
        link_jobs = run_collection_link(
            compiled,
            device,
            model_name,
            model,
            target_runtime,
        )
    target_models = assert_success_and_get_target_models(link_jobs or compile_jobs)

    profile_opts = model.get_hub_profile_options(target_runtime, profile_options)

    # 5. Profile each component.
    profile_jobs: ComponentGroup[hub.client.ProfileJob] | None = None
    if not skip_profiling:
        profile_jobs = run_collection_profile(
            model_name,
            device,
            profile_opts,
            target_models,
            components,
        )

    # 6. Inference per component.
    inference_jobs: ComponentGroup[hub.client.InferenceJob] | None = None
    if not skip_inferencing:
        inference_jobs = run_collection_inference(
            model.sample_inputs(
                input_specs=input_specs,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            profile_opts,
            target_models,
            components,
        )

    # 7. Tool versions — pick any one job from each group.
    if not skip_summary or not skip_downloading:
        tool_versions, tool_versions_from_device = extract_tool_versions(
            next(iter(profile_jobs.values()), None) if profile_jobs else None,
            next(iter(inference_jobs.values()), None) if inference_jobs else None,
            next(iter(compile_jobs.values()), None) if compile_jobs else None,
        )
    else:
        tool_versions, tool_versions_from_device = None, False

    # 8. Download all components together.
    download_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        download_path = download_collection_model_bundle(
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
            for job in profile_jobs.values():
                print_profile_summary(job)
        if inference_jobs:
            for component_name, inference_job in inference_jobs.items():
                component = model.components[component_name]
                print_inference_summary(
                    component,
                    inference_job,
                    input_specs[component_name],
                    target_runtime,
                    outputs_to_skip=list(
                        (manifest.outputs_to_skip_validation or {}).keys()
                    ),
                    metrics=manifest.inference_metrics,
                )
        print_tool_versions(tool_versions, tool_versions_from_device)
        if manifest.has_on_target_demo:
            print_on_target_demo_cmd(
                list((link_jobs or compile_jobs).values()),
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
        quantize_jobs=quantize_jobs,
        compile_jobs=compile_jobs,
        link_jobs=link_jobs,
        profile_jobs=profile_jobs,
        inference_jobs=inference_jobs,
        download_path=download_path,
        tool_versions=tool_versions,
    )
