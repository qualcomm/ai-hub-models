# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, cast

import qai_hub as hub

from qai_hub_models.models.common import Precision
from qai_hub_models.models.llama_v2_7b_chat import Model
from qai_hub_models.models.llama_v2_7b_chat.model import MODEL_ASSET_VERSION, MODEL_ID
from qai_hub_models.utils import model_cache
from qai_hub_models.utils.args import (
    export_parser,
    get_input_spec_kwargs,
    get_model_kwargs,
)
from qai_hub_models.utils.base_model import TargetRuntime
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.export_result import CollectionExportResult, ExportResult
from qai_hub_models.utils.export_without_hub_access import export_without_hub_access
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_on_target_demo_cmd,
    print_profile_metrics_from_job,
)
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub

ALL_COMPONENTS = [
    "Llama2_Part1_Quantized",
    "Llama2_Part2_Quantized",
    "Llama2_Part3_Quantized",
    "Llama2_Part4_Quantized",
]

DEFAULT_COMPONENTS = ALL_COMPONENTS

# Each components is two sub-components linked together with shared weights
ALL_SUB_COMPONENTS = {
    "Llama2_Part1_Quantized": [
        "PromptProcessor_1",
        "TokenGenerator_1",
    ],
    "Llama2_Part2_Quantized": [
        "PromptProcessor_2",
        "TokenGenerator_2",
    ],
    "Llama2_Part3_Quantized": [
        "PromptProcessor_3",
        "TokenGenerator_3",
    ],
    "Llama2_Part4_Quantized": [
        "PromptProcessor_4",
        "TokenGenerator_4",
    ],
}

DEFAULT_EXPORT_DEVICE = "Samsung Galaxy S24 (Family)"
BASE_NAME = "llama_v2_7b_chat"


def export_model(
    device: hub.Device,
    model_name: str = MODEL_ID,
    model_asset_version: int = MODEL_ASSET_VERSION,
    components: list[str] | None = None,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    compile_options: str = "",
    link_options: str = "",
    profile_options: str = "",
    model_cache_mode: CacheMode = CacheMode.ENABLE,
    **additional_model_kwargs: Any,
) -> CollectionExportResult:
    """
    This function accomplishes 6 main tasks:

        1. Instantiates a PyTorch model and converts it to a traced TorchScript format.
        2. Compiles the model to an asset that can be run on device.
        3. Profiles the model performance on real devices.
        4. Inferences the model on sample inputs.
        5. Downloads the model asset to the local directory.
        6. Summarizes the results from profiling and inference.

    Each of the last four steps can be optionally skipped using the input options.

    Parameters
    ----------
    device
        Device for which to export the model (e.g. hub.Device("Samsung Galaxy S25")).
        Full list of available devices can be found by running `hub.get_devices()`.
    model_name
        Model name.
    model_asset_version
        Identifier used as a cache key to store the model asset.
    components
        List of sub-components of the model that will be exported.
        Each component is compiled and profiled separately.
        Defaults to ALL_COMPONENTS if not specified.
    skip_profiling
        If set, skips profiling of compiled model on real devices.
    skip_inferencing
        If set, skips computing on-device outputs from sample data.
    skip_downloading
        If set, skips downloading of compiled model.
    skip_summary
        If set, skips waiting for and summarizing results
        from profiling and inference.
    output_dir
        Directory to store generated assets (e.g. compiled model).
        Defaults to `<cwd>/build/<model_name>`.
    target_runtime
        Which on-device runtime to target. Default is TFLite.
    compile_options
        Additional options to pass when submitting the compile job.
    link_options
        Additional options to pass when submitting the link job.
    profile_options
        Additional options to pass when submitting the profile job.
    model_cache_mode
        Whether to cache uploaded AI Hub Workbench model during export.
        If enable, caches uploaded model (i.e. re-uses uploaded AI Hub Workbench model from cache).
        If disable, disables caching i.e. no reading from and write to cache.
        If overwrite, ignores and overwrites previous cache with newly uploaded AI Hub Workbench model instead.
    **additional_model_kwargs
        Additional optional kwargs used to customize
        `model_cls.from_pretrained`

    Returns
    -------
    result : CollectionExportResult
        A Mapping from component_name to a 3-tuple of:
        * A LinkJob object containing metadata about the link job submitted to hub.
        * A ProfileJob containing metadata about the profile job (None if profiling skipped).
        * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
    """
    # Resolves all of the device attributes from a partially specified hub.Device
    hub_devices = hub.get_devices(
        name=device.name,
        attributes=device.attributes,
        os=device.os,
    )
    if len(hub_devices) == 0:
        raise ValueError(
            f"No device found compatible with the supplied name ({device.name}), os ({device.os}), and attributes ({device.attributes})."
        )
    device = hub_devices[-1]
    model_name = BASE_NAME
    output_path = Path(output_dir or Path.cwd() / "build" / model_name)
    component_arg = components
    components = components or DEFAULT_COMPONENTS
    for component_name in components:
        if component_name not in ALL_COMPONENTS:
            raise ValueError(f"Invalid component {component_name}.")
    if not can_access_qualcomm_ai_hub():
        static_model_path = export_without_hub_access(
            "llama_v2_7b_chat",
            device,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            Precision.w4a16,
            compile_options + " " + profile_options + " " + link_options,
            component_arg,
        )
        return CollectionExportResult(
            components={
                component_name: ExportResult() for component_name in components
            },
            download_path=static_model_path,
        )

    # 1. Initialize PyTorch model
    model = Model.from_pretrained(**get_model_kwargs(Model, additional_model_kwargs))

    compile_jobs_to_link: dict[str, list[hub.client.CompileJob]] = {}
    compile_jobs: dict[str, hub.client.CompileJob] = {}
    profile_options_per_sub_component: dict[str, str] = {}
    link_jobs: dict[str, hub.client.LinkJob] = {}

    for component_name in components:
        for sub_component_name in ALL_SUB_COMPONENTS[component_name]:
            # Load model part
            component = model.load_model_part(sub_component_name)

            input_spec = component.get_input_spec(
                **get_input_spec_kwargs(component, additional_model_kwargs)
            )

            # Create source .aimet model
            source_model = component.convert_to_hub_source_model(
                target_runtime,
                output_path,
                input_spec,
                external_onnx_weights=True,
                output_names=component.get_output_names(),
            )

            seq_length, context_length = input_spec["attention_mask"][0][-2:]
            current_model = model_cache.get_or_create_cached_model(
                model_name=model_name,
                model_asset_version=model_asset_version,
                cache_name=sub_component_name,
                cache_mode=model_cache_mode,
                model_path=str(source_model),
                additional_keys={
                    "context_length": str(context_length),
                    "sequence_length": str(seq_length),
                },
            )

            if target_runtime == TargetRuntime.TFLITE:
                quant_calibration_data = None
            else:
                quant_calibration_data = component.get_calibration_data(  # type: ignore[operator, unused-ignore]
                    input_spec=input_spec
                )

            # 2. Compile the models to an on-device asset
            model_compile_options = component.get_hub_compile_options(
                target_runtime, Precision.w8a16, compile_options
            )
            print(f"Optimizing model {sub_component_name} to run on-device")
            submitted_compile_job = hub.submit_compile_job(
                model=current_model,
                input_specs=input_spec,
                device=device,
                name=f"{model_name}_{sub_component_name}",
                calibration_data=quant_calibration_data,
                options=model_compile_options,
            )
            assert isinstance(submitted_compile_job, hub.CompileJob)
            if component_name not in compile_jobs_to_link:
                compile_jobs_to_link[component_name] = []

            compile_jobs_to_link[component_name].append(
                cast(hub.client.CompileJob, submitted_compile_job)
            )
            compile_jobs[sub_component_name] = cast(
                hub.client.CompileJob, submitted_compile_job
            )

            profile_options_per_sub_component[sub_component_name] = (
                component.get_hub_profile_options(target_runtime, profile_options)
            )
            link_options = component.get_hub_link_options(target_runtime, link_options)

            # Free model part to reduce memory-pressure
            del component

    for component_name, cjobs in compile_jobs_to_link.items():
        models = [cast(hub.Model, cjob.get_target_model()) for cjob in cjobs]
        # Link Prompt processor and Token generator
        link_jobs[component_name] = hub.submit_link_job(
            models,  # type: ignore[arg-type]
            device=device,
            name=f"{model_name}_{component_name}",
            options=link_options,
        )

    # 4. Profile the model assets on real devices
    profile_jobs: dict[str, hub.client.ProfileJob] = {}
    if not skip_profiling:
        for component_name in components:
            hub_model = link_jobs[component_name].get_target_model()
            for sub_component_name in ALL_SUB_COMPONENTS[component_name]:
                profile_options_all = profile_options_per_sub_component[
                    sub_component_name
                ]
                print(f"Profiling model {component_name} on a hosted device.")
                submitted_profile_job = hub.submit_profile_job(
                    model=hub_model,
                    device=device,
                    name=f"{model_name}_{sub_component_name}",
                    options=profile_options_all,
                )
                profile_jobs[sub_component_name] = cast(
                    hub.client.ProfileJob, submitted_profile_job
                )

    # 5. Run inference on-device with sample inputs
    inference_jobs: dict[str, hub.client.InferenceJob] = {}
    if not skip_inferencing:
        for component_name in components:
            for sub_component_name in ALL_SUB_COMPONENTS[component_name]:
                print(
                    f"Running inference for {sub_component_name} on a hosted device with example inputs."
                )
                # Load model with no-AIMET mode
                component = model.load_model_part(sub_component_name)
                profile_options_all = profile_options_per_sub_component[
                    sub_component_name
                ]
                # Load individual model part
                sample_inputs = component.sample_inputs()
                submitted_inference_job = hub.submit_inference_job(
                    model=link_jobs[component_name].get_target_model(),
                    inputs=sample_inputs,
                    device=device,
                    name=f"{model_name}_{sub_component_name}",
                    options=profile_options_all,
                )
                inference_jobs[sub_component_name] = cast(
                    hub.client.InferenceJob, submitted_inference_job
                )

    # 6. Download the model assets to a local file
    if not skip_downloading:
        os.makedirs(output_path, exist_ok=True)
        for component_name, link_job in link_jobs.items():
            target_model = link_job.get_target_model()
            assert target_model is not None, "Link job did not produce a target model."
            target_model.download(str(output_path / f"{component_name}.bin"))

    # 7. Summarize the results from profiling and inference
    if not skip_summary and not skip_profiling:
        for component_name in components:
            for sub_component_name in ALL_SUB_COMPONENTS[component_name]:
                profile_job = profile_jobs[sub_component_name]
                assert profile_job is not None and profile_job.wait().success
                profile_data: dict[str, Any] = profile_job.download_profile()
                print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and not skip_inferencing:
        for component_name in components:
            for sub_component_name in ALL_SUB_COMPONENTS[component_name]:
                inference_job = inference_jobs[sub_component_name]
                # Load individual model part
                component = model.load_model_part(sub_component_name)
                # Get ordered model output names
                output_names = component.get_output_names()
                sample_inputs = component.sample_inputs()
                torch_out = torch_inference(component, sample_inputs)
                assert inference_job is not None and inference_job.wait().success
                inference_result = inference_job.download_output_data()
                assert inference_result is not None
                print_inference_metrics(
                    inference_job,
                    inference_result,
                    torch_out,
                    output_names=output_names,
                )

    if not skip_summary:
        print_on_target_demo_cmd(
            link_jobs.values(), Path(__file__).parent.resolve(), device
        )

    print(
        "These models can be deployed on-device using the Genie SDK. For a full tutorial, please follow the instructions here: https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
    )

    return CollectionExportResult(
        components={
            component_name: ExportResult(
                compile_job=compile_jobs[sub_component_name],
                link_job=link_jobs[component_name],
                profile_job=profile_jobs.get(sub_component_name),
                inference_job=inference_jobs.get(sub_component_name),
            )
            for component_name in components
            for sub_component_name in ALL_SUB_COMPONENTS[component_name]
        }
    )


def main(argv: list[str] | None = None) -> None:
    warnings.filterwarnings("ignore")
    parser = export_parser(
        model_cls=Model,
        export_fn=export_model,
        components=ALL_COMPONENTS,
        supported_precision_runtimes={Precision.w8a16: [TargetRuntime.GENIE]},
        default_export_device=DEFAULT_EXPORT_DEVICE,
    )
    args = parser.parse_args(argv)
    export_model(
        model_name=MODEL_ID, model_asset_version=MODEL_ASSET_VERSION, **vars(args)
    )


if __name__ == "__main__":
    main()
