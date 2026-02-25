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
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.metadata_yaml import ModelFileMetadata, ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.models.inception_v3 import MODEL_ID, Model
from qai_hub_models.utils import quantization as quantization_utils
from qai_hub_models.utils.args import (
    export_parser,
    get_export_model_name,
    get_input_spec_kwargs,
    get_model_kwargs,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.compare import torch_inference
from qai_hub_models.utils.export_result import ExportResult
from qai_hub_models.utils.export_without_hub_access import export_without_hub_access
from qai_hub_models.utils.input_spec import InputSpec, make_torch_inputs
from qai_hub_models.utils.onnx.helpers import download_and_unzip_workbench_onnx_model
from qai_hub_models.utils.path_helpers import get_next_free_path
from qai_hub_models.utils.printing import (
    print_inference_metrics,
    print_on_target_demo_cmd,
    print_profile_metrics_from_job,
    print_tool_versions,
)
from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub


def quantize_model(
    precision: Precision,
    model: BaseModel,
    model_name: str,
    onnx_model: hub.Model,
    num_calibration_samples: int | None,
    extra_options: str = "",
    input_spec: InputSpec | None = None,
) -> hub.client.QuantizeJob:
    input_spec = input_spec or model.get_input_spec()
    print(f"Quantizing {model_name}.")
    if not precision.activations_type or not precision.weights_type:
        raise ValueError(
            "Quantization is only supported if both weights and activations are quantized."
        )

    calibration_data = quantization_utils.get_calibration_data(
        model, input_spec, num_calibration_samples
    )
    return hub.submit_quantize_job(
        model=onnx_model,
        calibration_data=calibration_data,
        activations_dtype=precision.activations_type,
        weights_dtype=precision.weights_type,
        name=model_name,
        options=model.get_hub_quantize_options(precision, extra_options),
    )


def compile_model(
    model: BaseModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_model: hub.Model | None = None,
    input_spec: InputSpec | None = None,
    extra_options: str = "",
) -> hub.client.CompileJob:
    input_spec = input_spec or model.get_input_spec()
    if source_model:
        model_to_compile = source_model
    else:
        # Trace the model
        model_to_compile = torch.jit.trace(
            model.to("cpu"), make_torch_inputs(input_spec)
        )

    model_compile_options = model.get_hub_compile_options(
        target_runtime, precision, extra_options, device
    )
    print(f"Optimizing model {model_name} to run on-device")
    submitted_compile_job = hub.submit_compile_job(
        model=model_to_compile,
        input_specs=input_spec,
        device=device,
        name=model_name,
        options=model_compile_options,
    )
    return cast(hub.client.CompileJob, submitted_compile_job)


def profile_model(
    model_name: str,
    device: hub.Device,
    options: str,
    compile_job: hub.client.CompileJob,
) -> hub.client.ProfileJob:
    print(f"Profiling model {model_name} on a hosted device.")
    submitted_profile_job = hub.submit_profile_job(
        model=compile_job.get_target_model(),
        device=device,
        name=model_name,
        options=options,
    )
    return cast(hub.client.ProfileJob, submitted_profile_job)


def inference_model(
    inputs: SampleInputsType,
    model_name: str,
    device: hub.Device,
    options: str,
    compile_job: hub.client.CompileJob,
) -> hub.client.InferenceJob:
    print(f"Running inference for {model_name} on a hosted device with example inputs.")
    submitted_inference_job = hub.submit_inference_job(
        model=compile_job.get_target_model(),
        inputs=inputs,
        device=device,
        name=model_name,
        options=options,
    )
    return cast(hub.client.InferenceJob, submitted_inference_job)


def download_model(
    output_dir: os.PathLike | str,
    model: BaseModel,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    compile_job: hub.client.CompileJob,
    model_name: str,
    zip_assets: bool,
) -> Path:
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

    target_model = compile_job.get_target_model()
    assert target_model, f"Compile Job Failed:\n{compile_job}"

    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = Path(tmpdir) / output_folder_name
        dst_path.mkdir()

        if target_model.model_type == hub.SourceModelType.ONNX:
            onnx_result = download_and_unzip_workbench_onnx_model(
                target_model, dst_path, model_name
            )
            model_file_name = onnx_result.onnx_graph_name
        else:
            downloaded_path = target_model.download(os.path.join(dst_path, model_name))
            model_file_name = os.path.basename(downloaded_path)

        # Extract and save metadata alongside downloaded model
        metadata_path = dst_path / "metadata.yaml"
        file_metadata = ModelFileMetadata.from_hub_model(target_model)
        model_metadata = ModelMetadata(
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files={model_file_name: file_metadata},
        )

        # Dump supplementary files into the model folder
        model.write_supplementary_files(dst_path, model_metadata)

        model_metadata.to_yaml(metadata_path)
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
    precision: Precision = Precision.float,
    num_calibration_samples: int | None = None,
    skip_compiling: bool = False,
    skip_profiling: bool = False,
    skip_inferencing: bool = False,
    skip_downloading: bool = False,
    skip_summary: bool = False,
    output_dir: str | None = None,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    compile_options: str = "",
    quantize_options: str = "",
    profile_options: str = "",
    fetch_static_assets: str | None = None,
    zip_assets: bool = False,
    **additional_model_kwargs: Any,
) -> ExportResult:
    """
    This function executes the following recipe:

        1. Instantiates a PyTorch model and converts it to a traced TorchScript format
        2. Converts the PyTorch model to ONNX and quantizes the ONNX model.
        3. Compiles the model to an asset that can be run on device
        4. Profiles the model performance on a real device
        5. Inferences the model on sample inputs
        6. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
        7. Downloads the model asset to the local directory
        8. Summarizes the results from profiling and inference

    Each of the last 6 steps can be optionally skipped using the input options.

    Parameters
    ----------
    device
        Device for which to export the model (e.g., hub.Device("Samsung Galaxy S25")).
        Full list of available devices can be found by running `hub.get_devices()`.
    precision
        The precision to which this model should be quantized.
        Quantization is skipped if the precision is float.
    num_calibration_samples
        The number of calibration data samples
        to use for quantization. If not set, uses the default number
        specified by the dataset. If model doesn't have a calibration dataset
        specified, this must be None.
    skip_compiling
        If set, skips compiling of model to format that can run on device.
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
        Defaults to `<cwd>/export_assets`.
    target_runtime
        Which on-device runtime to target. Default is TFLite.
    compile_options
        Additional options to pass when submitting the compile job.
    quantize_options
        Additional options to pass when submitting the quantize job.
    profile_options
        Additional options to pass when submitting the profile job.
    fetch_static_assets
        If set, known assets are fetched from the given version rather than re-computing them. Can be passed as "latest" or "v<version>".
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Additional optional kwargs used to customize
        `model_cls.from_pretrained` and `model.get_input_spec`

    Returns
    -------
    ExportResult
        * A CompileJob object containing metadata about the compile job submitted to hub (None if compiling skipped).
        * An InferenceJob containing metadata about the inference job (None if inferencing skipped).
        * A ProfileJob containing metadata about the profile job (None if profiling skipped).
        * A QuantizeJob object containing metadata about the quantize job submitted to hub
        * The path to the downloaded model folder (or zip), or None if one or more of: skip_downloading is True, fetch_static_assets is set, or AI Hub Workbench is not accessible
    """
    model_name = get_export_model_name(
        Model, MODEL_ID, precision, additional_model_kwargs
    )

    output_path = Path(output_dir or Path.cwd() / "export_assets")
    if fetch_static_assets or not can_access_qualcomm_ai_hub():
        static_model_path = export_without_hub_access(
            MODEL_ID,
            device,
            skip_profiling,
            skip_inferencing,
            skip_downloading,
            skip_summary,
            output_path,
            target_runtime,
            precision,
            quantize_options + compile_options + profile_options,
            qaihm_version_tag=fetch_static_assets,
        )
        return ExportResult(download_path=static_model_path)

    hub_device = hub.get_devices(
        name=device.name, attributes=device.attributes, os=device.os
    )[-1]
    chipset_attr = next(
        (attr for attr in hub_device.attributes if "chipset" in attr), None
    )
    chipset = chipset_attr.split(":")[-1] if chipset_attr else None

    # 1. Instantiates a PyTorch model and converts it to a traced TorchScript format
    model = Model.from_pretrained(
        **get_model_kwargs(Model, dict(**additional_model_kwargs, precision=precision))
    )
    input_spec = model.get_input_spec(
        **get_input_spec_kwargs(model, additional_model_kwargs)
    )

    # 2. Converts the PyTorch model to ONNX and quantizes the ONNX model.
    quantize_job: hub.client.QuantizeJob | None = None
    quantized_model: hub.Model | None = None
    if precision != Precision.float:
        onnx_compile_job = compile_model(
            model,
            model_name,
            device,
            TargetRuntime.ONNX,
            precision,
            input_spec=input_spec,
        )
        onnx_model = onnx_compile_job.get_target_model()
        assert onnx_model is not None, f"ONNX compile job failed: {onnx_compile_job}"
        quantize_job = quantize_model(
            precision,
            model,
            model_name,
            onnx_model,
            num_calibration_samples,
            quantize_options,
            input_spec,
        )
        if skip_compiling:
            return ExportResult(quantize_job=quantize_job)
        quantized_model = quantize_job.get_target_model()
        assert quantized_model is not None, f"Quantize job failed: {quantize_job}"

    # 3. Compiles the model to an asset that can be run on device
    compile_job = compile_model(
        model,
        model_name,
        device,
        target_runtime,
        precision,
        quantized_model,
        input_spec=input_spec,
        extra_options=compile_options,
    )

    # 4. Profiles the model performance on a real device
    profile_job: hub.client.ProfileJob | None = None
    if not skip_profiling:
        profile_job = profile_model(
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            compile_job,
        )

    # 5. Inferences the model on sample inputs
    inference_job: hub.client.InferenceJob | None = None
    if not skip_inferencing:
        inference_job = inference_model(
            model.sample_inputs(
                input_spec=input_spec,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            compile_job,
        )

    # 6. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
    tool_versions: ToolVersions | None = None
    tool_versions_are_from_device_job = False
    if not skip_summary or not skip_downloading:
        if profile_job is not None and profile_job.wait():
            tool_versions = ToolVersions.from_job(profile_job)
            tool_versions_are_from_device_job = True
        elif inference_job is not None and inference_job.wait():
            tool_versions = ToolVersions.from_job(inference_job)
            tool_versions_are_from_device_job = True
        elif compile_job and compile_job.wait():
            tool_versions = ToolVersions.from_job(compile_job)

    # 7. Downloads the model asset to the local directory
    downloaded_model_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        model_directory = output_path / ASSET_CONFIG.get_release_asset_name(
            MODEL_ID, target_runtime, precision, chipset
        )
        downloaded_model_path = download_model(
            model_directory,
            model,
            target_runtime,
            precision,
            tool_versions,
            compile_job,
            MODEL_ID,
            zip_assets,
        )

    # 8. Summarizes the results from profiling and inference
    if not skip_summary and profile_job is not None:
        assert profile_job.wait().success, "Job failed: " + profile_job.url
        profile_data: dict[str, Any] = profile_job.download_profile()
        print_profile_metrics_from_job(profile_job, profile_data)

    if not skip_summary and inference_job is not None:
        sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)
        torch_out = torch_inference(
            model,
            sample_inputs,
            return_channel_last_output=target_runtime.channel_last_native_execution,
        )
        assert inference_job.wait().success, "Job failed: " + inference_job.url
        inference_result = inference_job.download_output_data()
        assert inference_result is not None
        print_inference_metrics(
            inference_job, inference_result, torch_out, model.get_output_names()
        )

    if not skip_summary:
        print_tool_versions(tool_versions, tool_versions_are_from_device_job)
        print_on_target_demo_cmd(compile_job, Path(__file__).parent, device)

    if downloaded_model_path:
        print(f"{model_name} was saved to {downloaded_model_path}\n")

    return ExportResult(
        compile_job=compile_job,
        inference_job=inference_job,
        profile_job=profile_job,
        quantize_job=quantize_job,
        download_path=downloaded_model_path,
        tool_versions=tool_versions,
    )


def main() -> None:
    warnings.filterwarnings("ignore")
    supported_precision_runtimes: dict[Precision, list[TargetRuntime]] = {
        Precision.float: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
        ],
        Precision.w8a8: [
            TargetRuntime.TFLITE,
            TargetRuntime.QNN_DLC,
            TargetRuntime.QNN_CONTEXT_BINARY,
            TargetRuntime.ONNX,
            TargetRuntime.PRECOMPILED_QNN_ONNX,
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
