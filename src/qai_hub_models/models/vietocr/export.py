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
from typing import Any

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.common import SampleInputsType
from qai_hub_models.configs.model_metadata import (
    ChipsetAttributes,
    ModelFileMetadata,
    ModelMetadata,
    merge_input_metadata,
    merge_output_metadata,
)
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.vietocr import MODEL_ID, Model
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
from qai_hub_models.utils.input_spec import InputSpec, to_hub_input_specs
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


def upload_model(
    model: BaseModel,
    input_spec: InputSpec | None = None,
) -> hub.Model:
    input_spec = input_spec or model.get_input_spec()
    with tempfile.TemporaryDirectory() as tmpdir:
        return hub.upload_model(str(model.serialize(tmpdir, input_spec)))


def compile_model(
    model: BaseModel,
    model_name: str,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision,
    source_model: hub.Model,
    input_spec: InputSpec | None = None,
    extra_options: str = "",
) -> hub.client.CompileJob:
    input_spec = input_spec or model.get_input_spec()

    model_compile_options = model.get_hub_compile_options(
        target_runtime, precision, extra_options, device
    )
    print(f"Optimizing model {model_name} to run on-device")
    return hub.submit_compile_job(
        model=source_model,
        input_specs=to_hub_input_specs(input_spec),
        device=device,
        name=model_name,
        options=model_compile_options,
    )


def link_model(
    compiled_model: hub.Model,
    device: hub.Device,
    model_name: str,
    model: BaseModel,
    target_runtime: TargetRuntime,
    extra_options: str = "",
) -> hub.client.LinkJob:
    """Link compiled DLC to context binary for AOT."""
    assert target_runtime.is_aot_compiled, (
        f"link_model() requires an AOT runtime, got {target_runtime}"
    )
    link_options = model.get_hub_link_options(target_runtime, extra_options)
    print(f"Linking {model_name} to context binary")
    return hub.submit_link_job(
        [compiled_model],
        device=device,
        name=model_name,
        options=link_options,
    )


def profile_model(
    model_name: str,
    device: hub.Device,
    options: str,
    target_model: hub.Model,
) -> hub.client.ProfileJob:
    print(f"Profiling model {model_name} on a hosted device.")
    return hub.submit_profile_job(
        model=target_model,
        device=device,
        name=model_name,
        options=options,
    )


def inference_model(
    inputs: SampleInputsType,
    model_name: str,
    device: hub.Device,
    options: str,
    target_model: hub.Model,
) -> hub.client.InferenceJob:
    print(f"Running inference for {model_name} on a hosted device with example inputs.")
    return hub.submit_inference_job(
        model=target_model,
        inputs=inputs,
        device=device,
        name=model_name,
        options=options,
    )


def download_model(
    output_dir: os.PathLike | str,
    model: BaseModel,
    runtime: TargetRuntime,
    precision: Precision,
    tool_versions: ToolVersions,
    target_model: hub.Model,
    model_name: str,
    zip_assets: bool,
    hub_device: hub.Device | None = None,
) -> Path:
    output_folder_name = os.path.basename(output_dir)
    output_path = get_next_free_path(output_dir)

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
        metadata_path = dst_path / "metadata.json"
        file_metadata = ModelFileMetadata.from_hub_model(target_model)
        # Merge semantic metadata from get_input_spec()
        merge_input_metadata(file_metadata, model.get_input_spec())
        merge_output_metadata(file_metadata, model.get_output_spec())
        model_metadata = ModelMetadata(
            model_id=MODEL_ID,
            model_name="VietOCR",
            runtime=runtime,
            precision=precision,
            tool_versions=tool_versions,
            model_files={model_file_name: file_metadata},
            chipset_attributes=ChipsetAttributes.from_hub_device(hub_device)
            if runtime.is_aot_compiled
            else None,
        )

        # Dump supplementary files into the model folder
        model.write_supplementary_files(dst_path, model_metadata)

        model_metadata.to_json(metadata_path)
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
    quantized_model_id: str | None = None,
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
    quantized_model_id
        A quantized ONNX hub model id, skips quantizing model.
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
    source_model_to_compile = upload_model(model, input_spec)

    # 2. Converts the PyTorch model to ONNX and quantizes the ONNX model.
    quantize_job: hub.client.QuantizeJob | None = None
    quantized_model: hub.Model | None = None
    if precision != Precision.float:
        if quantized_model_id:
            quantized_model = hub.get_model(quantized_model_id)
            assert quantized_model is not None
        else:
            onnx_compile_result = compile_model(
                model,
                model_name,
                device,
                TargetRuntime.ONNX,
                precision,
                source_model_to_compile,
                input_spec=input_spec,
            )
            onnx_model = onnx_compile_result.get_target_model()
            assert onnx_model is not None, (
                f"ONNX compile job failed: {onnx_compile_result}"
            )
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
    if quantized_model:
        source_model_to_compile = quantized_model
    compile_result = compile_model(
        model,
        model_name,
        device,
        target_runtime,
        precision,
        source_model_to_compile,
        input_spec=input_spec,
        extra_options=compile_options,
    )

    link_result: hub.client.LinkJob | None = None
    target_model: hub.Model | None
    if target_runtime.uses_hub_link:
        compiled_model = compile_result.get_target_model()
        assert compiled_model is not None, f"Compile job failed: {compile_result}"
        link_result = link_model(
            compiled_model,
            device,
            model_name,
            model,
            target_runtime,
        )
        # Extract target models from link jobs for profile/inference
        target_model = link_result.get_target_model()
        assert target_model is not None, f"Link job failed: {link_result}"
    else:
        # For JIT runtimes, extract models from compile jobs
        target_model = compile_result.get_target_model()
        assert target_model is not None, f"Compile job failed: {compile_result}"

    # 4. Profiles the model performance on a real device
    profile_result: hub.client.ProfileJob | None = None
    if not skip_profiling:
        profile_result = profile_model(
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            target_model,
        )

    # 5. Inferences the model on sample inputs
    inference_result: hub.client.InferenceJob | None = None
    if not skip_inferencing:
        inference_result = inference_model(
            model.sample_inputs(
                input_spec=input_spec,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            model.get_hub_profile_options(target_runtime, profile_options),
            target_model,
        )

    # 6. Extracts relevant tool (eg. SDK) versions used to compile and profile this model
    tool_versions: ToolVersions | None = None
    tool_versions_are_from_device_job = False
    if not skip_summary or not skip_downloading:
        if profile_result is not None and profile_result.wait():
            tool_versions = ToolVersions.from_job(profile_result)
            tool_versions_are_from_device_job = True
        elif inference_result is not None and inference_result.wait():
            tool_versions = ToolVersions.from_job(inference_result)
            tool_versions_are_from_device_job = True
        elif compile_result and compile_result.wait():
            tool_versions = ToolVersions.from_job(compile_result)

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
            target_model,
            MODEL_ID,
            zip_assets,
            hub_device=hub_device,
        )

    # 8. Summarizes the results from profiling and inference
    if not skip_summary and profile_result is not None:
        assert profile_result.wait().success, "Job failed: " + profile_result.url
        profile_data: dict[str, Any] = profile_result.download_profile()
        print_profile_metrics_from_job(profile_result, profile_data)

    if not skip_summary and inference_result is not None:
        sample_inputs = model.sample_inputs(input_spec, use_channel_last_format=False)
        torch_out = torch_inference(
            model,
            sample_inputs,
            return_channel_last_output=target_runtime.channel_last_native_execution,
        )
        assert inference_result.wait().success, "Job failed: " + inference_result.url
        ij_output = inference_result.download_output_data()
        assert ij_output is not None
        print_inference_metrics(
            inference_result, ij_output, torch_out, model.get_output_names()
        )

    if not skip_summary:
        print_tool_versions(tool_versions, tool_versions_are_from_device_job)
        print_on_target_demo_cmd(
            link_result or compile_result,
            Path(__file__).parent,
            device,
        )

    if downloaded_model_path:
        print(f"{model_name} was saved to {downloaded_model_path}\n")

    return ExportResult(
        quantize_job=quantize_job,
        compile_job=compile_result,
        link_job=link_result,
        inference_job=inference_result,
        profile_job=profile_result,
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
            TargetRuntime.ONNX,
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
