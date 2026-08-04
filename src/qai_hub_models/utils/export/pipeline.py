# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
End-to-end export pipeline for a single (non-collection, non-precompiled) model.

A reader can scan the eight numbered comments in :func:`export_model` to see
the full recipe; each step's implementation lives in its own sibling module
(``upload.py``, ``quantize.py``, ``compile.py``, etc.).
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
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.compile import run_compile
from qai_hub_models.utils.export.context import (
    resolve_model_cls,
    resolve_model_dir,
    resolve_model_display_name,
)
from qai_hub_models.utils.export.download import download_model_bundle
from qai_hub_models.utils.export.inference import run_inference
from qai_hub_models.utils.export.link import run_link
from qai_hub_models.utils.export.profile import run_profile
from qai_hub_models.utils.export.quantize import run_quantize
from qai_hub_models.utils.export.result import ExportResult
from qai_hub_models.utils.export.summary import (
    extract_tool_versions,
    print_inference_summary,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_source
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.printing import print_on_target_demo_cmd, print_tool_versions
from qai_hub_models.utils.qai_hub_helpers import _AIHUB_URL, get_device_and_chipset_name


def export_model(
    model_id: str,
    device: hub.Device,
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    num_calibration_samples: int | None = None,
    quantized_model_id: str | None = None,
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
) -> ExportResult:
    """
    Run the eight-step export recipe end-to-end:

        1. Instantiate the PyTorch model and upload its TorchScript form to AI Hub
        2. Quantize the model (skipped for float)
        3. Compile the (possibly quantized) source model to the target runtime
        4. Link the compiled DLC to a context binary (AOT runtimes only)
        5. Profile the compiled model on a real device
        6. Run on-device inference on sample inputs
        7. Extract the SDK / runtime versions used by the jobs
        8. Download the compiled asset locally

    Parameters
    ----------
    model_id
        Model folder name (e.g. ``yolov8_det``); used as the asset and job-name root.
    device
        Hub device to export for (e.g. ``hub.Device("Samsung Galaxy S25")``).
    precision
        Target precision for the export.
    target_runtime
        On-device runtime to target. Defaults to TFLite.
    num_calibration_samples
        Number of calibration data samples to use for quantization. If None,
        uses the default specified by the dataset.
    quantized_model_id
        A quantized ONNX hub model id; if set, skips the quantize job.
    skip_compiling
        If set, skips compiling the model to an on-device format.
    skip_profiling
        If set, skips profiling the compiled model on real devices.
    skip_inferencing
        If set, skips computing on-device outputs from sample data.
    skip_downloading
        If set, skips downloading the compiled model.
    skip_summary
        If set, skips waiting for and summarizing results.
    output_dir
        Directory to store generated assets. Defaults to ``<cwd>/export_assets``.
    compile_options
        Extra options to pass when submitting the compile job.
    quantize_options
        Extra options to pass when submitting the quantize job.
    profile_options
        Extra options to pass when submitting the profile job.
    zip_assets
        If set, zip the assets after downloading.
    **additional_model_kwargs
        Any additional keyword arguments passed to ``Model.from_pretrained``.

    Returns
    -------
    ExportResult
        Jobs submitted, downloaded asset path, and resolved tool versions.
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
    display_name = resolve_model_display_name(model_id)
    source_dir = resolve_model_dir(model_id)
    model_name = get_export_model_name(
        model_cls, model_id, precision, additional_model_kwargs
    )
    output_path = Path(output_dir or Path.cwd() / "export_assets")

    if not manifest.can_use_quantize_job and precision != Precision.float:
        assert precision in manifest.supported_precisions, (
            f"Precision {precision!s} is not supported by {model_name}"
        )

    _, chipset = get_device_and_chipset_name(device)

    # 1. Instantiate the PyTorch model and upload its TorchScript form.
    model = model_cls.from_pretrained(
        **get_model_kwargs(
            model_cls, dict(**additional_model_kwargs, precision=precision)
        )
    )
    input_spec = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )
    source_model = upload_source(model, input_spec)

    # 2. Quantize (float skips this entirely; aimet-quantized models also skip).
    quantize_job: hub.client.QuantizeJob | None = None
    if precision != Precision.float and manifest.can_use_quantize_job:
        if quantized_model_id:
            prequantized = hub.get_model(quantized_model_id)
            assert prequantized is not None
            source_model = prequantized
        else:
            onnx_compile_job = run_compile(
                model,
                model_name,
                device,
                TargetRuntime.ONNX,
                precision,
                source_model,
                input_spec,
            )
            onnx_model = onnx_compile_job.get_target_model()
            assert onnx_model is not None, (
                f"ONNX compile job failed: {onnx_compile_job}"
            )
            quantize_job = run_quantize(
                precision,
                model,
                model_name,
                onnx_model,
                num_calibration_samples,
                extra_options=quantize_options,
                input_spec=input_spec,
            )
            if skip_compiling:
                return ExportResult(quantize_job=quantize_job)
            quantized_model = quantize_job.get_target_model()
            assert quantized_model is not None, f"Quantize job failed: {quantize_job}"
            source_model = quantized_model

    # 3. Compile.
    compile_job = run_compile(
        model,
        model_name,
        device,
        target_runtime,
        precision,
        source_model,
        input_spec,
        extra_options=compile_options,
        calibration_data=_aimet_calibration_data(model, manifest),
    )

    # 4. Link (AOT runtimes only). Only wait on the compile job if a downstream
    # step actually needs the compiled artifact.
    needs_target_model = (
        target_runtime.uses_hub_link
        or not skip_profiling
        or not skip_inferencing
        or not skip_downloading
    )
    link_job: hub.client.LinkJob | None = None
    target_model: hub.Model | None = None
    if needs_target_model:
        compiled_model = compile_job.get_target_model()
        assert compiled_model is not None, f"Compile job failed: {compile_job}"
        if target_runtime.uses_hub_link:
            link_job = run_link(
                compiled_model, device, model_name, model, target_runtime
            )
            target_model = link_job.get_target_model()
        else:
            target_model = compiled_model
        assert target_model is not None, "Link job did not produce a target model"

    profile_opts = model.get_hub_profile_options(target_runtime, profile_options)

    # 5. Profile.
    profile_job: hub.client.ProfileJob | None = None
    if not skip_profiling:
        assert target_model is not None
        profile_job = run_profile(model_name, device, profile_opts, target_model)

    # 6. Inference.
    inference_job: hub.client.InferenceJob | None = None
    if not skip_inferencing:
        assert target_model is not None
        inference_job = run_inference(
            model.sample_inputs(
                input_spec=input_spec,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            profile_opts,
            target_model,
        )

    # 7. Tool versions (needed for both download metadata and the summary).
    if not skip_summary or not skip_downloading:
        tool_versions, tool_versions_from_device = extract_tool_versions(
            profile_job,
            inference_job,
            compile_job,
        )
    else:
        tool_versions, tool_versions_from_device = None, False

    # 8. Download.
    download_path: Path | None = None
    if not skip_downloading and tool_versions is not None:
        assert target_model is not None
        download_path = download_model_bundle(
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
        if profile_job is not None:
            print_profile_summary(profile_job)
        if inference_job is not None:
            print_inference_summary(
                model,
                inference_job,
                input_spec,
                target_runtime,
                outputs_to_skip=list(
                    (manifest.outputs_to_skip_validation or {}).keys()
                ),
                metrics=manifest.inference_metrics,
            )
        print_tool_versions(tool_versions, tool_versions_from_device)
        if manifest.has_on_target_demo:
            print_on_target_demo_cmd(link_job or compile_job, source_dir, device)
        if download_path is not None:
            print(f"{model_name} was saved to {download_path}\n")
        if target_runtime in (TargetRuntime.GENIE, TargetRuntime.GENIEX_QAIRT):
            print(
                "These models can be deployed on-device using the Genie SDK. "
                "For a full tutorial, please follow the instructions here: "
                "https://github.com/quic/ai-hub-apps/tree/main/tutorials/llm_on_genie."
            )

    return ExportResult(
        quantize_job=quantize_job,
        compile_job=compile_job,
        link_job=link_job,
        profile_job=profile_job,
        inference_job=inference_job,
        download_path=download_path,
        tool_versions=tool_versions,
    )


def _aimet_calibration_data(
    model: BaseModel, manifest: QAIHMModelManifest
) -> hub.Dataset | None:
    """AIMET models bake calibration data into the compile job rather than a quantize job."""
    if not (manifest.is_aimet and manifest.num_calibration_samples):
        return None
    get_calib = getattr(model, "get_calibration_data", None)
    if get_calib is None:
        return None
    return get_calib(num_samples=manifest.num_calibration_samples)
