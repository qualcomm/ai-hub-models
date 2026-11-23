# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
End-to-end export pipeline for a single (non-collection, non-precompiled) model.

A reader can scan the seven numbered comments in :func:`export_model` to see
the full recipe; each step's implementation lives in its own sibling module
(``upload.py``, ``quantize.py``, ``compile.py``, etc.).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import qai_hub as hub
from qai_hub_models_cli.errors import CLIRuntimeError

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.utils.ai_hub_access import can_access_qualcomm_ai_hub
from qai_hub_models.utils.args import get_export_model_name, get_model_kwargs
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.compile import run_compile
from qai_hub_models.utils.export.context import (
    resolve_manifest,
    resolve_model_cls,
    resolve_model_display_name,
    resolve_model_id,
)
from qai_hub_models.utils.export.download import download_model_bundle
from qai_hub_models.utils.export.inference import run_inference
from qai_hub_models.utils.export.link import run_link
from qai_hub_models.utils.export.quantize import run_quantize
from qai_hub_models.utils.export.result import (
    ExportIntermediates,
    ExportResult,
    validate_export_intermediates,
)
from qai_hub_models.utils.export.summary import (
    compute_inference_summary,
    extract_tool_versions,
    print_profile_summary,
)
from qai_hub_models.utils.export.upload import upload_source
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
    precision: Precision = Precision.float,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    num_calibration_samples: int | None = None,
    export_intermediates: ExportIntermediates | None = None,
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
) -> ExportResult:
    """
    Run the seven-step export recipe end-to-end:

        1. Instantiate the PyTorch model and upload its serialized form to AI Hub
        2. Quantize the model (skipped for float)
        3. Compile the (possibly quantized) source model to the target runtime
        4. Link the compiled DLC to a context binary (AOT runtimes only)
        5. Run on-device inference on sample inputs, profiling at the same time
        6. Extract the SDK / runtime versions used by the jobs
        7. Download the compiled asset locally

    Parameters
    ----------
    source_dir
        On-disk path to the recipe folder (contains ``manifest.yaml`` and
        ``model.py``). The recipe's ``model_id`` is derived from
        ``source_dir.name``.
    device
        Hub device to export for (e.g. ``hub.Device("Samsung Galaxy S25")``).
    precision
        Target precision for the export.
    target_runtime
        On-device runtime to target. Defaults to TFLite.
    num_calibration_samples
        Number of calibration data samples to use for quantization. If None,
        uses the default specified by the dataset.
    export_intermediates
        Already-built assets to start from, in place of rebuilding them. If
        ``compiled_model``/``compile_job`` is set, upload/quantize/compile are
        all skipped; if only ``quantized_model``/``quantize_job`` is set,
        upload/quantize are skipped; if ``source_model`` is set, it's the
        already-uploaded serialized model: only the upload is skipped, and
        quantize (when the precision needs it) and compile still run on it. A
        job field takes precedence over its sibling model field when both are
        set.
    skip_compiling
        If set, skips compiling the model to an on-device format.
    skip_linking
        If set, stops after compiling -- skips linking (AOT runtimes only)
        and everything after it.
    skip_inferencing
        If set, skips the inference job entirely -- no outputs and no profile
        metrics.
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
    link_options
        Extra options to pass when submitting the link job. Ignored for runtimes that do not link.
    inference_options
        Extra options to pass when submitting the inference job.
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
    model_name = get_export_model_name(
        model_cls, model_id, precision, additional_model_kwargs
    )
    output_path = Path(output_dir or Path.cwd() / "export_assets")

    if not manifest.can_use_quantize_job and precision != Precision.float:
        assert precision in manifest.supported_precisions, (
            f"Precision {precision!s} is not supported by {model_name}"
        )

    _, chipset = get_device_and_chipset_name(device)
    export_intermediates = export_intermediates or ExportIntermediates()
    validate_export_intermediates(export_intermediates, target_runtime.uses_hub_link)

    # 1. Instantiate the PyTorch model. Upload it only if no compiled or
    # quantized asset was supplied; the upload is the most expensive setup step.
    model = model_cls.from_pretrained(
        **get_model_kwargs(
            model_cls, dict(**additional_model_kwargs, precision=precision)
        )
    )
    input_spec = model.get_input_spec(
        **filter_kwargs(model.get_input_spec, additional_model_kwargs)
    )
    compile_job: hub.client.CompileJob | None = export_intermediates.compile_job
    quantize_job: hub.client.QuantizeJob | None = export_intermediates.quantize_job
    source_model: hub.Model | None = export_intermediates.source_model
    if compile_job is None and export_intermediates.compiled_model is None:
        if quantize_job is not None:
            source_model = get_target_models(quantize_job)
        elif export_intermediates.quantized_model is not None:
            source_model = export_intermediates.quantized_model
        else:
            if source_model is None:
                source_model = upload_source(model, input_spec)

            # 2. Quantize (float and aimet-quantized models skip this entirely)
            if precision != Precision.float and manifest.can_use_quantize_job:
                onnx_compile_job = run_compile(
                    model,
                    model_name,
                    device,
                    TargetRuntime.ONNX,
                    Precision.float,
                    source_model,
                    input_spec,
                )
                quantize_job = run_quantize(
                    precision,
                    model,
                    model_name,
                    get_target_models(onnx_compile_job),
                    num_calibration_samples,
                    extra_options=quantize_options,
                    input_spec=input_spec,
                    calibration_data=export_intermediates.calibration_data,
                )
                if skip_compiling:
                    return ExportResult(quantize_job=quantize_job)
                source_model = get_target_models(quantize_job)

    # 3. Compile.
    if compile_job is None and export_intermediates.compiled_model is None:
        assert source_model is not None
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
    if skip_linking:
        return ExportResult(quantize_job=quantize_job, compile_job=compile_job)

    # 4. Link (AOT runtimes only).
    needs_target_model = (
        target_runtime.uses_hub_link or not skip_inferencing or not skip_downloading
    )
    link_job: hub.client.LinkJob | None = export_intermediates.link_job
    target_model: hub.Model | None = None
    if needs_target_model:
        if link_job is not None:
            target_model = get_target_models(link_job)
        elif target_runtime.uses_hub_link:
            if export_intermediates.compiled_model is not None:
                compiled_model = export_intermediates.compiled_model
            else:
                assert compile_job is not None
                compiled_model = get_target_models(compile_job)
            link_job = run_link(
                compiled_model,
                device,
                model_name,
                model,
                target_runtime,
                extra_options=link_options,
            )
            target_model = get_target_models(link_job)
        elif export_intermediates.compiled_model is not None:
            target_model = export_intermediates.compiled_model
        else:
            assert compile_job is not None
            target_model = get_target_models(compile_job)

    inference_opts = model.get_hub_inference_options(target_runtime, inference_options)

    # 5. Run on-device inference, which also collects profile metrics.
    inference_job: hub.client.InferenceJob | None = export_intermediates.inference_job
    if inference_job is None and not skip_inferencing:
        assert target_model is not None
        inference_job = run_inference(
            model.sample_inputs(
                input_spec=input_spec,
                use_channel_last_format=target_runtime.channel_last_native_execution,
            ),
            model_name,
            device,
            inference_opts,
            target_model,
        )

    # 6. Tool versions (needed for both download metadata and the summary).
    if not skip_summary or not skip_downloading:
        tool_versions, tool_versions_from_inference = extract_tool_versions(
            inference_job,
            compile_job,
        )
    else:
        tool_versions, tool_versions_from_inference = None, False

    # 7. Download.
    download_path: Path | None = None
    metadata: ModelMetadata | None = None
    if not skip_downloading and tool_versions is not None:
        assert target_model is not None
        download_path, metadata = download_model_bundle(
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

    inference_metrics: pd.DataFrame | None = None
    if not skip_summary:
        if inference_job is not None:
            print_profile_summary(inference_job)
            inference_metrics = compute_inference_summary(
                model,
                inference_job,
                input_spec,
                target_runtime,
                outputs_to_skip=list(
                    (manifest.outputs_to_skip_validation or {}).keys()
                ),
                metrics=manifest.inference_metrics,
                print_result=True,
            )
        print_tool_versions(tool_versions, tool_versions_from_inference)
        demo_job = link_job or compile_job
        if manifest.has_on_target_demo and demo_job is not None:
            print_on_target_demo_cmd(demo_job, source_dir, device)
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
        inference_job=inference_job,
        download_path=download_path,
        tool_versions=tool_versions,
        metadata=metadata,
        inference_metrics=inference_metrics,
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
