# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""End-to-end evaluate pipeline for a single (non-collection) model."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, cast

import qai_hub as hub

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.args import get_model_kwargs
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.evaluate.helpers import (
    EvaluateIntermediates,
    EvaluateResult,
    _load_quant_cpu_onnx,
    evaluate_on_dataset,
)
from qai_hub_models.utils.export.context import (
    resolve_manifest,
    resolve_model_cls,
)
from qai_hub_models.utils.export.pipeline import export_model as export_model_fn
from qai_hub_models.utils.export.result import ExportIntermediates
from qai_hub_models.utils.inference import AsyncOnDeviceModel, compile_model_from_args
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.kwarg_helpers import filter_kwargs
from qai_hub_models.utils.precompiled_qnn_onnx import (
    PrecompiledQnnOnnxModel,
    find_component_onnx,
)


def evaluate_model(
    source_dir: Path,
    device: hub.Device,
    target_runtime: TargetRuntime = TargetRuntime.TFLITE,
    precision: Precision = Precision.float,
    compile_options: str = "",
    inference_options: str = "",
    quantize_options: str = "",
    hub_model_id: str | None = None,
    dataset_cls: type[BaseDataset] | None = None,
    samples_per_job: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
    use_dataset_cache: bool = False,
    force_rebuild_cache: bool = False,
    skip_torch_accuracy: bool = False,
    skip_device_accuracy: bool = False,
    compute_quant_cpu_accuracy: bool = False,
    compute_psnr: bool = False,
    num_calibration_samples: int | None = None,
    local_model_dir: str | None = None,
    evaluate_intermediates: EvaluateIntermediates | None = None,
    skip_waiting: bool = False,
    **kwargs: Any,
) -> EvaluateResult:
    """
    Evaluate a single-graph model's accuracy on a dataset.

    ``supports_quant_cpu`` (i.e. whether the QDQ ONNX CPU accuracy check
    should be run for non-float precisions) is derived from the manifest —
    a model can use it iff it supports quantization and can go through the
    Hub quantize job.

    Parameters
    ----------
    source_dir
        On-disk path to the recipe folder (contains ``manifest.yaml`` and
        ``model.py``).
    device
        Device object (e.g. ``hub.Device("Samsung Galaxy S25 (Family)")``).
    target_runtime
        Target runtime (e.g. ``TargetRuntime.TFLITE``).
    precision
        Precision (e.g. ``Precision.float``).
    compile_options
        Additional options to pass when submitting the compile job.
    inference_options
        Additional options to pass when submitting the inference job.
    quantize_options
        Additional options to pass when submitting the quantize job.
    hub_model_id
        A compiled hub model id.
    dataset_cls
        Dataset class to use for evaluation.
    samples_per_job
        Max size to be submitted in a single inference job.
    num_samples
        Number of samples to run. If set to -1, will run on full dataset.
    seed
        Random seed to use when shuffling the data. If not set, samples data deterministically.
    use_dataset_cache
        If set, will store hub dataset ids in a local file and re-use for subsequent evaluations on the same dataset.
    force_rebuild_cache
        If set with use_dataset_cache, discards the local cache for this dataset and builds it again.
    skip_torch_accuracy
        If flag is set, skips computing accuracy with the torch model.
    skip_device_accuracy
        If flag is set, skips computing accuracy on device.
    compute_quant_cpu_accuracy
        If flag is set, computes the accuracy of the quantized onnx model on the CPU.
    compute_psnr
        If flag is set, also compares on-device outputs against torch on the
        model's sample inputs and reports psnr. Models with no eval dataset
        do this unconditionally, since it is the only numeric check available.
    num_calibration_samples
        The number of calibration data samples to use for quantization.
    local_model_dir
        Directory holding the model's precompiled_qnn_onnx wrapper ``.onnx`` and
        its ``.bin``. If set, the model runs on this machine as the "on-device"
        executor and nothing is compiled or submitted to Workbench.
    evaluate_intermediates
        If its ``inference_jobs`` list is non-empty, resume from these
        already-submitted inference jobs instead of resubmitting.
    skip_waiting
        If set, submits the on-device inference jobs and returns before
        waiting on them. ``device_accuracy`` on the result will be ``None``;
        ``inference_jobs`` carries the submitted jobs to resume from later.
    **kwargs
        Additional kwargs for model instantiation (from_pretrained) and input_spec.

    Returns
    -------
    result : EvaluateResult
        Torch, quant-cpu and on-device accuracy, plus the on-device
        inference jobs submitted (populated even when ``skip_waiting``).
    """
    warnings.filterwarnings("ignore")

    if local_model_dir is not None and compute_quant_cpu_accuracy:
        raise ValueError(
            "--local-model-dir cannot be combined with --compute-quant-cpu-accuracy: "
            "the quantized CPU path needs a Workbench-compiled model."
        )

    model_cls = cast(type[BaseModel], resolve_model_cls(source_dir))
    manifest = resolve_manifest(source_dir)
    supports_quant_cpu = (
        manifest.can_use_quantize_job and manifest.supports_quantization
    )
    model_kwargs = get_model_kwargs(model_cls, kwargs)
    input_spec_kwargs = filter_kwargs(model_cls.get_input_spec, kwargs)

    eval_dataset_classes = model_cls.get_eval_dataset_classes()
    has_eval_dataset = len(eval_dataset_classes) > 0
    if not has_eval_dataset:
        if local_model_dir is not None:
            raise ValueError(
                "--local-model-dir requires a model with an eval dataset; "
                "PSNR needs a Workbench-compiled model."
            )
        print(
            "Model does not have evaluation dataset specified. Evaluating PSNR on a single sample."
        )
        compute_psnr = True
        skip_device_accuracy = True
        skip_torch_accuracy = True

    input_spec: InputSpec | None = None
    torch_model = model_cls.from_pretrained(**model_kwargs)
    model_executors: dict[str, ExecutableModelProtocol] = {}

    if not skip_torch_accuracy:
        model_executors["torch"] = torch_model
        input_spec = torch_model.get_input_spec(**input_spec_kwargs)

    if local_model_dir is not None:
        local_input_spec = torch_model.get_input_spec(**input_spec_kwargs)
        if not skip_device_accuracy:
            model_executors["on-device"] = PrecompiledQnnOnnxModel(
                find_component_onnx(local_model_dir),
                input_names=list(local_input_spec),
            )
        if input_spec is None:
            input_spec = local_input_spec
    # On-device, quant-cpu, or psnr path -- all three need the compiled asset.
    elif (
        not skip_device_accuracy
        or compute_psnr
        or (supports_quant_cpu and compute_quant_cpu_accuracy)
    ):
        if hub_model_id is not None:
            compiled_model: hub.Model = hub.get_model(hub_model_id)
        else:
            # Reconstruct a Namespace for compile_model_from_args
            args_for_compile = argparse.Namespace(
                device=device,
                chipset=None,
                target_runtime=target_runtime,
                precision=precision,
                compile_options=compile_options,
                inference_options=inference_options,
                quantize_options=quantize_options,
                num_calibration_samples=num_calibration_samples,
            )
            compiled_result = compile_model_from_args(
                source_dir, args_for_compile, {**model_kwargs, **input_spec_kwargs}
            )
            assert isinstance(compiled_result, hub.Model)
            compiled_model = compiled_result

        if compiled_model.get_producer() is None:
            raise ValueError(
                "Compiled models must be compiled with AI Hub Workbench; they cannot be uploaded manually."
            )

        on_device_model = AsyncOnDeviceModel(
            model=compiled_model,
            input_names=list(input_spec) if input_spec else None,
            device=device,
            inference_options=torch_model.get_hub_inference_options(
                target_runtime, inference_options
            ),
        )
        if not skip_device_accuracy:
            model_executors["on-device"] = on_device_model

        if (
            supports_quant_cpu
            and compute_quant_cpu_accuracy
            and precision != Precision.float
        ):
            model_executors["quant cpu"] = _load_quant_cpu_onnx(compiled_model)

        input_spec = on_device_model.get_input_spec()

    if input_spec is None:
        raise ValueError("Cannot extract input spec.")

    if not has_eval_dataset and compute_psnr:
        export_result = export_model_fn(
            source_dir,
            device,
            precision=precision,
            target_runtime=target_runtime,
            inference_options=inference_options,
            skip_compiling=True,
            skip_linking=True,
            skip_downloading=True,
            export_intermediates=ExportIntermediates(compiled_model=compiled_model),
            **model_kwargs,
            **input_spec_kwargs,
        )
        return EvaluateResult(psnr_metrics=export_result.inference_metrics)

    if dataset_cls is None:
        raise ValueError("dataset_cls is required for evaluation on a dataset.")

    return evaluate_on_dataset(
        evaluator_func=torch_model.get_evaluator,
        dataset_cls=dataset_cls,
        model_executors=model_executors,
        input_spec=input_spec,
        samples_per_job=samples_per_job,
        num_samples=num_samples,
        seed=seed,
        use_cache=use_dataset_cache,
        evaluate_intermediates=evaluate_intermediates,
        skip_waiting=skip_waiting,
        compute_psnr=compute_psnr,
        force_rebuild_cache=force_rebuild_cache,
    )
