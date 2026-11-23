# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
from collections.abc import Mapping, ValuesView
from pathlib import Path
from typing import Any, cast

import numpy as np
import qai_hub as hub
import torch
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models import TargetRuntime
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.asset_loaders import ModelZooAssetConfig, VersionType
from qai_hub_models.utils.export.result import CollectionExportResult
from qai_hub_models.utils.input_spec import (
    InputSpec,
    workbench_to_qihm_input_spec,
)
from qai_hub_models.utils.qai_hub_helpers import (
    _AIHUB_NAME,
    make_hub_dataset_entries,
    parse_compile_options,
)
from qai_hub_models.utils.transpose_channel import (
    transpose_channel_last_to_first,
    transpose_channel_last_to_first_input_specs,
)

try:
    from qai_hub_models.utils.quantization_aimet_onnx import AIMETOnnxQuantizableMixin
except NotImplementedError:
    AIMETOnnxQuantizableMixin = None  # type: ignore[assignment,misc]


def compile_model_from_args(
    source_dir: Path,
    cli_args: argparse.Namespace,
    model_kwargs: Mapping[str, Any],
    component: str | None = None,
) -> hub.Model | list[hub.Model]:
    """
    Parameters
    ----------
    source_dir
        On-disk path to the recipe folder (contains ``manifest.yaml`` and
        ``model.py``).
    cli_args
        CLI arguments. We will use cli_args.chipset, .device,
        .target_runtime.
    model_kwargs
        kwargs pertaining to model's .from_pretrained.
    component
        For qai_hub_models.utils.base_model.CollectionModel, set
        component to compile a specific components. None (default) to compile all
        components.

    Returns
    -------
    hub.Model | list[hub.Model]
        The compiled hub.Model object(s).
    """
    model_kwargs_dict = dict(model_kwargs)
    cli_str = ""
    if "precision" in cli_args:
        model_kwargs_dict["precision"] = cli_args.precision
        cli_str += f"--precision {cli_args.precision} "

    from qai_hub_models.utils.export.dispatch import select_pipeline

    model_id = source_dir.name
    export_model = select_pipeline(source_dir)
    if getattr(cli_args, "num_calibration_samples", None):
        model_kwargs_dict["num_calibration_samples"] = cli_args.num_calibration_samples
        cli_str += f"--num-calibration-samples {cli_args.num_calibration_samples} "
    if getattr(cli_args, "compile_options", ""):
        model_kwargs_dict["compile_options"] = cli_args.compile_options
        cli_str += f'--compile-options="{cli_args.compile_options}" '
    if getattr(cli_args, "quantize_options", ""):
        model_kwargs_dict["quantize_options"] = cli_args.quantize_options
        cli_str += f'--quantize-options="{cli_args.quantize_options}" '
    if cli_args.device is not None:
        if cli_args.chipset:
            cli_str += f"--chipset {cli_args.chipset} "
        if cli_args.device.name:
            cli_str += f'--device "{cli_args.device.name}" '
    model_name = model_id + (f".{component}" if component else "")
    print(f"Compiling on-device model asset for {model_name}.")
    print(
        f"Running qai-hub-models export {model_id} {cli_str} --target-runtime {cli_args.target_runtime.name.lower()}\n"
    )
    component_kwargs = {}
    if component is not None:
        component_kwargs = {"components": [component]}
    export_output = export_model(
        device=cli_args.device,
        skip_inferencing=True,
        skip_downloading=True,
        skip_summary=True,
        target_runtime=cli_args.target_runtime,
        **model_kwargs_dict,
        **component_kwargs,
    )

    no_hub_access = isinstance(export_output, list)

    if no_hub_access:
        # The export returned local file paths, which mean Hub credentials were not found.
        raise NotImplementedError(
            f"Please sign-up for {_AIHUB_NAME} to continue the demo with on-device inference."
        )
    if isinstance(export_output, CollectionExportResult):
        assert export_output.compile_jobs is not None
        # For AOT runtimes the runnable artifact is the linked context binary,
        # not the unlinked compiled DLC. Fall back to compile jobs for JIT.
        jobs = export_output.link_jobs or export_output.compile_jobs
        if component is not None:
            target_model = jobs[component].get_target_model()
            assert target_model is not None
            return target_model
        target_model_list: list[hub.Model] = []
        for job in jobs.values():
            model = job.get_target_model()
            assert model is not None
            target_model_list.append(model)
        return target_model_list
    job = export_output.link_job or export_output.compile_job
    target_model = job.get_target_model()
    assert target_model is not None
    return target_model


def dataset_entries_from_batch(
    batch: Any,
    input_names: list[str],
    channel_last_input: list[str] | None,
) -> tuple[DatasetEntries, DatasetEntries]:
    """
    Given a batch from a torch dataloader with the standard (inputs, gt) format,

    Inputs will be converted to a format compatible with inference of a compiled
    model on AI Hub Workbench. Inputs will be broken into N tensors with a batch size of 1, since
    compiled models require a batch size of 1.

    GT data will be left as-is (1 tensor with a batch dim of size N), as it's not passed to the model.

    Parameters
    ----------
    batch
        A batch from a torch dataloader in (inputs, gt) format.
    input_names
        Names of the input tensors for the model.
    channel_last_input
        List of input names that should be converted to channel-last format.

    Returns
    -------
    input_entries : DatasetEntries
        Dataset entries for inputs that can be uploaded to hub.
    gt_entries : DatasetEntries
        Dataset entries for ground truth that can be uploaded to hub.
    """
    model_inputs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    model_inputs, ground_truth_values, *_ = batch
    if not isinstance(model_inputs, (list, tuple)):
        # Generalize: treat "single-input" model as a list of 1 input.
        # This allows us to support single and multi-input models in 1 loop.
        model_inputs = [model_inputs]
    model_inputs_split_by_batch = tuple(
        cast(
            np.ndarray | tuple[np.ndarray],
            np.array_split(x.numpy(), x.shape[0], axis=0),
        )
        for x in model_inputs
    )

    if isinstance(ground_truth_values, (list, tuple)):
        output_names = [f"output_{i}" for i in range(len(ground_truth_values))]
        ground_truth_values = tuple(ground_truth_values)
    else:
        output_names = ["output_0"]
        ground_truth_values = (ground_truth_values,)
    input_entries = make_hub_dataset_entries(
        model_inputs_split_by_batch,
        input_names,
        channel_last_input,
    )
    gt_entries = make_hub_dataset_entries(ground_truth_values, output_names, None)
    return input_entries, gt_entries


class WorkbenchJobFailed(ValueError):
    """A Workbench job ended unsuccessfully, which is a result rather than a bug."""


def _failed_producers(job: hub.Job) -> list[hub.Job]:
    """The failed jobs that produced the models `job` consumes."""
    inputs = getattr(job, "models", None) or [getattr(job, "model", None)]
    producers = (m.get_producer() for m in inputs if m is not None)
    return [
        p
        for p in producers
        if p is not None and p.get_status().state == hub.JobStatus.State.FAILED
    ]


def root_cause_job(job: hub.Job) -> hub.Job:
    """Follow "producer failed" failures back to the job that actually failed."""
    while "producer failed" in (job.get_status().message or ""):
        if not (producers := _failed_producers(job)):
            break
        job = producers[0]
    return job


def job_failed_error(job: hub.Job) -> WorkbenchJobFailed:
    """The error for a failed job, carrying its type, id, Hub failure message and URL."""
    job = root_cause_job(job)
    message = job.get_status().message or "(no job failure message)"
    return WorkbenchJobFailed(
        f"{job.job_type.capitalize()} Job {job.job_id} failed: {message} ({job.url})"
    )


class AsyncOnDeviceResult:
    def __init__(
        self,
        inference_job: hub.InferenceJob,
        target_runtime: TargetRuntime,
        channel_last_output: list[str],
        output_names: list[str],
        num_retries: int = 1,
    ) -> None:
        self.inference_job = inference_job
        self.target_runtime = target_runtime
        self.channel_last_output = channel_last_output
        self.output_names = output_names
        self.num_retries = num_retries

    def wait(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        retries = 0
        while not self.inference_job.wait().success and retries < self.num_retries:
            retries += 1
            msg = self.inference_job.get_status().message
            print(
                f"Retrying ({retries}/{self.num_retries}) inference job "
                f"({self.inference_job.job_id}) ({msg})"
            )
            ijob = hub.submit_inference_job(
                model=self.inference_job.model,
                inputs=self.inference_job.inputs,
                device=self.inference_job.device,
            )
            assert isinstance(ijob, hub.InferenceJob)
            self.inference_job = ijob

        if not self.inference_job.wait().success:
            raise job_failed_error(self.inference_job)

        output_ds_handle = self.inference_job.get_output_dataset()
        assert output_ds_handle is not None
        output_dataset = output_ds_handle.download()
        assert not isinstance(output_dataset, str)

        if self.channel_last_output:
            output_dataset = transpose_channel_last_to_first(
                self.channel_last_output,
                output_dataset,
            )

        outputs: ValuesView[list[np.ndarray]] | list[list[np.ndarray]] = (
            output_dataset.values()
        )
        if len(self.output_names) > 0:
            outputs = [output_dataset[out_name] for out_name in self.output_names]

        output_torch = [
            torch.from_numpy(np.concatenate(output, axis=0)) for output in outputs
        ]

        if len(output_torch) == 1:
            return output_torch[0]
        return tuple(output_torch)


class AsyncOnDeviceModel:
    """
    Class that behaves like a pytorch model except when called, it runs an
    inference job on hub and returns an AsyncOnDeviceResult. Calling
    AsyncOnDeviceResult.wait() will return a torch result in the same format
    as the PyTorch model.
    """

    def __init__(
        self,
        model: hub.Model,
        input_names: list[str] | None,
        device: hub.Device,
        inference_options: str = "",
        output_names: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            Model stored in hub that will be used to run inference.
        input_names
            List of input names to the model.
        device
            Device on which to execute inference.
        inference_options
            Options to pass to the inference job.
        output_names
            List of output names from the model.
        """
        self.model = model
        self.input_names = input_names or []
        self.device = device
        self.inference_options = inference_options

        # Determine whether I/O is channel last
        self.channel_last_input, self.channel_last_output = [], []
        compile_output_names = None
        producer = self.model.get_producer()
        if isinstance(producer, (hub.CompileJob, hub.QuantizeJob)):
            self.input_names = self.input_names or list(producer.shapes)
        elif isinstance(producer, hub.LinkJob) and isinstance(
            producer.models[0].get_producer(), hub.CompileJob
        ):
            self.input_names = self.input_names or list(
                cast(hub.CompileJob, producer.models[0].get_producer()).shapes
            )
        if isinstance(producer, (hub.CompileJob, hub.LinkJob)):
            if isinstance(producer, hub.LinkJob) and isinstance(
                producer.models[0].get_producer(), hub.CompileJob
            ):
                compile_options = parse_compile_options(
                    cast(hub.CompileJob, producer.models[0].get_producer())
                )
            elif isinstance(producer, hub.CompileJob):
                compile_options = parse_compile_options(producer)
            self.channel_last_input = compile_options.channel_last_input or []
            self.channel_last_output = compile_options.channel_last_output or []
            compile_output_names = compile_options.output_names or []
        self.output_names = output_names or compile_output_names or []

    @property
    def target_runtime(self) -> TargetRuntime:
        return TargetRuntime.from_hub_model_type(self.model.model_type)

    def __call__(
        self,
        *args: torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | hub.Dataset
        | DatasetEntries,
    ) -> AsyncOnDeviceResult:
        assert len(args) > 0, "At least 1 input should be provided for inference."

        dataset: hub.Dataset | DatasetEntries
        if isinstance(args[0], (hub.Dataset, Mapping)):
            # Use the existing provided dataset
            assert len(args) == 1, "Only 1 dataset can be provided for inference."
            dataset = args[0]
        else:
            tensors = tuple(args)
            dataset_entries = make_hub_dataset_entries(
                tensors,  # type: ignore[arg-type]
                self.input_names,
                self.channel_last_input,
            )
            dataset = hub.upload_dataset(dataset_entries)

        inference_job = hub.submit_inference_job(
            model=self.model,
            inputs=dataset,
            device=self.device,
            name=f"{self.model.name}_demo_inference",
            options=self.inference_options,
        )
        assert isinstance(inference_job, hub.InferenceJob)
        return AsyncOnDeviceResult(
            inference_job,
            self.target_runtime,
            self.channel_last_output,
            self.output_names,
        )

    def get_input_spec(self) -> InputSpec:
        """
        Get the input specs for this model.

        OnDeviceModel will adapt channel-first pyTorch input to channel last. Thus
        channel-last shapes in AI Hub Workbench model will always be returned in channel-first format.
        """
        producer = self.model.get_producer()
        if producer is None:
            raise ValueError(
                "Unable to extract input shape from a model that was not compiled with AI Hub Workbench."
            )
        if producer._job_type == hub.JobType.QUANTIZE:
            return workbench_to_qihm_input_spec(cast(hub.QuantizeJob, producer).shapes)
        if producer._job_type == hub.JobType.COMPILE:
            target_shapes = cast(hub.CompileJob, producer).get_target_shapes()
            assert target_shapes is not None
            out = workbench_to_qihm_input_spec(
                {name: target_shapes[name] for name in self.input_names}
            )
            return transpose_channel_last_to_first_input_specs(
                out, self.channel_last_input
            )
        if producer._job_type == hub.JobType.LINK:
            target_shapes = cast(
                hub.CompileJob,
                cast(hub.LinkJob, producer).models[0].get_producer(),
            ).get_target_shapes()
            assert target_shapes is not None
            out = workbench_to_qihm_input_spec(
                {name: target_shapes[name] for name in self.input_names}
            )
            return transpose_channel_last_to_first_input_specs(
                out, self.channel_last_input
            )
        raise NotImplementedError(
            f"Can't extract shapes from producer job of type {producer.job_type}"
        )


class OnDeviceModel(ExecutableModelProtocol):
    """
    Class that behaves like a pytorch model except when called, it runs an
    inference job on hub and returns a torch output.

    Intended to be passed as in input to app.py to run an app on-device.
    """

    def __init__(
        self,
        model: hub.Model,
        input_names: list[str],
        device: hub.Device,
        inference_options: str = "",
        output_names: list[str] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model
            Model stored in hub that will be used to run inference.
        input_names
            List of input names to the model.
        device
            Device on which to execute inference.
        inference_options
            Options to pass to the inference job.
        output_names
            List of output names from the model.
        """
        self.async_model = AsyncOnDeviceModel(
            model,
            input_names,
            device,
            inference_options,
            output_names,
        )
        self.model = model

    @property
    def input_names(self) -> list[str]:
        return self.async_model.input_names

    @property
    def channel_last_input(self) -> list[str]:
        return self.async_model.channel_last_input

    def __call__(
        self,
        *args: torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | hub.Dataset
        | DatasetEntries,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        return self.forward(*args)

    def forward(
        self,
        *args: torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | hub.Dataset
        | DatasetEntries,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        async_result = self.async_model(*args)
        return async_result.wait()

    def get_input_spec(self) -> InputSpec:
        """
        Get the input specs for this model.

        OnDeviceModel will adapt channel-first pyTorch input to channel last. Thus
        channel-last shapes in AI Hub Workbench model will always be returned in channel-first format.
        """
        return self.async_model.get_input_spec()


def get_uploaded_precompiled_model(
    model_path: str,
    model_name: str,
    model_version: VersionType,
    model_component: str,
    ignore_cached_model: bool = False,
) -> hub.Model:
    """Caches pre-compiled model in default asset path to be used in sub-sequence demos."""
    asset_config = ModelZooAssetConfig.from_cfg()
    model_id_path = asset_config.get_local_store_model_path(
        model_name, model_version, f"{model_component}_model_id.cached"
    )

    uploaded_model = None
    if not ignore_cached_model:
        try:
            with open(model_id_path) as model_id_file:
                model_id = model_id_file.readline().strip()
            print(f"Using previously uploaded model({model_id}) for {model_component}")
            uploaded_model = hub.get_model(model_id=model_id)
            if uploaded_model is not None:
                return uploaded_model

        except Exception:
            pass

    # Upload model on hub
    uploaded_model = hub.upload_model(model_path)
    with open(model_id_path, "w") as model_id_file:
        model_id_file.writelines([f"{uploaded_model.model_id}"])
    return uploaded_model
