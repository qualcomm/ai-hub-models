# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
import shutil
import time
import zipfile
from collections.abc import Callable
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, overload

import numpy as np
import onnx
import qai_hub as hub
import torch
from qai_hub.client import Device
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models import (
    Precision,
    QAIRTVersion,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.export.result import (
    ComponentGroup,
    MultiGraphComponentGroup,
    MultiGraphGroup,
)
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    broadcast_data_to_multi_batch,
    get_batch_size,
    get_channel_last,
    make_torch_inputs,
)
from qai_hub_models.utils.onnx.helpers import (
    safe_torch_onnx_export,
)
from qai_hub_models.utils.onnx.torch_wrapper import extract_onnx_zip
from qai_hub_models.utils.transpose_channel import (
    transpose_channel_first_to_last,
)

_AIHUB_URL = "https://aihub.qualcomm.com"
_AIHUB_NAME = "Qualcomm® AI Hub"


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def get_hub_endpoint() -> str:
    # The deployment endpoint is the subdomain of AIHub that is used by the config.
    # e.g. for blah endpoint, returns https://blah.aihub.qualcomm.com
    return hub.hub._global_client.config.api_url


def export_torch_to_onnx_zip(
    torch_model: torch.nn.Module,
    f: str | PathLike,
    example_input: tuple[torch.Tensor, ...] | list[torch.Tensor],
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    onnx_transforms: Callable[[onnx.ModelProto], onnx.ModelProto] | None = None,
    skip_zip: bool = False,
    torch_export_kwargs: dict[str, Any] | None = None,
    prefer_external_weight: bool = True,
) -> str:
    """
    Export a torch model to ONNX, possibly as zip if model size exceeds 2GB,
    to conform to the input spec of AI Hub Workbench. Export as regular ONNX file if
    <2GB. If prefer_external_weight is True, always export with external
    weights regardless of size.

    Parameters
    ----------
    torch_model
        The torch.nn.Module to export.
    f
        The base filename. For models <2GB, this must end in ".onnx".
        For models >=2GB with skip_zip True, this must NOT end in
        ".onnx" (treated as a directory name). Otherwise, for models
        >=2GB with skip_zip False, the final output will be f+".zip".
    example_input
        A tuple of example input tensors for the export.
    input_names
        Optional list of input names.
    output_names
        Optional list of output names.
    onnx_transforms
        If defined, run this on the exported ONNX before packaging.
    skip_zip
        True to suppress zipping even for models >2GB. Default is False.
    torch_export_kwargs
        Additional keyword arguments to pass to torch.onnx.export.
    prefer_external_weight
        If True, export using external data format regardless of model size.
        Default is True.

    Returns
    -------
    exported_file_path : str
        The path to the exported file (either a .onnx file, a .onnx.zip file,
        or a directory if skip_zip is True and model size is >2GB).
    """
    f = Path(f)
    if isinstance(example_input, list):
        example_input = tuple(example_input)

    # Estimate total weight size (parameters and buffers) in bytes.
    # state_dict includes buffers etc, while model.parameters include only
    # learnable params.
    total_bytes = 0
    for tensor in torch_model.state_dict().values():
        # Some tensors may not have a defined element_size() (e.g.
        # non-numeric); skip them.
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total_bytes += tensor.numel() * tensor.element_size()
    threshold_bytes = 2 * 1024**3  # 2GB threshold

    torch_export_kwargs = torch_export_kwargs or {}

    if not f.name.endswith(".onnx"):
        f = f.with_suffix(".onnx")

    # Decide whether to export as single file or external data.
    use_external = prefer_external_weight or (total_bytes >= threshold_bytes)

    if not use_external:
        # For models under 2GB, export as a single ONNX file.
        start_time = time.time()
        safe_torch_onnx_export(
            torch_model,
            example_input,
            str(f),
            input_names=input_names,
            output_names=output_names,
            **torch_export_kwargs,
        )
        export_time = time.time() - start_time
        print(f"ONNX exported to {f} in {export_time:.1f} seconds")

        # Apply transforms if provided
        if onnx_transforms is not None:
            transform_start = time.time()
            print("Running onnx transforms on single file...")
            model_proto = onnx.load(str(f))
            model_proto = onnx_transforms(model_proto)
            onnx.save_model(model_proto, str(f))
            transform_time = time.time() - transform_start
            print(f"ONNX transform finished in {transform_time:.1f} seconds")

        return str(f)
    # Export with external data using two temporary directories.
    with qaihm_temp_dir() as tmpdir1, qaihm_temp_dir() as tmpdir2:
        tmpdir1_path = Path(tmpdir1)
        tmpdir2_path = Path(tmpdir2)
        export_path = (
            tmpdir1_path / f.with_suffix(".onnx").name
        )  # use .onnx extension for export

        start_time = time.time()
        safe_torch_onnx_export(
            torch_model,
            example_input,
            str(export_path),
            input_names=input_names,
            output_names=output_names,
            **torch_export_kwargs,
        )
        export_time = time.time() - start_time
        print(f"torch.onnx.export finished in {export_time:.1f} seconds")

        onnx_model = onnx.load(str(export_path))

        if onnx_transforms is not None:
            transform_start_time = time.time()
            print("Running onnx to onnx transforms...")
            onnx_model = onnx_transforms(onnx_model)
            transform_time = time.time() - transform_start_time
            print(f"ONNX transform finished in {transform_time:.1f} seconds")

        save_start_time = time.time()
        # .onnx and .data must have the same base name per hub requirement
        export_path2 = tmpdir2_path / "model.onnx"
        onnx.save_model(
            onnx_model,
            str(export_path2),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            # Weight file name must end with .data per Hub requirement
            location="model.data",
            convert_attribute=True,
        )
        save_time = time.time() - save_start_time
        print(f"onnx.save_model finished in {save_time:.1f} seconds")

        if skip_zip:
            # Instead of creating a zip, create a directory.
            out_dir = f  # f is expected to be a directory name already (without .onnx)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Copy the ONNX file to the directory.
            shutil.copy(export_path2, out_dir / "model.onnx")
            # Copy the external data file to the directory.
            external_data_path = export_path2.parent / "model.data"
            shutil.copy(external_data_path, out_dir / "model.data")

            print(f"ONNX with external data saved to directory {out_dir}")
            return str(out_dir)
        # Package the files into a zip.
        zip_start_time = time.time()
        zip_path = f.with_name(f.name + ".zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in tmpdir2_path.iterdir():
                # In the zip, files are placed under a folder named after the base name.
                arcname = f.with_suffix("").name + "/" + file_path.name
                zip_file.write(file_path, arcname=arcname)
        zip_time = time.time() - zip_start_time
        print(f"zipping onnx finished in {zip_time:.1f} seconds")
        print(f"ONNX with external data saved to {zip_path}")
        return str(zip_path)


class CompileOptions(NamedTuple):
    """A struct representing parsed compile options."""

    channel_last_input: list[str] | None = None
    channel_last_output: list[str] | None = None
    output_names: list[str] | None = None


def parse_compile_options(compile_job: hub.CompileJob) -> CompileOptions:
    model_options = compile_job.options.strip().split()
    channel_last_input: list[str] | None = None
    channel_last_output: list[str] | None = None
    output_names: list[str] | None = None
    for option_num in range(len(model_options)):
        if model_options[option_num] == "--force_channel_last_input":
            channel_last_input = model_options[option_num + 1].strip().split(",")
        if model_options[option_num] == "--force_channel_last_output":
            channel_last_output = model_options[option_num + 1].strip().split(",")
        if model_options[option_num] == "--output_names":
            output_names = model_options[option_num + 1].strip().split(",")
    return CompileOptions(channel_last_input, channel_last_output, output_names)


def make_hub_dataset_entries(
    tensors_tuple: tuple[
        torch.Tensor
        | np.ndarray
        | list[torch.Tensor | np.ndarray]
        | tuple[torch.Tensor | np.ndarray],
        ...,
    ],
    input_names: list[str],
    channel_last_input: list[str] | None = None,
) -> DatasetEntries:
    """
    Given input tensor(s) in either numpy or torch format,
    convert to hub DatasetEntries format.

    Parameters
    ----------
    tensors_tuple
        Tensor data in numpy or torch.Tensor format.
    input_names
        List of input names.
    channel_last_input
        Comma-separated list of input names to transpose channel.

    Returns
    -------
    dataset_entries : DatasetEntries
        Dataset entries in hub DatasetEntries format.
    """
    dataset = {}
    assert len(tensors_tuple) == len(input_names), (
        "Number of elements in tensors_tuple must match number of inputs"
    )
    for name, inputs in zip(input_names, tensors_tuple, strict=False):
        input_seq = inputs if isinstance(inputs, (list, tuple)) else [inputs]

        converted_inputs = []
        for curr_input in input_seq:
            if isinstance(curr_input, torch.Tensor):
                curr_input = tensor_to_numpy(curr_input)
            assert isinstance(curr_input, np.ndarray)
            if curr_input.dtype == np.int64:
                curr_input = curr_input.astype(np.int32)
            if curr_input.dtype == np.float64:
                curr_input = curr_input.astype(np.float32)
            converted_inputs.append(curr_input)
        dataset[name] = converted_inputs

    # Transpose dataset I/O if necessary to fit with the on-device model format
    if channel_last_input:
        dataset = transpose_channel_first_to_last(channel_last_input, dataset)
    return dataset


def ensure_hexagon_version(
    min_version: int, target_runtime: TargetRuntime, device: Device, model_name: str
) -> None | str:
    if not target_runtime.is_aot_compiled:
        return (
            f"Unsupported {target_runtime=}. {model_name} "
            "requires precompiled target runtime."
        )
    hex_attrs = [attr for attr in device.attributes if attr.startswith("hexagon:")]
    if len(hex_attrs) != 1:
        return f"Unable to determine hexagon version for {device.name}"
    hex_str = hex_attrs[0]
    # Extract hexagon version
    match = re.search(r"\d+", hex_str)
    hex_version = None
    if match:
        hex_version = int(match.group())
    else:
        return f"Unable to determine hexagon version for {device.name}"
    if hex_version < min_version:
        return f"{model_name} requires hexagon v{min_version} or above."
    return None


def download_model_in_memory(model: hub.Model) -> Any:
    """
    Download the model to a file and load it into memory.
    This replicates functionality that used to exist natively in the workbench client.
    """
    if model.model_type not in [
        hub.SourceModelType.TORCHSCRIPT,
        hub.SourceModelType.ONNX,
    ]:
        raise ValueError(
            "Downloading model in memory is currently only supported for torchscript and onnx."
        )
    with qaihm_temp_dir() as tmp_dir:
        model_file = model.download(os.path.join(tmp_dir, "tmp_model"))
        if model.model_type == hub.SourceModelType.TORCHSCRIPT:
            return torch.jit.load(model_file)
        if os.path.splitext(model_file)[1] == ".zip":
            onnx_path, _ = extract_onnx_zip(model_file)
            return onnx.load(onnx_path)
        return onnx.load(model_file)


def raise_if_fp_is_unsupported(device: hub.Device, precision: Precision) -> None:
    """
    Raise ValueError if the device does not support FP16 but the precision
    requires floating-point activations on the NPU.

    Parameters
    ----------
    device
        A fully-resolved ``hub.Device`` (as returned by
        :func:`qai_hub_models.utils.device.resolve_hub_device`), so
        ``device.attributes`` is guaranteed to be Hub's full attribute list.
    precision
        The precision requested for export.
    """
    if not precision.has_float_activations:
        return
    if "htp-supports-fp16:true" not in device.attributes:
        raise ValueError(
            f"The selected precision ({precision}) requires FP16 support, "
            "but the selected device does not support FP16 "
            "(missing htp-supports-fp16:true attribute). "
            "Please try a different precision or target device."
        )


def get_device_and_chipset_name(device: hub.Device) -> tuple[str | None, str | None]:
    """
    Given a hub Device, return the device name and chipset name.

    Parameters
    ----------
    device
        A hub Device object.

    Returns
    -------
    device_name : str | None
        Device name.
    chipset_name : str | None
        Chipset name.
    """
    chipset = None
    if device.attributes:
        if isinstance(device.attributes, list):
            for attr in device.attributes:
                if attr.startswith("chipset:"):
                    chipset = attr[len("chipset:") :]
                    break
        elif device.attributes.startswith("chipset:"):
            chipset = device.attributes[len("chipset:") :]
    return (device.name or None, chipset)


JobT = TypeVar("JobT", hub.CompileJob, hub.QuantizeJob, hub.LinkJob)


@overload
def assert_success_and_get_target_models(
    jobs: JobT,
) -> hub.Model: ...


@overload
def assert_success_and_get_target_models(
    jobs: MultiGraphGroup[JobT],
) -> MultiGraphGroup[hub.Model]: ...


@overload
def assert_success_and_get_target_models(
    jobs: ComponentGroup[JobT],
) -> ComponentGroup[hub.Model]: ...


@overload
def assert_success_and_get_target_models(
    jobs: MultiGraphComponentGroup[JobT],
) -> MultiGraphComponentGroup[hub.Model]: ...


def assert_success_and_get_target_models(  # type: ignore[misc]
    jobs: JobT
    | MultiGraphGroup[JobT]
    | ComponentGroup[JobT]
    | MultiGraphComponentGroup[JobT],
) -> (
    hub.Model
    | MultiGraphGroup[hub.Model]
    | ComponentGroup[hub.Model]
    | MultiGraphComponentGroup[hub.Model]
):
    """
    Assert all jobs succeeded and extract their target models.

    Parameters
    ----------
    jobs
        A single job, a ComponentGroup of jobs, a MultiGraphGroup,
        or a MultiGraphComponentGroup.

    Returns
    -------
    hub.Model | MultiGraphGroup[hub.Model] | ComponentGroup[hub.Model] | MultiGraphComponentGroup[hub.Model]
        The target model(s) extracted from the job(s), preserving the input structure.

    Raises
    ------
    AssertionError
        If any job failed and no target model is available.
    """
    if isinstance(jobs, MultiGraphComponentGroup):
        out_mgcg: MultiGraphComponentGroup[hub.Model] = MultiGraphComponentGroup()
        for (comp, gn), job in jobs.items():
            target_model = job.get_target_model()
            assert target_model is not None, f"Job failed for {comp}/{gn}: {job.url}"
            out_mgcg[(comp, gn)] = target_model
        return out_mgcg

    if isinstance(jobs, ComponentGroup):
        out_comp: ComponentGroup[hub.Model] = ComponentGroup()
        for name, job in jobs.items():
            target_model = job.get_target_model()
            assert target_model is not None, f"Job failed for {name}: {job.url}"
            out_comp[name] = target_model
        return out_comp

    if isinstance(jobs, MultiGraphGroup):
        graph_models: MultiGraphGroup[hub.Model] = MultiGraphGroup()
        for name, job in jobs.items():
            target_model = job.get_target_model()
            assert target_model is not None, f"Job failed for {name}: {job.url}"
            graph_models[name] = target_model
        return graph_models

    target_model = jobs.get_target_model()
    assert target_model is not None, f"Job failed: {jobs.url}"
    return target_model


def build_quantize_options(
    precision: Precision,
    litemp_percentage: float | None = None,
    other_quantize_options: str = "",
) -> str:
    all_options = other_quantize_options
    precision_options = precision.get_hub_quantize_options(litemp_percentage)
    if all_options and precision_options:
        all_options += " "
    all_options += precision_options
    return all_options


def build_compile_options(
    target_runtime: TargetRuntime,
    precision: Precision,
    input_spec: InputSpec,
    output_spec: OutputSpec,
    context_graph_name: str | None = None,
    other_compile_options: str = "",
) -> str:
    compile_options = ""
    if "--target_runtime" not in other_compile_options:
        flag = target_runtime.aihub_target_runtime_flag
        if flag is None:
            raise ValueError(
                f"Runtime {target_runtime.name} does not support AI Hub compilation."
            )
        compile_options = flag
    if (
        QAIRTVersion.HUB_FLAG not in other_compile_options
        and target_runtime.qairt_version_changes_compilation
    ):
        compile_options += f" {target_runtime.default_qairt_version.hub_option}"

    compile_options += f" --output_names {','.join(output_spec.keys())}"

    if channel_last_inputs := get_channel_last(input_spec, target_runtime):
        compile_options += (
            f" --force_channel_last_input {','.join(channel_last_inputs)}"
        )

    if channel_last_outputs := get_channel_last(output_spec, target_runtime):
        compile_options += (
            f" --force_channel_last_output {','.join(channel_last_outputs)}"
        )

    if precision.activations_type is not None:
        compile_options += " --quantize_io"
        if target_runtime == TargetRuntime.TFLITE:
            compile_options += " --quantize_io_type uint8"

    if target_runtime.is_aot_compiled:
        assert context_graph_name is not None, (
            f"Must specify a context_graph_name to compile for runtime {target_runtime.value}."
        )
        compile_options += f" --qnn_options context_enable_graphs={context_graph_name}"

    if other_compile_options:
        compile_options += " " + other_compile_options

    return compile_options


def build_link_options(
    target_runtime: TargetRuntime,
    other_link_options: str = "",
) -> str:
    if QAIRTVersion.HUB_FLAG not in other_link_options:
        other_link_options += f" {target_runtime.default_qairt_version.hub_option}"
    return other_link_options


def build_profile_options(
    target_runtime: TargetRuntime,
    context_graph_name: str | None = None,
    other_profile_options: str = "",
) -> str:
    if QAIRTVersion.HUB_FLAG not in other_profile_options:
        other_profile_options += f" {target_runtime.default_qairt_version.hub_option}"
    if context_graph_name is not None:
        if not target_runtime.is_aot_compiled:
            raise ValueError(
                "Cannot specify a context binary graph name if the target is not precompiled QAIRT."
            )
        other_profile_options += (
            f" --qnn_options context_enable_graphs={context_graph_name}"
        )
    return other_profile_options


def make_sample_inputs(input_spec: InputSpec) -> SampleInputsType:
    """
    Default implementation that returns a single random data array
    for each input name based on the shapes and dtypes in `get_input_spec`.

    A subclass may choose to override this and fetch a batch of real input data
    from a data source.
    """
    inputs_dict = {}
    inputs_list = make_torch_inputs(input_spec)
    for i, input_name in enumerate(input_spec.keys()):
        inputs_dict[input_name] = [inputs_list[i].numpy()]
    return inputs_dict


def expand_to_batch_size(
    sample_inputs: SampleInputsType,
    input_spec: InputSpec,
) -> SampleInputsType:
    """Expand single-batch sample inputs to match a multi-batch input spec."""
    batch_size = get_batch_size(input_spec)
    if batch_size is not None and batch_size > 1:
        return broadcast_data_to_multi_batch(input_spec, sample_inputs)
    return sample_inputs
