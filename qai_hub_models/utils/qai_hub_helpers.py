# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
import shlex
import shutil
import time
import zipfile
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any, NamedTuple, overload

import numpy as np
import onnx
import qai_hub as hub
import torch
from qai_hub.client import DatasetEntries, Device

from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.utils.asset_loaders import qaihm_temp_dir
from qai_hub_models.utils.onnx.helpers import (
    safe_torch_onnx_export,
)
from qai_hub_models.utils.onnx.torch_wrapper import extract_onnx_zip
from qai_hub_models.utils.transpose_channel import (
    transpose_channel_first_to_last,
)

_AIHUB_URL = "https://aihub.qualcomm.com"
_AIHUB_NAME = "Qualcomm® AI Hub"
_CAN_ACCESS_HUB: bool | None = None


def can_access_qualcomm_ai_hub() -> bool:
    global _CAN_ACCESS_HUB  # noqa: PLW0603
    if _CAN_ACCESS_HUB is not None:
        return _CAN_ACCESS_HUB
    try:
        hub.get_frameworks()
    except Exception:
        _CAN_ACCESS_HUB = False
    else:
        _CAN_ACCESS_HUB = True
    return _CAN_ACCESS_HUB


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


def extract_job_options(job: hub.Job) -> dict[str, str | bool]:
    """
    Get a dictionary of all options passed for this job.

    Options that are not passed explicitly in the workbench job options (that Hub will treat as defaults) are not included in the dictionary.

    Parameters
    ----------
    job
        The job from which to extract options.

    Returns
    -------
    options : dict[str, str | bool]
        Dictionary of all options passed for this job.

    Examples
    --------
    If a hub job is submitted with options "--qairt_version 2.33 --dequantize_outputs --dict_input='w=x;y=z'"
    Then the returned dict would be:

    .. code-block:: python

        {
            "qairt_version": "2.33",
            "dequantize_outputs": True,
            "dict_input": "w=x;y=z"
        }
    """
    out = {}

    model_options = shlex.split(job.options.strip())
    for i in range(len(model_options)):
        option = model_options[i]
        if option.startswith("--"):
            value: str | bool
            if "=" in option:
                # Handle args of form "--blah=blah"
                #
                # If the option starts with '--' and has an =, then it must be in the format --x=y.
                # If the option was in the format --x y=x, then it would be parsed as two different options by shlex.
                # So we can safely split this option on the first '=' in the string to get the option key and value.
                key, value = option.split("=", maxsplit=1)
                if (value.startswith("'") and value.endswith("'")) or (
                    value.startswith('"') and value.endswith('"')
                ):
                    value = value[1 : len(value) - 1]
            elif i == len(model_options) - 1 or model_options[i + 1].startswith("--"):
                # Either:
                #   - this is the last arg and has no value
                #   - this --arg is immediately followed by another --arg
                # Therefore it must be a boolean. All Hub booleans are true if explicitly passed as an option.
                key = option
                value = True
            else:
                # This is a standard --key value pair.
                key = option
                value = model_options[i + 1]
            # Strip "--" from arg name.
            out[key[2:]] = value

    return out


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


@overload
def assert_success_and_get_target_models(
    jobs: Mapping[str, hub.CompileJob | hub.QuantizeJob],
) -> dict[str, hub.Model]: ...


@overload
def assert_success_and_get_target_models(
    jobs: Mapping[str | None, hub.CompileJob | hub.QuantizeJob],
) -> dict[str | None, hub.Model]: ...


def assert_success_and_get_target_models(
    jobs: Mapping[str | None, hub.CompileJob | hub.QuantizeJob]
    | Mapping[str, hub.CompileJob | hub.QuantizeJob],
) -> dict[str | None, hub.Model] | dict[str, hub.Model]:
    """
    Assert all jobs succeeded and extract their target models.

    Parameters
    ----------
    jobs
        A dict mapping names to compile or quantize jobs.

    Returns
    -------
    target_models : dict[str | None, hub.Model] | dict[str, hub.Model]
        A dict mapping names to target models.

    Raises
    ------
    AssertionError
        If any job failed and no target model is available.
    """
    target_models = {}
    for name, job in jobs.items():
        target_model = job.get_target_model()
        assert target_model is not None, f"Job failed for {name}: {job.url}"
        target_models[name] = target_model
    return target_models
