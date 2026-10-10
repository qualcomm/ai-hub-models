# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import importlib
import importlib.metadata
import os
import shutil
import struct
import tempfile
import warnings
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import qai_hub as hub
import torch
from onnx.helper import (
    get_all_tensor_dtypes,
    tensor_dtype_to_np_dtype,
    tensor_dtype_to_string,
)
from packaging.version import parse as parse_version
from typing_extensions import Self

from qai_hub_models.utils.runtime_torch_wrapper import (
    ModelIODetails,
    kwargs_to_dict,
)


@dataclass
class ONNXBundle:
    """
    Represents an ONNX "bundle" folder containing an ONNX graph file
    and associated supporting files (like encodings and external weights).
    """

    # The path to the ONNX parent folder (that contains model.onnx, encodings, etc.)
    bundle_path: Path
    # The name of the .onnx graph file in the bundle folder.
    onnx_graph_name: str
    # The name of the external weights file in the bundle folder.
    # None if this bundle does not include external weights.
    onnx_weights_name: str | None = None
    # The name of the external QAIRT context binary file in the bundle folder.
    # None if this bundle does not include a QAIRT context binary.
    qairt_bin_name: str | None = None
    # The name of the .encodings file in the bundle folder.
    # None if this bundle does not include encodings.
    aimet_encodings_name: str | None = None
    # If True, bundle_path is a temporary directory that will be deleted
    # when this ONNXBundle is garbage-collected or used as a context manager.
    ephemeral: bool = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        if self.ephemeral and self.bundle_path.is_dir():
            shutil.rmtree(self.bundle_path, ignore_errors=True)

    def __del__(self) -> None:
        self.cleanup()

    @property
    def onnx_graph_path(self) -> Path:
        return self.bundle_path / self.onnx_graph_name

    @property
    def onnx_weights_path(self) -> Path | None:
        if self.onnx_weights_name is None:
            return None
        return self.bundle_path / self.onnx_weights_name

    @property
    def qairt_context_binary_path(self) -> Path | None:
        if self.qairt_bin_name is None:
            return None
        return self.bundle_path / self.qairt_bin_name

    @property
    def aimet_encodings_path(self) -> Path | None:
        if self.aimet_encodings_name is None:
            return None
        return self.bundle_path / self.aimet_encodings_name

    def load_onnx_model(self) -> onnx.ModelProto:
        """Load and return the ONNX ModelProto from this bundle."""
        return onnx.load(str(self.onnx_graph_path), load_external_data=True)

    @staticmethod
    def from_bundle_path(
        bundle_path: str | os.PathLike,
        model_name: str | None = None,
        ephemeral: bool = False,
    ) -> ONNXBundle:
        """
        Creates an ONNX bundle object from the given path.

        Parameters
        ----------
        bundle_path
            The folder in which the ONNX files exst.
        model_name
            The name of each bundle file: {model_name}.onnx, optional {model_name}.data, optional {model_name}.encodings
            If None, this method will glob the folder to find the file names and raise if more than 1 file is found.
        ephemeral
            If True, the bundle directory is deleted when this object is garbage-collected.

        Returns
        -------
        onnx_bundle : ONNXBundle
            ONNX Bundle object.
        """
        model_name_grep = model_name or "*"
        onnx_folder_path = Path(bundle_path)

        weights_files = list(onnx_folder_path.glob(f"{model_name_grep}.data"))
        if len(weights_files) > 1:
            raise ValueError(
                f"Found more than 1 ONNX weight file in {bundle_path}: {' '.join(x.name for x in weights_files)} "
            )

        encodings_files = list(onnx_folder_path.glob(f"{model_name_grep}.encodings"))
        if len(encodings_files) > 1:
            raise ValueError(
                f"Found more than 1 AIMET encodings file in {bundle_path}: {' '.join(x.name for x in encodings_files)} "
            )

        qairt_bin_files = list(onnx_folder_path.glob(f"{model_name_grep}.bin"))
        if len(qairt_bin_files) > 1:
            raise ValueError(
                f"Found more than 1 QAIRT context binary file in {bundle_path}: {' '.join(x.name for x in qairt_bin_files)} "
            )

        return ONNXBundle(
            bundle_path=onnx_folder_path,
            onnx_graph_name=next(onnx_folder_path.glob(f"{model_name_grep}.onnx")).name,
            onnx_weights_name=weights_files[0].name if weights_files else None,
            aimet_encodings_name=encodings_files[0].name if encodings_files else None,
            qairt_bin_name=qairt_bin_files[0].name if qairt_bin_files else None,
            ephemeral=ephemeral,
        )

    def move(
        self, dst_folder: str | os.PathLike, dst_model_name: str, copy: bool = False
    ) -> ONNXBundle:
        """
        Moves the ONNX file, external weights, and AIMET encodings in this ONNX bundle to a new location.

        Parameters
        ----------
        dst_folder
            Base folder path in which to store the moved ONNX files.
        dst_model_name
            The name of each moved file: {dst_model_name}.onnx, {dst_model_name}.data, {dst_model_name}.encodings
        copy
            If False, bundle files are moved to the destination path. `self` will modified in-place to reflect the new bundle location.
            If True, bundle files are copied to the destination path. `self` will be un-changed.

        Returns
        -------
        moved_bundle : ONNXBundle
            `Self` is modified in-place and returned if `copy` is False (default behavior). If `copy` is True, returns a new bundle and does not modify `Self`.
        """
        dst = ONNXBundle(
            Path(dst_folder),
            f"{dst_model_name}.onnx",
            f"{dst_model_name}.data" if self.onnx_weights_name else None,
            f"{dst_model_name}.bin" if self.qairt_bin_name else None,
            f"{dst_model_name}.encodings" if self.aimet_encodings_name else None,
        )

        os.makedirs(dst_folder, exist_ok=True)
        copy_or_move = shutil.copyfile if copy else shutil.move
        copy_or_move(self.onnx_graph_path, dst.onnx_graph_path)

        if self.onnx_weights_path and dst.onnx_weights_path and dst.onnx_weights_name:
            copy_or_move(self.onnx_weights_path, dst.onnx_weights_path)

            # Only change the ONNX model references to external weights if the external weights path changed.
            if self.onnx_weights_name != dst.onnx_weights_name:
                onnx_model = onnx.load(dst.onnx_graph_path, load_external_data=False)
                for initializer in onnx_model.graph.initializer:
                    if initializer.data_location == onnx.TensorProto.EXTERNAL:
                        for entry in initializer.external_data:
                            if entry.key == "location":
                                entry.value = dst.onnx_weights_name
                onnx.save(onnx_model, dst.onnx_graph_path)

        if (
            self.qairt_context_binary_path
            and dst.qairt_context_binary_path
            and self.qairt_bin_name
            and dst.qairt_bin_name
        ):
            copy_or_move(self.qairt_context_binary_path, dst.qairt_context_binary_path)

            # Only change the ONNX model references to external context binary if the external context binary path changed.
            if self.qairt_bin_name != dst.qairt_bin_name:

                def modify_qairt_bin_name(
                    model: onnx.ModelProto, curr_bin_name: str, dst_bin_name: str
                ) -> None:
                    # Assumes only one EPContext node and one context binary in the graph, so we can return after modifying the first match.
                    for node in model.graph.node:
                        if node.op_type == "EPContext":
                            for attribute in node.attribute:
                                if (
                                    attribute.name == "ep_cache_context"
                                    and os.path.normpath(attribute.s.decode())
                                    == curr_bin_name
                                ):
                                    attribute.s = (
                                        "./" + os.path.normpath(dst_bin_name)
                                    ).encode()  # onnx attributes are bytes, so encode the string to bytes
                                    return

                onnx_model = onnx.load(dst.onnx_graph_path, load_external_data=False)
                modify_qairt_bin_name(
                    onnx_model, self.qairt_bin_name, dst.qairt_bin_name
                )
                onnx.save(onnx_model, dst.onnx_graph_path)

        if self.aimet_encodings_path and dst.aimet_encodings_path:
            copy_or_move(self.aimet_encodings_path, dst.aimet_encodings_path)

        if copy:
            return dst

        self.bundle_path = dst.bundle_path
        self.onnx_graph_name = dst.onnx_graph_name
        self.onnx_weights_name = dst.onnx_weights_name
        self.qairt_bin_name = dst.qairt_bin_name
        self.aimet_encodings_name = dst.aimet_encodings_name
        return self


def download_and_unzip_workbench_onnx_model(
    model: hub.Model, dst_folder: str | os.PathLike, model_name: str | None = None
) -> ONNXBundle:
    """
    Downloads, unzips, and renames an AI Hub Workbench ONNX model.

    Parameters
    ----------
    model
        The AI Hub Workbench model object.
    dst_folder
        Path to the folder in which the downloaded bundle files should live.
    model_name
        The name to use for each downloaded bundle file: {model_name}.onnx, optional {model_name}.data, optional {model_name}.encodings
        If None, uses the name defined by the AI Hub workbench model object.

    Returns
    -------
    onnx_bundle : ONNXBundle
        ONNX Bundle object created from the downloaded workbench model.
    """
    if model.model_type not in [
        hub.SourceModelType.ONNX,
        hub.SourceModelType.AIMET_ONNX,
    ]:
        raise ValueError(
            f"An ONNX bundle must be created from an ONNX model. The AI Hub Workbench Model is a {model.model_type.name} model."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = model.download(tmpdir)
        if model_path.endswith(".zip"):
            shutil.unpack_archive(model_path, tmpdir)
            os.remove(path=model_path)
            archive_contents = os.listdir(tmpdir)
            if len(archive_contents) == 0:
                raise ValueError("Empty model archive.")
            if len(archive_contents) == 1 and os.path.isdir(
                os.path.join(tmpdir, archive_contents[0])
            ):
                bundle_folder = os.path.join(tmpdir, archive_contents[0])
            else:
                bundle_folder = tmpdir
        else:
            bundle_folder = tmpdir

        bundle = ONNXBundle.from_bundle_path(bundle_folder)
        dst_model_name = model_name or model.name
        use_subfolder = (
            bundle.onnx_weights_name
            or bundle.aimet_encodings_name
            or bundle.qairt_bin_name
        )
        bundle.move(
            (str(Path(dst_folder) / dst_model_name) + "_onnx")
            if use_subfolder
            else dst_folder,
            dst_model_name,
        )

    return bundle


def get_onnx_model_bundle_metadata(
    bundle: ONNXBundle,
    model_bundle_path: str | os.PathLike,
    model_name: str,
) -> tuple[str, dict[str, str]]:
    """
    Reads the given onnx bundle and returns metadata.

    Parameters
    ----------
    bundle
        The ONNX bundle to read (created by download_and_unzip_workbench_onnx_model)
    model_bundle_path
        The path to the root of the model bundle being downloaded (this is the dst_path param passed to download_and_unzip_workbench_onnx_model)
    model_name
        The target model name (this is the model_name param passed to download_and_unzip_workbench_onnx_model)

    Returns
    -------
    relative_onnx_path : str
        Path to the onnx file relative to model_bundle_path.
    supplementary_files : dict[str, str]
        map<path to supplementary file relative to model_bundle_path, description of supplementary file>
    """
    model_file_name = str(bundle.onnx_graph_path.relative_to(model_bundle_path))

    supplementary_files: dict[str, str] = {}
    if bundle.onnx_weights_path:
        supplementary_files[
            str(bundle.onnx_weights_path.relative_to(model_bundle_path))
        ] = f"Model weights loaded by {model_name}.onnx"
    if bundle.qairt_context_binary_path:
        supplementary_files[
            str(bundle.qairt_context_binary_path.relative_to(model_bundle_path))
        ] = f"QAIRT context loaded by {model_name}.onnx"
    if bundle.aimet_encodings_path:
        supplementary_files[
            str(bundle.aimet_encodings_path.relative_to(model_bundle_path))
        ] = f"AIMET encodings file to combine with {model_name}.onnx"

    return model_file_name, supplementary_files


# Maps type strings returned by onnxruntime.InferenceSession.get_inputs() to numpy types.
ORT_TENSOR_STR_TO_NP_TYPE = {
    f"tensor({tensor_dtype_to_string(dtype)[len('TensorProto.') :].lower()})": tensor_dtype_to_np_dtype(
        dtype
    )
    for dtype in get_all_tensor_dtypes()
}

QUANTIZED_IO_TYPES = [np.uint8, np.uint16, np.int8, np.int16]

# torch <-> numpy dtype maps used by the CUDA io-binding inference path.
TORCH_TO_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}
NP_TO_TORCH_DTYPE = {
    np.dtype(np_type): torch_type for torch_type, np_type in TORCH_TO_NP_DTYPE.items()
}


@wraps(torch.onnx.export)
def safe_torch_onnx_export(*args: Any, **kwargs: Any) -> None:
    """
    Calls torch.onnx.export.

    1. Makes sure ONNX installed is compatible with AI Hub Workbench.
    2. Makes sure dynamo export is not used by default.
    3. Catches large model export failures caused by a bug in Torch 2.5.
    """
    try:
        if "dynamo" not in kwargs:
            kwargs = {**kwargs, "dynamo": False}
        verify_onnx_export_is_compatible_with_ai_hub()
        torch.onnx.export(*args, **kwargs)
    except RuntimeError as e:
        if torch.__version__.startswith(
            "2.5."
        ) and "The serialized model is larger than the 2GiB" in str(e):
            raise ValueError(
                "Large model export to ONNX is broken in torch 2.5. Install a different torch version and try again."
            ) from None
        raise


def _resolve_output_shapes(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
) -> list[tuple[int, ...]] | None:
    """Resolve concrete output shapes from session metadata and input shapes.

    The io-binding path needs to pre-allocate output buffers, which requires
    fully concrete shapes. ONNX graphs frequently leave output dims symbolic
    (e.g. dynamic sequence/context length). We resolve them from (a) symbolic
    dim names shared with the inputs, and (b) a few well-known LLM conventions
    (growing past_* KV caches, batch-sized logits). Returns ``None`` if any
    dimension can't be resolved, signalling the caller to fall back to numpy.
    """
    input_shapes = {name: tuple(t.shape) for name, t in inputs.items()}
    sym_table: dict[str, int] = {}
    for meta in session.get_inputs():
        # Unfed, or fed at a lower rank than the graph declares: the symbolic
        # dims are unresolvable, so bail instead of indexing out of range.
        fed_shape = input_shapes.get(meta.name)
        if fed_shape is None or len(fed_shape) != len(meta.shape):
            return None
        for i, dim in enumerate(meta.shape):
            if isinstance(dim, str):
                sym_table[dim] = fed_shape[i]

    if "input_ids" in input_shapes:
        batch_size: int | None = input_shapes["input_ids"][0]
        seq_len: int | None = input_shapes["input_ids"][1]
    elif "inputs_embeds" in input_shapes:
        batch_size = input_shapes["inputs_embeds"][0]
        seq_len = input_shapes["inputs_embeds"][1]
    elif "attention_mask" in input_shapes and len(input_shapes["attention_mask"]) == 4:
        # Split parts after the first take a hidden-state input (not input_ids /
        # inputs_embeds), so derive the new-token count from the 4D causal mask
        # (batch, 1, seq_len, context_length). Without this these parts can't
        # resolve their growing past_*_out shapes and fall back to the slow
        # numpy (host round-trip) path instead of staying on-device.
        batch_size = input_shapes["attention_mask"][0]
        seq_len = input_shapes["attention_mask"][2]
    else:
        batch_size = None
        seq_len = None

    output_shapes: list[tuple[int, ...]] = []
    for meta in session.get_outputs():
        shape: list[int] = []
        for dim_idx, dim in enumerate(meta.shape):
            if isinstance(dim, int):
                shape.append(dim)
            elif isinstance(dim, str) and dim in sym_table:
                shape.append(sym_table[dim])
            elif "past_" in meta.name and "_out" in meta.name and dim_idx == 2:
                in_name = meta.name.replace("_out", "_in").replace("_updated", "")
                if in_name in input_shapes and seq_len is not None:
                    shape.append(input_shapes[in_name][2] + seq_len)
                else:
                    return None
            elif "logits" in meta.name and dim_idx == 0 and batch_size is not None:
                shape.append(batch_size)
            else:
                return None
        output_shapes.append(tuple(shape))
    return output_shapes


def _numpy_inference(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
    target_device: torch.device | str,
) -> torch.Tensor | Collection[torch.Tensor]:
    """Run inference by moving inputs/outputs through numpy on the host."""
    np_inputs = {k: v.cpu().detach().numpy() for k, v in inputs.items()}
    output_np = session.run(None, np_inputs)
    output_tensors = [torch.from_numpy(out).to(target_device) for out in output_np]

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def _iobinding_inference(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor | Collection[torch.Tensor]:
    """Run inference on a CUDA session via io-binding, keeping data on-device.

    Avoids the GPU->host->GPU numpy round-trip by binding torch CUDA tensors
    directly to ONNX Runtime. Falls back to :func:`_numpy_inference` when the
    output shapes can't be resolved or an I/O dtype isn't representable in torch.
    """
    cuda_options = session.get_provider_options().get("CUDAExecutionProvider", {})
    device_id = int(cuda_options.get("device_id", "0"))
    torch_device = torch.device(f"cuda:{device_id}")

    output_shapes = _resolve_output_shapes(session, inputs)
    if output_shapes is None:
        return _numpy_inference(session, inputs, torch_device)

    binding = session.io_binding()

    # Keep python references to bound tensors alive for the duration of the run.
    bound_inputs: list[torch.Tensor] = []
    for name, tensor in inputs.items():
        if tensor.dtype not in TORCH_TO_NP_DTYPE:
            return _numpy_inference(session, inputs, torch_device)
        tensor = tensor.to(torch_device).contiguous()
        bound_inputs.append(tensor)
        binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=TORCH_TO_NP_DTYPE[tensor.dtype],
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )

    output_tensors: list[torch.Tensor] = []
    for meta, shape in zip(session.get_outputs(), output_shapes, strict=False):
        np_dtype = ORT_TENSOR_STR_TO_NP_TYPE.get(meta.type)
        if np_dtype is None or np.dtype(np_dtype) not in NP_TO_TORCH_DTYPE:
            return _numpy_inference(session, inputs, torch_device)
        tensor = torch.empty(
            shape, dtype=NP_TO_TORCH_DTYPE[np.dtype(np_dtype)], device=torch_device
        )
        output_tensors.append(tensor)
        binding.bind_output(
            name=meta.name,
            device_type="cuda",
            device_id=device_id,
            element_type=np_dtype,
            shape=shape,
            buffer_ptr=tensor.data_ptr(),
        )

    torch.cuda.synchronize()
    session.run_with_iobinding(binding)
    torch.cuda.synchronize()

    if len(output_tensors) == 1:
        return output_tensors[0]
    return output_tensors


def mock_torch_onnx_inference(
    session: onnxruntime.InferenceSession,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | Collection[torch.Tensor]:
    """Run an ONNX InferenceSession with a torch-like calling convention.

    On a CUDA session this uses io-binding to keep tensors on-device (with a
    numpy fallback when output shapes/dtypes can't be resolved); on a CPU
    session it runs through numpy. The interface is unchanged: pass torch
    tensors positionally or by name, get torch tensor(s) back.
    """
    input_names = [inp.name for inp in session.get_inputs()]
    inputs = kwargs_to_dict(input_names, *args, **kwargs)

    if "CUDAExecutionProvider" in session.get_providers():
        return _iobinding_inference(session, inputs)

    return _numpy_inference(session, inputs, "cpu")


# Initializer proto definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L499
def _extract_scale(initializer: onnx.TensorProto) -> float:
    assert initializer.data_type == onnx.TensorProto.DataType.Value("FLOAT")
    if len(initializer.float_data) == 1:
        return initializer.float_data[0]
    assert len(initializer.raw_data) == 4, "Expected four bytes of raw float data."
    return struct.unpack("<f", initializer.raw_data)[0]


def _extract_zero_point(initializer: onnx.TensorProto) -> int:
    valid_data_types: dict[str, tuple[str, int]] = {
        "UINT8": ("<B", 1),
        "INT8": ("<b", 1),
        "UINT16": ("<H", 2),
        "INT16": ("<h", 2),
        "INT32": ("<i", 4),
    }
    for dtype, (sformat, size) in valid_data_types.items():
        if initializer.data_type == onnx.TensorProto.DataType.Value(dtype):
            if len(initializer.int32_data) == 1:
                return initializer.int32_data[0]
            assert len(initializer.raw_data) == size, (
                f"Expect raw data to have {size} byte(s)."
            )
            return struct.unpack(sformat, initializer.raw_data)[0]
    raise ValueError(
        f"Quantization zero point constant has unknown data type {initializer.data_type}.",
    )


def _extract_qdq_scale_zp(
    onnx_model: onnx.GraphProto,
    initializer_indices: dict[str, int],
    qdq_node: onnx.NodeProto,
) -> ModelIODetails.QDQParams:
    scale = _extract_scale(
        onnx_model.initializer[initializer_indices[qdq_node.input[1]]]
    )
    optional_zero_point_index = 2
    zero_point = (
        _extract_zero_point(
            onnx_model.initializer[
                initializer_indices[qdq_node.input[optional_zero_point_index]]
            ]
        )
        if optional_zero_point_index < len(qdq_node.input)
        else 0
    )
    return ModelIODetails.QDQParams(scale, zero_point)


def extract_io_types_from_onnx_model(
    onnx_model: onnx.ModelProto | onnxruntime.InferenceSession,
) -> tuple[
    dict[str, ModelIODetails],
    dict[str, ModelIODetails],
]:
    """
    Extract I/O details from an ONNX model.

    Parameters
    ----------
    onnx_model
        ONNX model protobuf, or ONNX inference session.

    Returns
    -------
    model_input_details : dict[str, ModelIODetails]
        Model Input Details
    model_output_details : dict[str, ModelIODetails]
        Model Output Details
    """
    inputs: dict[str, ModelIODetails]
    outputs: dict[str, ModelIODetails]
    if isinstance(onnx_model, onnxruntime.InferenceSession):
        # extract from inference session
        input_names = {i.name for i in onnx_model.get_inputs()}
        output_names = {output.name for output in onnx_model.get_outputs()}

        inputs = {
            i.name: ModelIODetails(
                tuple(i.shape), ORT_TENSOR_STR_TO_NP_TYPE[i.type], None
            )
            for i in onnx_model.get_inputs()
        }
        outputs = {
            o.name: ModelIODetails(
                tuple(o.shape), ORT_TENSOR_STR_TO_NP_TYPE[o.type], None
            )
            for o in onnx_model.get_outputs()
        }
    else:
        # extract from onnx GraphProto
        input_names = {i.name for i in onnx_model.graph.input}
        output_names = {output.name for output in onnx_model.graph.output}
        initializer_indices = {
            init.name: idx for idx, init in enumerate(onnx_model.graph.initializer)
        }

        inputs = {
            i.name: ModelIODetails(
                tuple(x.dim_value for x in i.type.tensor_type.shape.dim),
                tensor_dtype_to_np_dtype(i.type.tensor_type.elem_type),
                None,
            )
            for i in onnx_model.graph.input
        }
        outputs = {
            o.name: ModelIODetails(
                tuple(x.dim_value for x in o.type.tensor_type.shape.dim),
                tensor_dtype_to_np_dtype(o.type.tensor_type.elem_type),
                None,
            )
            for o in onnx_model.graph.output
        }

        # Extract I/O QDQ Params
        for node in onnx_model.graph.node:
            if node.op_type == "EPContext":
                for input_name in node.input:
                    if input_name in input_names:
                        dtype = inputs[input_name].dtype
                        if dtype in QUANTIZED_IO_TYPES:
                            warnings.warn(
                                f"Network input {input_name} is an input to an EPContext node, and is {dtype} quantized."
                                " Cannot determine the QDQ parameters for the input.",
                                stacklevel=2,
                            )

                for output_name in node.output:
                    if output_name in output_names:
                        dtype = outputs[output_name].dtype
                        if dtype in QUANTIZED_IO_TYPES:
                            warnings.warn(
                                f"Network output {output_name} is an output of an EPContext node, and is {dtype} quantized."
                                " Cannot determine the QDQ parameters for the output.",
                                stacklevel=2,
                            )

            if node.op_type == "DequantizeLinear":
                if node.input[0] in input_names:
                    inputs[node.input[0]].qdq_params = _extract_qdq_scale_zp(
                        onnx_model.graph, initializer_indices, node
                    )
            elif node.op_type == "QuantizeLinear" and node.output[0] in output_names:
                outputs[node.output[0]].qdq_params = _extract_qdq_scale_zp(
                    onnx_model.graph, initializer_indices, node
                )

    return inputs, outputs


def onnx_model_is_precompiled_qairt(onnx_model: onnx.ModelProto) -> bool:
    """
    Determines if a model is pre-compiled to run on HTP via QAIRT.

    Parameters
    ----------
    onnx_model
        ONNX Model proto

    Returns
    -------
    is_precompiled : bool
        True if a model is pre-compiled.

    Notes
    -----
    A model is pre-compiled if it looks like this:
    Input -> Optional QDQ nodes -> EP Context Node -> Optional QDQ nodes -> Output
    Therefore it can have a maximum of 2 nodes (Q + DQ) per input and output, and 1 EP node.
    """
    # Only check the max number number of nodes allowed in a precompiled QDQ model.
    max_num_nodes = (len(onnx_model.graph.input) + len(onnx_model.graph.output)) * 2 + 1
    return len(onnx_model.graph.node) <= max_num_nodes and any(
        x.op_type == "EPContext" for x in onnx_model.graph.node
    )


ONNX_ENV_CHECKED: bool = False
ONNX_ENV_ERROR: str | None = None
ONNX_PACKAGE_NAME = "onnx"
ONNX_MAX_COMPATIBLE_VERSION = "1.18.0"
ONNX_MIN_INCOMPATIBLE_VERSION = "1.19.0"


def verify_onnx_export_is_compatible_with_ai_hub(
    pkg_versions: dict[str, str] | None = None,
) -> None:
    """
    Verifies the ONNX version installed on this machine can be used to export
    model files that are compatible with AI Hub Workbench.

    Runs only once then caches the result for this python session.

    Parameters
    ----------
    pkg_versions
        Installed pip package versions. If none, extracts packages from current environment.

    Raises
    ------
    ValueError
        If onnx:
        * is not installed
        * is too new (produces an IR version that AI Hub Workbench cannot handle)
    """
    global ONNX_ENV_CHECKED  # noqa: PLW0603
    global ONNX_ENV_ERROR  # noqa: PLW0603
    if not ONNX_ENV_CHECKED:
        if pkg_versions is None:
            pkgs = importlib.metadata.distributions()
            pkg_versions = {p.name: p.version for p in pkgs}

        if ONNX_PACKAGE_NAME not in pkg_versions:
            ONNX_ENV_ERROR = (
                "Package 'onnx' is not installed in your python environment."
            )
        elif parse_version(pkg_versions[ONNX_PACKAGE_NAME]) >= parse_version(
            ONNX_MIN_INCOMPATIBLE_VERSION
        ):
            ONNX_ENV_ERROR = f"Installed onnx package (onnx=={pkg_versions[ONNX_PACKAGE_NAME]}) is too new for compatibility with AI Hub Workbench."

        if ONNX_ENV_ERROR is not None:
            ONNX_ENV_ERROR = f"{ONNX_ENV_ERROR} Install {ONNX_MAX_COMPATIBLE_VERSION} or earlier:  pip install onnx=={ONNX_MAX_COMPATIBLE_VERSION}"
        ONNX_ENV_CHECKED = True

    if ONNX_ENV_CHECKED and ONNX_ENV_ERROR:
        raise ValueError(ONNX_ENV_ERROR)


def _add_io_helper(
    io_name: str,
    spec: tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails,
    is_input: bool,
) -> tuple[
    list[onnx.NodeProto],
    list[onnx.TensorProto],
    list[onnx.ValueInfoProto],
    list[onnx.ValueInfoProto],
    list[onnx.ValueInfoProto],
]:
    graph_nodes = []
    initializers = []
    input_nodes = []
    output_nodes = []
    value_info = []
    if isinstance(spec, ModelIODetails):
        onnx_dtype = onnx.helper.np_dtype_to_tensor_dtype(spec.dtype)
        if spec.qdq_params:
            # We have quantization parameters, add Q-DQ
            output_q = onnx.helper.make_tensor_value_info(
                f"{io_name}_q",
                onnx_dtype,
                spec.shape,
            )
            output_dq = onnx.helper.make_tensor_value_info(
                f"{io_name}_dq",
                onnx.TensorProto.FLOAT,
                spec.shape,
            )
            value_info += [output_q, output_dq]

            input_name = output_q.name if not is_input else io_name
            output_name = output_q.name if is_input else io_name
            onnx_input = onnx.helper.make_tensor_value_info(
                input_name,
                onnx_dtype,
                spec.shape,
            )
            onnx_output = onnx.helper.make_tensor_value_info(
                output_name,
                onnx_dtype,
                spec.shape,
            )
            value_info += [onnx_input, onnx_output]

            q_scale = onnx.helper.make_tensor(
                f"{io_name}_scale",
                onnx.TensorProto.FLOAT,
                [],
                [spec.qdq_params.scale],
            )
            q_zp = onnx.helper.make_tensor(
                f"{io_name}_zp", onnx_dtype, [], [spec.qdq_params.zero_point]
            )
            initializers += [q_scale, q_zp]

            q_op = onnx.helper.make_node(
                "DequantizeLinear",
                name=onnx_input.name,
                inputs=[onnx_input.name, q_scale.name, q_zp.name],
                outputs=[output_dq.name],
            )
            dq_op = onnx.helper.make_node(
                "QuantizeLinear",
                name=onnx_output.name,
                inputs=[output_dq.name, q_scale.name, q_zp.name],
                outputs=[onnx_output.name],
            )
            graph_nodes += [q_op, dq_op]
            input_nodes.append(onnx_input)
            output_nodes.append(onnx_output)
        else:
            onnx_input = onnx.helper.make_tensor_value_info(
                io_name, onnx_dtype, spec.shape
            )
            input_nodes.append(onnx_input)
            output_nodes.append(onnx_input)
            value_info.append(onnx_input)
    else:
        shape, onnx_dtype = spec
        onnx_input = onnx.helper.make_tensor_value_info(io_name, onnx_dtype, shape)
        input_nodes.append(onnx_input)
        output_nodes.append(onnx_input)
        value_info.append(onnx_input)
    return graph_nodes, initializers, value_info, input_nodes, output_nodes


def generate_wrapper_onnx_file(
    graph_name: str,
    onnx_output_path: str | Path,
    onnx_input_specs: Mapping[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ],
    onnx_output_specs: Mapping[
        str, tuple[tuple[int, ...], onnx.TensorProto.DataType] | ModelIODetails
    ],
    qnn_context_bin_path: str | Path,
    qairt_version: str,
) -> None:
    ep_cache_context_content = str(qnn_context_bin_path)
    ctx_embed_mode = 0

    graph_nodes = []
    initializers = []
    model_inputs = []
    ep_context_inputs = []
    value_info = []
    for key, spec in onnx_input_specs.items():
        graph_nodes_new, initializers_new, value_info_new, input_nodes, output_nodes = (
            _add_io_helper(key, spec, True)
        )
        initializers += initializers_new
        graph_nodes += graph_nodes_new
        value_info += value_info_new
        model_inputs += input_nodes
        ep_context_inputs += output_nodes

    model_outputs = []
    ep_context_outputs = []
    for key, spec in onnx_output_specs.items():
        graph_nodes_new, initializers_new, value_info_new, input_nodes, output_nodes = (
            _add_io_helper(key, spec, False)
        )
        initializers += initializers_new
        value_info += value_info_new
        graph_nodes += graph_nodes_new
        model_outputs += output_nodes
        ep_context_outputs += input_nodes

    qnn_ep_context_node = onnx.helper.make_node(
        "EPContext",
        name=graph_name,
        inputs=[node.name for node in ep_context_inputs],
        outputs=[node.name for node in ep_context_outputs],
        ep_cache_context=ep_cache_context_content,
        embed_mode=ctx_embed_mode,
        ep_sdk_version=qairt_version,
        source="Qnn",
        domain="com.microsoft",
    )
    graph_nodes.append(qnn_ep_context_node)

    graph_def = onnx.helper.make_graph(
        graph_nodes,
        "qnn-onnx-model",
        model_inputs,
        model_outputs,
        initializers,
        "",
        value_info,
    )
    model_def = onnx.helper.make_model(graph_def)

    onnx.save(model_def, onnx_output_path)
