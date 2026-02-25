# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import collections
import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast

import onnx
from packaging.version import Version

from qai_hub_models.utils.asset_loaders import PathLike
from qai_hub_models.utils.onnx.helpers import ONNXBundle

from .split_onnx import OnnxSplitter, save_model


def _target_name(
    name: str, deco_digit: bool = True, using_qairt_workflow: bool = False
) -> str:
    name = f"_{name}" if deco_digit and name.isdigit() else name
    # name = name.replace('.', '_')
    if not using_qairt_workflow:
        name = name.replace("/", "-")
    return name


def has_embedding_table(model: onnx.ModelProto) -> bool:
    return any(node.op_type == "Gather" for node in model.graph.node)


def get_onnx_input_output_names(
    onnxfile: PathLike,
    onnxmodel: onnx.ModelProto | None = None,
    deco_digit: bool = True,
    using_qairt_workflow: bool = False,
) -> tuple[list[str], list[str]]:
    onnxmodel = _load_model(onnxfile) if onnxmodel is None else onnxmodel
    input_names = [
        _target_name(
            i.name, deco_digit=deco_digit, using_qairt_workflow=using_qairt_workflow
        )
        for i in onnxmodel.graph.input
    ]
    output_names = [
        _target_name(
            i.name, deco_digit=deco_digit, using_qairt_workflow=using_qairt_workflow
        )
        for i in onnxmodel.graph.output
    ]
    return input_names, output_names


def get_split_tensors(
    onnxfile: PathLike,
    onnxmodel: onnx.ModelProto | None = None,
    include_first_input: bool = True,
) -> list[str]:
    """
    Model topology
            │ ←─────────  layers[0]  ────────────→ │       │ ←─────────  layers[-1]  ─────────────→ │
            │                                      │       │                                        │
    embed ────┬──────────── add0 ─┬─────────── add1 ── ┄┄┄  ─┬─────────────── add ─┬───────────── add ─── lmhead
            ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑       ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑
            │                                        │       │                                        │
            │                                        │       │                                        │
            valid splitting points
    """
    model = _load_model(onnxfile) if onnxmodel is None else onnxmodel

    def get_nodes() -> tuple[
        dict[str, onnx.NodeProto], dict[str, int], Mapping[str, str | None]
    ]:
        nodes = {i.name: i for i in model.graph.node}
        seq = {i.name: idx for idx, i in enumerate(model.graph.node)}
        producers: collections.defaultdict[str, str | None] = collections.defaultdict(
            lambda: None
        )
        producers.update({i.output[0]: i.name for i in model.graph.node})
        return nodes, seq, producers

    nodes, seq, producers = get_nodes()

    def maybe_skip_cast(a: str) -> str:
        if nodes[a].op_type == "Cast":
            inp = producers[nodes[a].input[0]]
            assert inp is not None
            return inp
        return a

    def can_visit(src: str, dst: str) -> bool:
        if seq[src] < seq[dst]:
            return False
        stack, visited = collections.deque([src]), set()
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            visited.add(cur)
            next_nodes = [
                producers[tensor]
                for tensor in nodes[cur].input
                if producers[tensor] is not None
            ]
            for name in next_nodes:
                if name is not None and name not in visited and seq[name] >= seq[dst]:
                    stack.append(name)
        return False

    def is_residual_add(nodename: str, strict: bool) -> bool:
        if nodes[nodename].op_type != "Add":
            return False
        a, b = (producers[tensor] for tensor in nodes[nodename].input)
        if a is None or b is None:
            return False
        a = maybe_skip_cast(a)
        b = maybe_skip_cast(b)
        begin, end = (a, b) if seq[a] < seq[b] else (b, a)
        if strict and nodes[begin].op_type != "Add":
            return False
        return can_visit(end, begin)

    def get_add0(add1: str) -> str:
        a, b = (producers[tensor] for tensor in nodes[add1].input)
        assert a is not None
        assert b is not None
        a = maybe_skip_cast(a)
        b = maybe_skip_cast(b)
        add0 = a if seq[a] < seq[b] else b
        assert is_residual_add(add0, strict=False)
        return add0

    def get_layer0_input(add0: str) -> str:
        a, b = (producers[tensor] for tensor in nodes[add0].input)
        assert a is not None
        assert b is not None
        return a if seq[a] < seq[b] else b

    residual_add_names = [name for name in nodes if is_residual_add(name, strict=True)]
    if len(residual_add_names) % 2 == 1 and has_embedding_table(model):
        # 'add0' is missing in residual_adds
        # 'add0' is not a valid split point if there is no embedding table provided
        add0 = get_add0(residual_add_names[0])
        residual_add_names.insert(0, add0)

    output_tensors: list[str] = []
    if include_first_input:
        layer0_input = maybe_skip_cast(get_layer0_input(residual_add_names[0]))
        output_tensors.append(nodes[layer0_input].output[0])
    output_tensors += [
        nodes[node].output[0] for i, node in enumerate(residual_add_names) if i % 2 == 1
    ]

    return output_tensors


def _load_model(
    onnxfile: PathLike,
    load_external_data: bool = False,
    model_cache: dict[str, onnx.ModelProto] | None = None,
) -> onnx.ModelProto:
    if model_cache is None:
        model_cache = {}
    cache_key = str(onnxfile)
    if onnxfile not in model_cache:
        model_cache[cache_key] = onnx.load(
            str(onnxfile), load_external_data=load_external_data
        )
    return model_cache[cache_key]


def _load_encoding(encodingfile: PathLike | None, no_merge: bool = False) -> Any:
    all_encodings = {}
    if encodingfile is not None:
        with open(encodingfile) as json_file:
            encodings = json.load(json_file)
        uses_lists = Version(encodings["version"]) >= Version("1.0.0")
        if uses_lists:
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }
        if no_merge:
            return encodings
        all_encodings.update(encodings["activation_encodings"])
        all_encodings.update(encodings["param_encodings"])
    return all_encodings


def _save_encoding(encodings: Any, encodingfile: PathLike) -> None:
    with open(encodingfile, "w") as json_file:
        json.dump(encodings, json_file, indent=4, sort_keys=True)


onnx_ret_t = TypeVar("onnx_ret_t", str, os.PathLike, ONNXBundle)


def split_onnx_by_names(
    onnxfile: onnx_ret_t,
    modelname: str,
    *list_of_output_tensors: str,
    output_dir: PathLike = ".",
    onnxmodel: onnx.ModelProto | None = None,
) -> list[onnx_ret_t]:
    """
    Split ONNX by the given output tensor names.

    Returns list[
        Path to an ONNX bundle or an ONNX Graph file for each split.
    ]
    """
    encodings = None
    uses_lists = None

    if isinstance(onnxfile, ONNXBundle):
        onnx_graph_file = str(onnxfile.onnx_graph_path)
        encoding_file = str(onnxfile.aimet_encodings_path)
        base_dir = str(onnxfile.bundle_path)
        dump_to_bundle = True
    else:
        onnx_graph_file = str(onnxfile)
        encoding_file = None
        base_dir = os.path.dirname(onnxfile)
        dump_to_bundle = False

    if encoding_file is not None:
        with open(encoding_file) as f:
            encodings = json.load(f)
        uses_lists = isinstance(encodings["activation_encodings"], list)
        if uses_lists:
            # Convert encodings to dictionary
            encodings["activation_encodings"] = {
                v["name"]: v for v in encodings["activation_encodings"]
            }
            encodings["param_encodings"] = {
                v["name"]: v for v in encodings["param_encodings"]
            }

    onnxmodel = (
        _load_model(onnx_graph_file, load_external_data=False)
        if onnxmodel is None
        else onnxmodel
    )
    splitter = OnnxSplitter(onnxmodel, verbose=False)
    using_external_data = OnnxSplitter.is_using_external_data(onnxmodel)

    list_of_output_tensors = tuple([i.split(",") for i in list_of_output_tensors])  # type: ignore[misc]
    num_splits = len(list_of_output_tensors) + 1

    # 1. split model
    output_paths: list[ONNXBundle | Path] = []
    new_model_info = []
    for i, subgraph in enumerate(splitter.split(list_of_output_tensors)):
        new_basename = f"{modelname}_{i + 1}_of_{num_splits}"
        input_tensor_names = [i.name for i in subgraph.input]
        output_tensor_names = [i.name for i in subgraph.output]
        new_model_info.append([new_basename, input_tensor_names, output_tensor_names])

        submodel = onnx.helper.make_model(
            subgraph, opset_imports=onnxmodel.opset_import
        )
        if (
            not using_external_data
            and submodel.ByteSize() < onnx.checker.MAXIMUM_PROTOBUF
        ):
            onnx.checker.check_model(submodel)

        if using_external_data:
            onnx.load_external_data_for_model(submodel, base_dir=str(base_dir))

        ext = ".aimet" if encoding_file is not None else ".onnx"
        part_root_path = Path(output_dir) / (new_basename + ext)
        part_root_path.mkdir(parents=True, exist_ok=True)

        newonnxfile = part_root_path / (new_basename + ".onnx")
        save_model(submodel, newonnxfile, using_external_data or dump_to_bundle)

        # Save subset of encodings
        new_encodings_path = None
        if encodings is not None:
            new_encodings = deepcopy(encodings)

            activation_names = (
                {o for x in submodel.graph.node for o in x.output}
                | {x.name for x in submodel.graph.input}
                | {x.name for x in submodel.graph.output}
            )
            param_names = {x.name for x in submodel.graph.initializer}

            for k in encodings["activation_encodings"]:
                if k not in activation_names:
                    del new_encodings["activation_encodings"][k]

            for k in encodings["param_encodings"]:
                if k not in param_names:
                    del new_encodings["param_encodings"][k]

            # Due to AISW-152612 we cannot have activations encodings for
            # Gather ops, so we clean them up.
            for node in submodel.graph.node:
                if (
                    node.op_type == "Gather"
                    and node.output[0] in new_encodings["activation_encodings"]
                ):
                    del new_encodings["activation_encodings"][node.output[0]]

            if uses_lists:
                # convert back
                new_encodings["activation_encodings"] = list(
                    new_encodings["activation_encodings"].values()
                )
                new_encodings["param_encodings"] = list(
                    new_encodings["param_encodings"].values()
                )

            new_encodings_path = part_root_path / (new_basename + ".encodings")
            with open(new_encodings_path, "w") as write_file:
                json.dump(new_encodings, write_file, indent=4, sort_keys=True)

        if dump_to_bundle:
            # This is a bundle (either model.onnx + model.weights, or model.onnx + model.encodings + (optional) model.weights)
            # Therefore, return the bundle directory
            output_paths.append(ONNXBundle.from_bundle_path(part_root_path))
        else:
            # This is a single ONNX graph file.
            output_paths.append(newonnxfile)

    return cast(list[onnx_ret_t], output_paths)


def _get_lm_head_sizes(onnxmodel: onnx.ModelProto) -> tuple[int, int]:
    """Get dimensions of the LM head : embedding_size, vocab_size"""
    lm_head_weight_name = next(
        node.input[1]
        for node in reversed(onnxmodel.graph.node)
        if node.op_type in ("Conv", "MatMul", "Gemm")
    )
    (lm_head_weight,) = (
        i for i in onnxmodel.graph.initializer if lm_head_weight_name == i.name
    )
    if len(lm_head_weight.dims) == 2:
        embedding_size, vocab_size = lm_head_weight.dims
    else:
        (lm_head,) = (i for i in onnxmodel.graph.node if lm_head_weight.name in i.input)
        if lm_head.op_type == "Conv":
            attr_group = [i.i for i in lm_head.attribute if i.name == "group"]
            group = attr_group[0] if len(attr_group) == 1 else 1
            grouped_vocab, group_size, _, _ = lm_head_weight.dims
            vocab_size, embedding_size = grouped_vocab // group, group * group_size
        elif lm_head.op_type == "MatMul":
            group, group_size, vocab_size = lm_head_weight.dims
            embedding_size = group * group_size
        else:
            raise RuntimeError(f"Unexpected lm_head op_type:{lm_head}")

    return embedding_size, vocab_size


def fill_input_encodings_of_split(
    onnxmodel: onnx.ModelProto,
    encodingfile: PathLike | None,
    output_tensor_list: list[str],
) -> None:
    changed = False
    encodings = _load_encoding(encodingfile, no_merge=True)
    enc_act, enc_param = encodings["activation_encodings"], encodings["param_encodings"]
    producer = {tensor: node for node in onnxmodel.graph.node for tensor in node.output}
    for split_tensor in output_tensor_list:
        if split_tensor not in enc_act:
            assert split_tensor in producer
            input_tensor = producer[split_tensor].input[0]  # use only 1st input
            if input_tensor in producer:
                while input_tensor not in enc_act and input_tensor not in enc_param:
                    input_tensor = producer[input_tensor].input[0]
                input_encoding = (
                    enc_act[input_tensor]
                    if input_tensor in enc_act
                    else enc_param[input_tensor]
                )
                enc_act[split_tensor] = input_encoding
                changed = True

    if encodingfile is not None and changed:
        backup = f"{encodingfile}.bak"
        if not os.path.exists(backup):
            shutil.move(encodingfile, backup)
        _save_encoding(encodings, encodingfile)


def split_onnx(
    onnxfile: onnx_ret_t,
    modelname: str,
    num_splits: int,
    num_layers_per_split: int | None = None,
    output_dir: PathLike = ".",
    split_embedding: bool = False,
    using_qairt_workflow: bool = False,
) -> list[onnx_ret_t]:
    """
    Split ONNX by the given number of splits.

    Returns list[
        Path to an ONNX bundle or an ONNX Graph file for each split.
    ]
    """

    def _is_cache(layer: int, name: str) -> bool:
        return re.search(f"past_(key|value)_{layer}_", name) is not None

    num_splits = int(num_splits)

    if isinstance(onnxfile, ONNXBundle):
        onnx_graph_file = str(onnxfile.onnx_graph_path)
    else:
        onnx_graph_file = str(onnxfile)

    onnxmodel = _load_model(onnx_graph_file, load_external_data=False)
    _input_names, output_names = get_onnx_input_output_names(
        onnx_graph_file,
        onnxmodel=onnxmodel,
        deco_digit=False,
        using_qairt_workflow=using_qairt_workflow,
    )
    output_tensor_list = get_split_tensors(
        onnx_graph_file, onnxmodel=onnxmodel, include_first_input=split_embedding
    )

    # Infer the shape of per-layer tensors
    (input_tokens,) = (
        i for i in onnxmodel.graph.input if i.name in {"input_ids", "inputs_embeds"}
    )
    input_tokens_shape = tuple(
        i.dim_value for i in input_tokens.type.tensor_type.shape.dim
    )
    batch_size = input_tokens_shape[0]
    seq_length = input_tokens_shape[1]

    embedding_size, _vocab_size = _get_lm_head_sizes(onnxmodel)

    per_layer_output_value_info = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, [batch_size, seq_length, embedding_size]
        )
        for name in output_tensor_list
    ]
    onnxmodel.graph.value_info.extend(per_layer_output_value_info)

    names_to_split = []
    # We should only split the embedding if there is an embedding table in the model
    split_embedding = split_embedding and has_embedding_table(onnxmodel)
    if split_embedding:
        first_output_tensors = output_tensor_list[0].split(",")
        if (
            isinstance(onnxfile, ONNXBundle)
            and onnxfile.aimet_encodings_path is not None
        ):
            fill_input_encodings_of_split(
                onnxmodel, onnxfile.aimet_encodings_path, first_output_tensors
            )
        names_to_split.append(output_tensor_list[0])
        output_tensor_list.pop(0)

    num_layers = len(output_tensor_list)

    computed_num_layers_per_split = (
        math.ceil((num_layers - 1) / num_splits)
        if split_embedding
        else math.ceil(num_layers / num_splits)
    )

    if num_layers_per_split is None:
        num_layers_per_split = computed_num_layers_per_split

    effective_num_splits = num_splits - 1 if split_embedding else num_splits
    effective_num_layers = num_layers - 1 if split_embedding else num_layers
    if effective_num_splits != math.ceil(effective_num_layers / num_layers_per_split):
        print(
            f"Warning: specified num_layers_per_split ({num_layers_per_split}) is not compatible with model. Overwriting with {computed_num_layers_per_split}"
        )
        num_layers_per_split = computed_num_layers_per_split

    past_key_values = {
        layer: [output for output in output_names if _is_cache(layer, output)]
        for layer in range(num_layers)
    }

    for layer_end in range(num_layers_per_split, num_layers, num_layers_per_split):
        outputs = [output_tensor_list[layer_end - 1]]
        for layer in range(layer_end - num_layers_per_split, layer_end):
            outputs += past_key_values[layer]
        names_to_split.append(",".join(outputs))

    names_to_split = names_to_split[: num_splits - 1]
    assert num_splits == len(names_to_split) + 1, (
        f"Failed to split into {num_splits} pieces!"
    )
    return split_onnx_by_names(
        onnxfile,
        modelname,
        *names_to_split,
        output_dir=output_dir,
        onnxmodel=onnxmodel,
    )
