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
from collections.abc import Generator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar, cast

import onnx
from packaging.version import Version

from qai_hub_models.models.templates.llm.split_onnx_utils.split_onnx import (
    OnnxSplitter,
    save_model,
)
from qai_hub_models.utils.asset_loaders import PathLike
from qai_hub_models.utils.onnx.helpers import ONNXBundle


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
            # If input is a graph input (no producer), don't skip the cast
            if inp is None:
                return a
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
        if not is_residual_add(add0, strict=False):
            # VLM models: add0 takes a graph input (e.g. input_embeds)
            # directly, so is_residual_add fails because the producer is
            # None. Verify it's still a valid residual Add with one graph
            # input.
            node = nodes[add0]
            assert node.op_type == "Add"
            a0, b0 = (producers[t] for t in node.input)
            assert a0 is None or b0 is None
        return add0

    def get_layer0_input(add0: str) -> str:
        a, b = (producers[tensor] for tensor in nodes[add0].input)
        assert a is not None
        assert b is not None
        return a if seq[a] < seq[b] else b

    residual_add_names = [name for name in nodes if is_residual_add(name, strict=True)]
    if len(residual_add_names) % 2 == 1:
        # 'add0' is missing in residual_adds because its input comes from
        # embedding (LLM) or inputs_embeds (VLM), not another Add node.
        # We need to insert it to get the correct layer count.
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


def _iter_graphs(graph: onnx.GraphProto) -> Generator[onnx.GraphProto, None, None]:
    """Graph and every subgraph nested in its nodes' attributes."""
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _iter_graphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _iter_graphs(subgraph)


def _encodings_to_name_dicts(encodings: dict) -> bool:
    """Key activation/param encodings by tensor name, in place.

    2.0.0 has one "encodings" list with no param/activation split; it goes to
    activation_encodings and _encodings_from_name_dicts writes it back.
    Returns whether the file stores lists (1.0.0 and later).
    """
    if "encodings" in encodings:
        encodings["activation_encodings"] = {
            v["name"]: v for v in encodings.pop("encodings")
        }
        encodings["param_encodings"] = {}
        return True
    if not isinstance(encodings["activation_encodings"], list):
        return False
    encodings["activation_encodings"] = {
        v["name"]: v for v in encodings["activation_encodings"]
    }
    encodings["param_encodings"] = {v["name"]: v for v in encodings["param_encodings"]}
    return True


def _encodings_from_name_dicts(encodings: dict) -> None:
    """Undo _encodings_to_name_dicts for a list-based (1.0.0+) file, in place."""
    activations = list(encodings.pop("activation_encodings").values())
    params = list(encodings.pop("param_encodings").values())
    if Version(encodings["version"]) >= Version("2.0.0"):
        encodings["encodings"] = activations + params
    else:
        encodings["activation_encodings"] = activations
        encodings["param_encodings"] = params


def _load_encoding(encodingfile: PathLike | None, no_merge: bool = False) -> Any:
    all_encodings = {}
    if encodingfile is not None:
        with open(encodingfile) as json_file:
            encodings = json.load(json_file)
        _encodings_to_name_dicts(encodings)
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
    reuse_emitted_graph_outputs: bool = False,
) -> list[onnx_ret_t]:
    """
    Split ONNX by the given output tensor names.

    reuse_emitted_graph_outputs: let a later split take an earlier split's
        graph output as a leaf input. See ``OnnxSplitter``; needed only by
        KV-shared models such as Gemma4.

    Returns list[
        Path to an ONNX bundle or an ONNX Graph file for each split.
    ]
    """
    encodings = None
    uses_lists = None

    if isinstance(onnxfile, ONNXBundle):
        onnx_graph_file = str(onnxfile.onnx_graph_path)
        encoding_file = (
            str(onnxfile.aimet_encodings_path)
            if onnxfile.aimet_encodings_path is not None
            else None
        )
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
        uses_lists = _encodings_to_name_dicts(encodings)

    onnxmodel = (
        _load_model(onnx_graph_file, load_external_data=False)
        if onnxmodel is None
        else onnxmodel
    )
    splitter = OnnxSplitter(
        onnxmodel,
        verbose=False,
        reuse_emitted_graph_outputs=reuse_emitted_graph_outputs,
    )
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

            # Scan/Loop/If bodies are quantized too (e.g. Qwen3.5's delta
            # rule), and their tensors aren't in the top-level graph.node.
            subgraphs = list(_iter_graphs(submodel.graph))
            activation_names = (
                {o for g in subgraphs for x in g.node for o in x.output}
                | {x.name for g in subgraphs for x in g.input}
                | {x.name for g in subgraphs for x in g.output}
            )
            param_names = {x.name for g in subgraphs for x in g.initializer}
            if Version(encodings["version"]) >= Version("2.0.0"):
                # Params share the activation dict (see _encodings_to_name_dicts).
                activation_names |= param_names

            for k in encodings["activation_encodings"]:
                if k not in activation_names:
                    del new_encodings["activation_encodings"][k]

            for k in encodings["param_encodings"]:
                if k not in param_names:
                    del new_encodings["param_encodings"][k]

            # Due to AISW-152612 we cannot have activations encodings for
            # Gather ops, so we clean them up.
            for node in (n for g in subgraphs for n in g.node):
                if (
                    node.op_type == "Gather"
                    and node.output[0] in new_encodings["activation_encodings"]
                ):
                    del new_encodings["activation_encodings"][node.output[0]]

            if uses_lists:
                _encodings_from_name_dicts(new_encodings)

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
    initializers = {i.name: i for i in onnxmodel.graph.initializer}

    # The lm_head weight is usually a direct initializer input to the final
    # MatMul/Conv/Gemm. Dynamo (torch.export) ONNX, however, often feeds the
    # weight through a Transpose of the named initializer (e.g. an nn.Linear
    # lm_head emits ``MatMul(x, Transpose(model.lm_head.weight))``). In that
    # case the weight name resolves to a Transpose node rather than an
    # initializer; follow it to the source initializer and remember that the
    # two trailing dims are swapped relative to the direct-initializer case.
    transposed = False
    if lm_head_weight_name not in initializers:
        producer = next(
            (
                n
                for n in onnxmodel.graph.node
                if lm_head_weight_name in n.output and n.op_type == "Transpose"
            ),
            None,
        )
        if producer is not None and producer.input[0] in initializers:
            lm_head_weight_name = producer.input[0]
            transposed = True

    (lm_head_weight,) = (
        i for i in onnxmodel.graph.initializer if lm_head_weight_name == i.name
    )
    if len(lm_head_weight.dims) == 2:
        if transposed:
            # Transposed weight is [vocab_size, embedding_size].
            vocab_size, embedding_size = lm_head_weight.dims
        else:
            embedding_size, vocab_size = lm_head_weight.dims
    else:
        (lm_head,) = (
            i
            for i in onnxmodel.graph.node
            if lm_head_weight.name in i.input and i.op_type in {"Conv", "MatMul"}
        )
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
        if Version(encodings["version"]) >= Version("1.0.0"):
            _encodings_from_name_dicts(encodings)
        _save_encoding(encodings, encodingfile)


def split_onnx(
    onnxfile: onnx_ret_t,
    modelname: str,
    num_splits: int,
    num_layers_per_split: int | None = None,
    output_dir: PathLike = ".",
    split_embedding: bool = False,
    split_lm_head: bool = False,
    using_qairt_workflow: bool = False,
    splitting_points: list[int] | None = None,
    reuse_emitted_graph_outputs: bool = False,
) -> list[onnx_ret_t]:
    """
    Split ONNX by the given number of splits.

    When split_lm_head is True, the LM head is separated into its own
    final part. This reduces the compute packed into the last CL part.
    The total number of output parts increases by one (the extra LM head part).

    splitting_points: explicit transformer-block boundaries (end-layer indices,
        exclusive) instead of uniform num_layers_per_split; [15] gives
        layers[0:15] and layers[15:N]. Lets a cut land on a KV boundary that
        uniform splitting would miss (Gemma4 shares KV after layer 15). The
        split_lm_head part is still appended after the last block.

    reuse_emitted_graph_outputs: let a later split take an earlier split's
        graph output as a leaf input instead of re-deriving its producers.
        See ``OnnxSplitter``; needed only by KV-shared models such as Gemma4.

    Returns list[
        Path to an ONNX bundle or an ONNX Graph file for each split.
    ]
    """

    def _is_cache(layer: int, name: str) -> bool:
        # Linear-attention layers (Qwen3.5) carry conv/recurrent state instead of KV.
        return (
            re.search(f"(?:past|swa)_(nativekvcache__)?(key|value)_{layer}_", name)
            is not None
            or re.search(f"(?:conv|recurrent)_state_{layer}_", name) is not None
        )

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

    # Check if embedding table exists before determining split points
    # VLM models don't have embedding table (Gather op using input_ids) in the ONNX
    split_embedding = split_embedding and has_embedding_table(onnxmodel)

    output_tensor_list = get_split_tensors(
        onnx_graph_file, onnxmodel=onnxmodel, include_first_input=split_embedding
    )

    # Infer the shape of per-layer tensors
    # Note: VLM models use "input_embeds" (singular), LLMs use "input_ids"
    (input_tokens,) = (
        i
        for i in onnxmodel.graph.input
        if i.name in {"input_ids", "input_embeds", "inputs_embeds"}
    )

    # Handle both concrete and symbolic (dynamic) dimensions
    def get_dim(dim: Any) -> int | str:
        """Return dim_value for concrete dims, or dim_param (symbolic name) for dynamic dims."""
        if dim.dim_param:
            return dim.dim_param  # Symbolic dimension (e.g., "seq_len")
        return dim.dim_value  # Concrete dimension

    input_tokens_dims = input_tokens.type.tensor_type.shape.dim
    batch_size = get_dim(input_tokens_dims[0])
    seq_length = get_dim(input_tokens_dims[1])

    embedding_size, _vocab_size = _get_lm_head_sizes(onnxmodel)

    per_layer_output_value_info = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, [batch_size, seq_length, embedding_size]
        )
        for name in output_tensor_list
    ]
    onnxmodel.graph.value_info.extend(per_layer_output_value_info)

    names_to_split = []
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

    # When splitting the LM head, one of the num_splits is reserved for it
    min_splits = 1 + int(split_embedding) + int(split_lm_head)
    assert num_splits >= min_splits, (
        f"num_splits must be >= {min_splits} (split_embedding={split_embedding}, "
        f"split_lm_head={split_lm_head}), got {num_splits}"
    )
    num_transformer_block_splits = (
        num_splits - int(split_embedding) - int(split_lm_head)
    )

    past_key_values = {
        layer: [output for output in output_names if _is_cache(layer, output)]
        for layer in range(num_layers)
    }

    if splitting_points is not None:
        # An unsorted, duplicate or 0 boundary would make
        # output_tensor_list[layer_end - 1] index the wrong layer (or -1).
        assert all(isinstance(p, int) for p in splitting_points), (
            f"splitting_points must be ints, got {splitting_points}"
        )
        assert splitting_points == sorted(set(splitting_points)), (
            f"splitting_points must be strictly increasing and unique, "
            f"got {splitting_points}"
        )
        assert splitting_points[0] > 0, (
            f"splitting_points must be > 0 (0 is not a valid block boundary), "
            f"got {splitting_points}"
        )
        block_ends = [p for p in splitting_points if p < num_layers]
        layer_start = 0
        for layer_end in block_ends:
            outputs = [output_tensor_list[layer_end - 1]]
            for layer in range(layer_start, layer_end):
                outputs += past_key_values[layer]
            names_to_split.append(",".join(outputs))
            layer_start = layer_end
        last_block_start = block_ends[-1] if block_ends else 0
    else:
        computed_num_layers_per_split = math.ceil(
            num_layers / num_transformer_block_splits
        )

        if num_layers_per_split is None:
            num_layers_per_split = computed_num_layers_per_split

        if num_transformer_block_splits != math.ceil(num_layers / num_layers_per_split):
            print(
                f"Warning: specified num_layers_per_split ({num_layers_per_split}) is not compatible with model. Overwriting with {computed_num_layers_per_split}"
            )
            num_layers_per_split = computed_num_layers_per_split

        for layer_end in range(num_layers_per_split, num_layers, num_layers_per_split):
            outputs = [output_tensor_list[layer_end - 1]]
            for layer in range(layer_end - num_layers_per_split, layer_end):
                outputs += past_key_values[layer]
            names_to_split.append(",".join(outputs))
        last_block_start = (num_transformer_block_splits - 1) * num_layers_per_split

    if split_lm_head:
        last_cl_outputs = [output_tensor_list[-1]]
        for layer in range(last_block_start, num_layers):
            last_cl_outputs += past_key_values[layer]
        names_to_split.append(",".join(last_cl_outputs))

    expected_split_points = num_splits - 1
    if len(names_to_split) != expected_split_points:
        raise ValueError(
            f"Expected {expected_split_points} split points for {num_splits} splits, "
            f"but got {len(names_to_split)}. Check num_splits, num_layers_per_split, "
            f"split_embedding={split_embedding}, split_lm_head={split_lm_head}."
        )
    return split_onnx_by_names(
        onnxfile,
        modelname,
        *names_to_split,
        output_dir=output_dir,
        onnxmodel=onnxmodel,
        reuse_emitted_graph_outputs=reuse_emitted_graph_outputs,
    )
