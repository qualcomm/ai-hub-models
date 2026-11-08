# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Per-layer sub-graph splitting for Phase B (isolated-input strategy).

Instead of one giant part-graph with all taps bolted on (which overflows the
2 GB output buffer and requires fragile whole-graph surgery), we cut each part
into ONE SUB-GRAPH PER DECODER LAYER using the production split machinery
(``split_onnx_by_names`` at residual boundaries). Each sub-graph:

  * takes the previous layer's residual as an INPUT (auto-wired by the splitter)
    plus the shared attention_mask / position_ids and this layer's KV inputs;
  * outputs this layer's residual + its o_proj / gate_up / down taps + KV out.

Because every sub-graph is fed golden inputs at its boundary, its SQNR is that
layer's own transfer function (isolated, immune to upstream accumulation), and a
compile/runtime failure localizes to a single layer. Each job's output is a
handful of small tensors -> no size limit, and the splitter subsets encodings
per sub-graph, so no manual encoding/value_info surgery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import onnx
from tap_per_layer import (
    Tap,
    _sanitize,
    _scratch_dir,
    build_tapped_bundle,
    discover_taps,
)

from qai_hub_models.models.templates.llm.split_onnx_utils.utils import (
    get_split_tensors,
    split_onnx_by_names,
)
from qai_hub_models.utils.onnx.helpers import ONNXBundle


@dataclass
class LayerSubgraph:
    """One decoder layer carved out of a part as its own ONNX bundle."""

    layer: int  # global decoder-layer index
    bundle: ONNXBundle  # the sub-graph (+ subset encodings)
    input_names: list[str]  # graph inputs (prev residual, mask, pos, KV)
    residual_out: str  # this layer's residual output tensor
    taps: list[Tap] = field(default_factory=list)  # o_proj / gate_up / down (+residual)
    # sanitized input name -> original tensor name (the key in captured part inputs)
    input_sources: dict[str, str] = field(default_factory=dict)


def split_part_into_layers(
    part: object, bundle: ONNXBundle, base_layer: int
) -> list[LayerSubgraph]:
    """Cut ``bundle`` into per-layer sub-graphs at residual boundaries.

    ``base_layer`` is the global index of this part's first decoder layer.
    Returns one LayerSubgraph per layer, in order. No device work.
    """
    out_dir = _scratch_dir() / part.__class__.__name__
    out_dir.mkdir(parents=True)

    # Residual boundaries in layer order (same topology the real split uses).
    # include_first_input=False -> per-layer residual OUTPUTS only.
    residual_tensors = get_split_tensors(
        bundle.onnx_graph_path, onnxmodel=None, include_first_input=False
    )

    # Cut at EVERY residual boundary (including the last). split_onnx_by_names
    # emits len(boundaries)+1 sub-graphs: the first N are the clean per-layer
    # sub-graphs (each wired to consume only the previous boundary), and the
    # trailing (N+1)th is the leftover tail (final norm / anything past the last
    # layer residual). We keep only the N per-layer sub-graphs.
    #
    # NB: cutting at all-but-last instead lumps the last layer into the leftover
    # graph, which then (correctly) declares every upstream boundary it still
    # references as an input -- i.e. the last "layer" isn't isolated. Cutting at
    # the last boundary too makes every layer a proper isolated sub-graph.
    n_layers = len(residual_tensors)
    sub_bundles = split_onnx_by_names(
        bundle,
        f"{part.__class__.__name__}_layer",
        *residual_tensors,
        output_dir=str(out_dir),
    )

    subgraphs: list[LayerSubgraph] = []
    for pos, sub in enumerate(sub_bundles[:n_layers]):
        sub_bundle = (
            sub if isinstance(sub, ONNXBundle) else ONNXBundle.from_bundle_path(sub)
        )
        model = onnx.load(str(sub_bundle.onnx_graph_path), load_external_data=False)
        input_names = [i.name for i in model.graph.input]
        residual_out = residual_tensors[pos]
        subgraphs.append(
            LayerSubgraph(
                layer=base_layer + pos,
                bundle=sub_bundle,
                input_names=input_names,
                residual_out=residual_out,
            )
        )
    return subgraphs


def add_projection_taps(sub: LayerSubgraph) -> LayerSubgraph:
    """Add this layer's o_proj / gate_up / down taps as outputs of the sub-graph.

    The sub-graph is a single layer, so this surgery is tiny and its output stays
    small. Reuses tap_per_layer's pure-topology discovery (``discover_taps``,
    which finds the residual-add branch inputs) with an explicit base_layer so
    the taps carry this layer's global index. ``build_tapped_bundle`` then
    promotes the projection tensors to graph outputs (stripping value_info/IO
    collisions and subsetting encodings).
    """
    # part=None: discover_taps only needs `part` for the layer base, which we
    # override with base_layer here (the sub-graph has exactly one layer).
    layer_taps, _dropped = discover_taps(None, sub.bundle, base_layer=sub.layer)

    # The residual boundary is already a graph output of the sub-graph, so
    # discover_taps skips it; record it explicitly for the comparison. The
    # projection taps (o_proj/gate_up/down) still need promoting to outputs.
    proj_taps = [t for t in layer_taps if t.projection != "residual"]
    taps = list(proj_taps)
    taps.append(
        Tap(sub.residual_out, _sanitize(sub.residual_out), "residual", sub.layer)
    )

    tapped = build_tapped_bundle(sub.bundle, proj_taps, f"layer_{sub.layer}")
    sub.bundle = tapped
    sub.taps = taps
    _sanitize_input_names(sub)
    return sub


def _sanitize_input_names(sub: LayerSubgraph) -> None:
    """Rename graph inputs containing '/' or '.' to their sanitized form, in place.

    AI Hub validates a dataset against the compiled model's raw input names, but
    the on-device runtime expects sanitized ones, so an input like the previous
    layer's residual ('/model/model/layers.0/Add_1_output_0') can satisfy only one.
    """
    model = onnx.load(str(sub.bundle.onnx_graph_path), load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    renames = {
        i.name: _sanitize(i.name)
        for i in model.graph.input
        if i.name not in initializers and _sanitize(i.name) != i.name
    }
    if not renames:
        return

    graph = model.graph
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        vi.name = renames.get(vi.name, vi.name)
    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name in renames:
                node.input[idx] = renames[name]
    onnx.save(model, str(sub.bundle.onnx_graph_path))

    enc_path = sub.bundle.aimet_encodings_path
    if enc_path is not None:
        data = json.loads(enc_path.read_text())
        acts = data.get("activation_encodings")
        if isinstance(acts, dict):
            data["activation_encodings"] = {
                renames.get(k, k): v for k, v in acts.items()
            }
        elif isinstance(acts, list):
            for e in acts:
                e["name"] = renames.get(e.get("name"), e.get("name"))
        enc_path.write_text(json.dumps(data))

    sub.input_names = [renames.get(n, n) for n in sub.input_names]
    sub.input_sources = {new: old for old, new in renames.items()}


def build_layer_feed(
    sub: LayerSubgraph,
    input_order: list[str],
    cap: object,
    golden_residual_by_layer: dict[int, list],
) -> dict[str, list]:
    """Assemble the golden-injected device input feed for one layer sub-graph.

    Sources (all golden, per the isolated-input design):
      * the previous layer's residual -> ``golden_residual_by_layer[sub.layer-1]``
        (the golden residual VALUES for layer L-1, keyed by LAYER INDEX -- NOT by
        tensor name, because the per-layer sub-graph and the full-part golden
        graph are independently produced and renumber tensors differently). The
        first transformer layer has no previous residual; its residual input is
        the PART entry hidden state, which comes from cap.inputs.
      * shared attention_mask / position_ids_* and this layer's
        past_key_L_in / past_value_L_in -> from cap.inputs (part inputs).

    Returns {input_name: [per-sample arrays]} in the graph's input order.
    """
    n = len(cap.inputs)
    feed: dict[str, list] = {name: [] for name in input_order}

    # Captured part inputs are keyed by the original (unsanitized) tensor name.
    def source(name: str) -> str:
        return sub.input_sources.get(name, name)

    # The residual input is the one sub-graph input NOT sourced from the part
    # inputs (mask/pos/KV all come from cap.inputs).
    part_keys = set(cap.inputs[0].keys()) if n else set()
    residual_input_name = next(
        (name for name in input_order if source(name) not in part_keys), None
    )

    # Golden residual VALUES for the previous layer (by index). Absent for the
    # first transformer layer, whose residual input is instead a part input
    # (the entry hidden state) already present in cap.inputs.
    prev_vals = golden_residual_by_layer.get(sub.layer - 1)

    for i in range(n):
        part_in = cap.inputs[i]
        for name in input_order:
            if source(name) in part_in:
                # Shared inputs (mask/pos/KV) and the part-entry hidden state.
                feed[name].append(part_in[source(name)])
            elif name == residual_input_name and prev_vals is not None:
                # Previous layer's residual, injected from the golden sim run.
                feed[name].append(prev_vals[i])

    missing = [k for k in input_order if not feed[k]]
    if missing:
        raise KeyError(
            f"layer {sub.layer}: could not populate inputs {missing}. "
            f"part-input keys={sorted(part_keys)[:8]}, "
            f"residual_input_name={residual_input_name}, "
            f"have_prev_residual={prev_vals is not None}"
        )
    return feed
