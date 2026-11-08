# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tap per-layer outputs -- fine sim-vs-device localization via per-layer split.

(The driver labels this "Phase B".) Given a Part to instrument, split it into
one sub-graph PER DECODER LAYER using the production split utility (cut at
residual-add boundaries), then for each layer sub-graph tap its projection
outputs (o_proj, gate/up, down) and residual. Each sub-graph is fed the GOLDEN
previous-layer residual + shared mask/pos + this layer's KV (isolated inputs),
compiled and run independently, and SQNR is computed at each tap. Because every
job outputs only one layer's small tensors, this avoids the output-size limits
of tapping a whole Part at once, and a failure localizes to a single layer.

Tap selection rule (keeps encodings untouched): only tensors that ALREADY carry
an INT activation encoding are tapped -- the compile path uses --quantize_io
(all graph I/O quantized), so an un-encoded (or floated) new output would break
compilation. Since encodings are keyed by tensor name and independent of
graph-output status, promoting an already-quantized tensor to an output needs
zero encoding edits.

The interval between consecutive taps that craters (down/residual vs o_proj/
gate_up) names the culprit region within the layer.
"""

from __future__ import annotations

import atexit
import collections
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import qai_hub as hub
import torch
from harness import HarnessContext
from metrics import TensorMetric, Verdict

from qai_hub_models import TargetRuntime
from qai_hub_models.models.templates.llm.model import LLMDynamic_AIMETOnnx
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.onnx.helpers import ONNXBundle

# Shape-only ops we transparently walk through when back-tracing.
_SHAPE_OPS = {"Transpose", "Unsqueeze", "Squeeze", "Reshape", "Cast", "Identity"}
# Column order in the report, in block EXECUTION order (topographic): the
# projections run first, then "residual" -- the layer-output boundary -- comes
# last because down_proj feeds the residual add that produces it.
_INTERVAL_ORDER = ["o_proj", "gate_up", "down", "residual"]


def _scratch_dir() -> Path:
    """A fresh system-temp scratch dir for tapped bundles, auto-removed on exit.

    Uses tempfile (system TMP, honoring $TMPDIR) rather than a hardcoded path,
    and registers cleanup so we don't leak the (multi-GB) bundle copies.
    """
    d = Path(tempfile.mkdtemp(prefix="sim_vs_device_taps_"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


def select_part(
    verdict: Verdict | None, ctx: HarnessContext, override: str | None
) -> str | None:
    """Pick which Part to instrument.

    override wins; else (verdict present) the first suspect Part for
    localized/shared_op; else None (verdict absent/clean -> caller decides).
    """
    if override is not None:
        return override
    if verdict is None:
        return None
    if verdict.kind in ("localized", "shared_op") and verdict.suspect_parts:
        return verdict.suspect_parts[0]
    return None


def _sanitize(name: str) -> str:
    """Runtime output-name sanitization (matches get_graph_output_spec)."""
    return name.replace("/", "_").replace(".", "_")


@dataclass
class Tap:
    tensor_name: str  # original ONNX tensor name
    output_name: str  # sanitized name as it comes back from the device
    projection: str  # one of _PROJECTION_MARKERS keys
    layer: int


@dataclass
class PhaseBResult:
    part_name: str
    taps: list[Tap]
    # metric[layer][projection] = TensorMetric
    grid: dict[int, dict[str, TensorMetric]] = field(default_factory=dict)
    dropped_unencoded: list[str] = field(default_factory=list)


def _is_int_encoding(e: dict) -> bool:
    """True iff an encoding entry is integer-quantized (tappable).

    Per the AIMET 1.0.0 encoding spec, a floated tensor has ``dtype == "FLOAT"``
    and omits the INT-only fields (scale/offset/is_sym). We must NOT tap such a
    tensor: promoting a float tensor to a graph output collides with the
    ``--quantize_io`` compile (which requires all graph I/O to be quantized).
    Treat an entry as int iff dtype is INT, or (defensively) it still carries a
    scale and isn't marked FLOAT.
    """
    return e.get("dtype") == "INT" or ("scale" in e and e.get("dtype") != "FLOAT")


def _encoded_activation_names(bundle: ONNXBundle) -> set[str]:
    """Names with an INT (tappable) activation encoding (list or dict schema).

    Float-encoded tensors are excluded so that intentionally-floated ops (e.g.
    a layer forced to float for a verification experiment) are auto-skipped as
    taps rather than breaking the quantize_io compile.
    """
    if bundle.aimet_encodings_path is None:
        return set()
    data = json.loads(bundle.aimet_encodings_path.read_text())
    acts = data.get("activation_encodings", [])
    if isinstance(acts, dict):
        return {
            k for k, v in acts.items() if isinstance(v, dict) and _is_int_encoding(v)
        }
    return {
        e["name"]
        for e in acts
        if isinstance(e, dict) and "name" in e and _is_int_encoding(e)
    }


def _first_transformer_layer(part: object) -> int:
    """Global index of this Part's first decoder layer.

    Sum of the layer counts of the preceding parts, each counted from its own
    split ONNX graph (harness.layers_per_part) -- so it matches the real split
    (e.g. num_layers_per_split=15 on 32 layers -> 15/15/2), not an assumed
    uniform ceil(num_layers / num_transformer_parts) distribution.
    """
    from harness import layers_per_part

    lpp = layers_per_part(part._presplit)
    # part_id is 1-indexed; sum layers of all parts before this one (0-indexed).
    return sum(lpp[i] for i in range(part.part_id - 1))


# TODO(unify-with-production): _residual_add_boundaries below re-implements the
# residual-add topology walk (is_residual_add / can_visit / maybe_skip_cast)
# that currently lives as private closures inside
# split_onnx_utils.utils.get_split_tensors. In a future PR, factor that walk out
# of get_split_tensors into a reusable exported helper (e.g.
# get_residual_add_nodes(model) -> ordered list of Add NodeProtos) and have BOTH
# the production splitter AND this tool consume it. That removes the duplication
# and guarantees our tap points can never drift from how the graph is actually
# cut. Until then, keep this in lockstep with the production logic.


def _residual_add_boundaries(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    """Ordered residual-add nodes (attn-add, mlp-add, attn-add, ...) by topology.

    Faithful re-implementation of the production splitter's residual-add
    detection (split_onnx_utils.utils.get_split_tensors): an Add whose two
    producer inputs include a skip path reachable from the branch operand. This
    is purely structural -- NO tensor-name matching -- so it works for any
    standard pre-norm transformer regardless of module naming. See the
    TODO(unify-with-production) note above.

    Returns the residual-add NodeProtos in execution order. For a standard block
    they alternate: index 2*L = layer L's attention residual add, 2*L+1 = layer
    L's MLP residual add.
    """
    nodes = {n.name: n for n in model.graph.node}
    seq = {n.name: i for i, n in enumerate(model.graph.node)}
    producers: dict[str, str | None] = collections.defaultdict(lambda: None)
    producers.update({n.output[0]: n.name for n in model.graph.node if n.output})

    def maybe_skip_cast(a: str) -> str:
        if nodes[a].op_type == "Cast":
            inp = producers[nodes[a].input[0]]
            return a if inp is None else inp
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
            for tensor in nodes[cur].input:
                name = producers[tensor]
                if name is not None and name not in visited and seq[name] >= seq[dst]:
                    stack.append(name)
        return False

    def is_residual_add(nodename: str, strict: bool) -> bool:
        if nodes[nodename].op_type != "Add":
            return False
        ins = list(nodes[nodename].input)
        if len(ins) != 2:
            return False
        a, b = (producers[t] for t in ins)
        if a is None or b is None:
            return False
        a, b = maybe_skip_cast(a), maybe_skip_cast(b)
        begin, end = (a, b) if seq[a] < seq[b] else (b, a)
        if strict and nodes[begin].op_type != "Add":
            return False
        return can_visit(end, begin)

    def get_add0(add1: str) -> str:
        # The attention residual add feeding this add's earlier (skip) operand.
        a, b = (producers[t] for t in nodes[add1].input)
        a, b = maybe_skip_cast(a), maybe_skip_cast(b)
        return a if seq[a] < seq[b] else b

    add_names = [name for name in nodes if is_residual_add(name, strict=True)]
    add_names.sort(key=lambda n: seq[n])
    # Layer 0's ATTENTION add is rejected by strict=True: its skip operand is the
    # embedding output, not an Add. That drops the first add and shifts the whole
    # attn/mlp alternation by one. Production get_split_tensors handles this via
    # an odd-count fixup; replicate it -- insert the missing add0 so the list
    # starts cleanly at layer 0's attention add. See TODO(unify-with-production).
    if len(add_names) % 2 == 1:
        add0 = get_add0(add_names[0])
        if add0 not in add_names:
            add_names.insert(0, add0)
    return [nodes[n] for n in add_names]


def _branch_input(
    add_node: onnx.NodeProto,
    producers: dict[str, object],
    seq: dict[str, int],
    initializer_names: set,
) -> str:
    """The projection-output tensor feeding a residual add.

    A residual add is ``residual_skip + branch_out``. The skip is the
    earlier-produced operand (the running residual stream); the branch is the
    later-produced one. But the branch operand is often a SHAPE-OP wrapper of the
    real projection output (e.g. ``Squeeze(conv2d_N)`` / ``Transpose(...)``), and
    that wrapper is not the tensor we want to tap. So after picking the branch
    operand, walk down through shape ops to the underlying compute tensor (the
    Conv/MatMul output that carries the projection's activation encoding).
    """
    a, b = add_node.input[0], add_node.input[1]
    pa, pb = producers.get(a), producers.get(b)
    sa = seq.get(pa.name, -1) if pa is not None else -1
    sb = seq.get(pb.name, -1) if pb is not None else -1
    branch = a if sa > sb else b

    # Skip shape-op wrappers to reach the real projection output.
    cur = branch
    hops = 0
    while hops < 8:
        prod = producers.get(cur)
        if prod is None or prod.op_type not in _SHAPE_OPS:
            break
        nxt = next((i for i in prod.input if i not in initializer_names), None)
        if nxt is None:
            break
        cur = nxt
        hops += 1
    return cur


def discover_taps(
    part: object, bundle: ONNXBundle, base_layer: int | None = None
) -> tuple[list[Tap], list[str]]:
    """Discover per-layer tap tensors purely by graph TOPOLOGY (no name markers).

    All tap points are derived from the residual-add structure the production
    splitter uses -- there is no dependence on module/weight names, so this works
    across transformer families:

      * residual  : each layer's residual-add output (the layer boundary).
      * o_proj    : the attention residual add's branch input (attn output).
      * down      : the MLP residual add's branch input (MLP output).
      * gate_up   : back-trace from the down (MLP-branch) tensor through shape
                    ops to the gated-MLP product (a Mul). Absent on non-gated
                    MLPs -> that column is simply dropped.

    Only tensors that already carry an INT activation encoding are tapped (see
    module docstring); others are reported in ``dropped``. Layer index is
    assigned by residual-add position (no name parsing).

    Returns (taps, dropped).
    """
    model = onnx.load(str(bundle.onnx_graph_path), load_external_data=False)
    encoded = _encoded_activation_names(bundle)
    existing_outputs = {o.name for o in model.graph.output}
    seq = {n.name: i for i, n in enumerate(model.graph.node)}
    producers: dict[str, object] = {o: n for n in model.graph.node for o in n.output}
    initializer_names = {i.name for i in model.graph.initializer}

    adds = _residual_add_boundaries(model)
    # base_layer override lets a single-layer sub-graph (layer_split) label its
    # taps with the correct global layer index directly.
    base = base_layer if base_layer is not None else _first_transformer_layer(part)

    taps: list[Tap] = []
    dropped: list[str] = []

    def maybe_tap(tensor: str, projection: str, layer: int) -> None:
        if tensor in existing_outputs:
            # Already a graph output (e.g. the last layer's residual boundary);
            # captured elsewhere, don't double-tap.
            return
        if tensor not in encoded:
            dropped.append(tensor)
            return
        taps.append(Tap(tensor, _sanitize(tensor), projection, layer))

    # Adds alternate attn-add, mlp-add per layer.
    for i, add in enumerate(adds):
        layer = base + i // 2
        is_attn_add = (i % 2) == 0
        branch = _branch_input(add, producers, seq, initializer_names)  # o_proj / down
        residual_out = add.output[0]

        if is_attn_add:
            maybe_tap(branch, "o_proj", layer)
        else:
            maybe_tap(branch, "down", layer)
            # gate_up: back-trace from the down branch to the gated-MLP product.
            gu = _gate_up_tap(
                branch, producers, initializer_names, encoded, existing_outputs
            )
            if gu is not None:
                maybe_tap(gu, "gate_up", layer)
            # residual boundary = the MLP add output (the layer's output).
            if residual_out not in existing_outputs and residual_out in encoded:
                taps.append(
                    Tap(residual_out, _sanitize(residual_out), "residual", layer)
                )
    return taps, dropped


def _gate_up_tap(
    down_tensor: str,
    producers: dict,
    initializer_names: set,
    encoded: set,
    existing_outputs: set,
) -> str | None:
    """Back-trace from the down (MLP-branch) tensor to the gated-MLP product.

    down_out (Conv/MatMul) <- input <- (shape ops) <- Mul(act(gate), up)

    ``down_tensor`` is the down projection's OUTPUT (a Conv/MatMul output). We
    first step into that op's activation input, then walk back through shape ops
    to the gated-MLP product (a Mul). The Mul output is the single encoded tensor
    summarizing the MLP interior (gate, up, activation). Returns its tensor name,
    or None if the path does not reach an encoded Mul (e.g. a non-gated MLP) --
    in which case the column is simply omitted. Pure topology; no names matched.
    """
    cur: str | None = down_tensor
    hops = 0
    while cur is not None and hops < 10:
        prod = producers.get(cur)
        if prod is None:
            return None
        if prod.op_type == "Mul":
            return cur if (cur in encoded and cur not in existing_outputs) else None
        # Step through the projection op (Conv/MatMul/Gemm) and any shape ops by
        # descending into the non-initializer (activation) input.
        if prod.op_type in _SHAPE_OPS or prod.op_type in {"Conv", "MatMul", "Gemm"}:
            cur = next((i for i in prod.input if i not in initializer_names), None)
            hops += 1
            continue
        return None
    return None


def _prune_to_outputs(model: onnx.ModelProto, keep_outputs: list[str]) -> None:
    """In-place: keep only nodes/initializers that feed ``keep_outputs``.

    Backward reachability from the kept output tensors. Removes everything
    downstream that isn't needed -- critically the LM head, whose on-device
    graph-prepare OOMs at large seq_len (the [vocab x seq] logits + format
    conversions). Graph inputs are preserved (the compiler tolerates unused
    inputs; the injected dataset still matches by name).
    """
    producers = {o: n for n in model.graph.node for o in n.output}
    keep_tensors: set[str] = set()
    keep_nodes: list = []
    seen_nodes: set[int] = set()
    stack = list(keep_outputs)
    while stack:
        t = stack.pop()
        if t in keep_tensors:
            continue
        keep_tensors.add(t)
        prod = producers.get(t)
        if prod is None or id(prod) in seen_nodes:
            continue
        seen_nodes.add(id(prod))
        keep_nodes.append(prod)
        stack.extend(prod.input)

    kept_node_ids = {id(n) for n in keep_nodes}
    surviving = [n for n in model.graph.node if id(n) in kept_node_ids]
    del model.graph.node[:]
    model.graph.node.extend(surviving)

    # Drop initializers no longer referenced by any surviving node.
    used = {i for n in model.graph.node for i in n.input}
    kept_init = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_init)


def build_tapped_bundle(
    bundle: ONNXBundle, taps: list[Tap], part_name: str, prune_to_taps: bool = False
) -> ONNXBundle:
    """Copy the Part bundle and add the tap tensors as graph outputs.

    Uses shape inference to attach proper ValueInfoProtos, then promotes each
    tapped tensor to a graph output. Encodings and external weights are copied
    verbatim (name-keyed, so still valid).

    ``prune_to_taps`` (used for the LM-head-bearing final part): replace the
    graph outputs with ONLY the taps and prune all nodes that don't feed them.
    This deletes the LM head, whose HTP graph-prepare OOMs at large seq_len.
    """
    dst = _scratch_dir() / f"{part_name}.aimet"
    dst.mkdir(parents=True)

    # Copy external weights + encodings verbatim (same file names).
    for src_path in (bundle.onnx_weights_path, bundle.aimet_encodings_path):
        if src_path is not None:
            shutil.copy2(src_path, dst / src_path.name)

    model = onnx.load(str(bundle.onnx_graph_path), load_external_data=False)
    inferred = onnx.shape_inference.infer_shapes(model)
    vi_by_name = {vi.name: vi for vi in inferred.graph.value_info}
    existing_outputs = {o.name for o in model.graph.output}

    tap_names = [t.tensor_name for t in taps]

    if prune_to_taps:
        # Outputs become exactly the taps; prune the LM head (and anything else
        # not feeding a tap) so it never reaches on-device graph-prepare.
        _prune_to_outputs(model, tap_names)
        del model.graph.output[:]
        for tname in tap_names:
            vi = vi_by_name.get(tname) or onnx.helper.make_empty_tensor_value_info(
                tname
            )
            model.graph.output.append(vi)
        # Pruning removed tensors (logits, norm.weight, later-layer KV) that the
        # copied encodings still reference -> load_encodings_to_sim asserts on
        # names absent from the model. Subset the encodings to surviving tensors
        # (encodings are name-keyed), mirroring split_onnx_by_names.
        _subset_encodings_file(dst / bundle.aimet_encodings_name, model)
    else:
        for tap in taps:
            if tap.tensor_name in existing_outputs:
                continue
            vi = vi_by_name.get(
                tap.tensor_name
            ) or onnx.helper.make_empty_tensor_value_info(tap.tensor_name)
            model.graph.output.append(vi)

    # A tensor may not be in both graph.value_info AND graph I/O (ONNX IR rule);
    # the compiler rejects the graph otherwise. Promoting taps to outputs makes
    # their value_info entries illegal, so drop any value_info that now names a
    # graph input or output.
    io_names = {t.name for t in model.graph.input} | {
        t.name for t in model.graph.output
    }
    surviving_vi = [vi for vi in model.graph.value_info if vi.name not in io_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(surviving_vi)

    onnx.save(model, str(dst / bundle.onnx_graph_name))
    return ONNXBundle.from_bundle_path(dst)


def _subset_encodings_file(enc_path: Path, model: onnx.ModelProto) -> None:
    """Drop encoding entries whose tensor no longer exists in ``model``.

    Matches the split utility's membership rule: keep an activation encoding iff
    its name is a node output / graph input / graph output, and a param encoding
    iff its name is an initializer. Handles both the list (>=1.0.0) and dict
    encoding schemas. No-op if the file is missing.
    """
    if not enc_path.exists():
        return
    activation_names = (
        {o for n in model.graph.node for o in n.output}
        | {i.name for i in model.graph.input}
        | {o.name for o in model.graph.output}
    )
    param_names = {i.name for i in model.graph.initializer}

    data = json.loads(enc_path.read_text())

    def keep(section_key: str, allowed: set[str]) -> None:
        sec = data.get(section_key)
        if isinstance(sec, dict):
            data[section_key] = {k: v for k, v in sec.items() if k in allowed}
        elif isinstance(sec, list):
            data[section_key] = [e for e in sec if e.get("name") in allowed]

    keep("activation_encodings", activation_names)
    keep("param_encodings", param_names)
    enc_path.write_text(json.dumps(data))


def golden_taps_from_sim(
    ctx: HarnessContext, part: object, tapped_bundle: ONNXBundle, cap: object
) -> dict[str, list[np.ndarray]]:
    """Re-run the QuantSim on the tapped graph to get golden tap values.

    Mirrors DynamicSplitPartBase._get_quant_sim: build a QuantSim on the
    modified ONNX, apply activation precision, load the (unchanged) encodings,
    then run the Part's captured golden inputs through it. Returns
    {sanitized_output_name: [per-sample arrays]} for every graph output.
    """
    onnx_model = onnx.load(str(tapped_bundle.onnx_graph_path), load_external_data=True)
    onnx_model.ir_version = min(onnx_model.ir_version, 10)

    presplit = ctx.presplit
    hd = getattr(presplit, "host_device", None)
    host_device = hd if isinstance(hd, torch.device) else torch.device("cpu")
    providers = presplit.get_ort_providers(host_device)

    quant_sim = LLMDynamic_AIMETOnnx._build_quantsim(onnx_model, providers)
    LLMDynamic_AIMETOnnx._apply_precision_activations(quant_sim, ctx.precision)
    if tapped_bundle.aimet_encodings_path is not None:
        from qai_hub_models.models.templates.llm.model import load_encodings_to_sim

        load_encodings_to_sim(
            quant_sim, str(tapped_bundle.aimet_encodings_path), strict=False
        )

    out_names = [o.name for o in quant_sim.session.get_outputs()]
    golden: dict[str, list[np.ndarray]] = {n: [] for n in out_names}
    for rec in cap.inputs:
        feeds = {k: np.asarray(v) for k, v in rec.items()}
        outs = quant_sim.session.run(None, feeds)
        for name, arr in zip(out_names, outs, strict=False):
            golden[name].append(np.asarray(arr))
    return golden


@dataclass
class _PreparedPart:
    """Local surgery + golden done; ready to submit device jobs."""

    idx: int
    part_name: str
    part: object
    cap: object
    taps: list[Tap]
    dropped: list[str]
    tapped_bundle: ONNXBundle
    golden: dict[str, list[np.ndarray]]


def _prepare_part(ctx: HarnessContext, part_name: str) -> _PreparedPart:
    """Local-only: discover taps, build the tapped bundle, capture golden taps.

    No device jobs -- this is the cheap, offline half of Phase B for one Part.
    """
    idx = next(
        (i for i, c in enumerate(ctx.captures) if c.part_name == part_name), None
    )
    if idx is None:
        raise ValueError(f"Part {part_name} not found in captures.")
    part = ctx.parts[idx]
    cap = ctx.captures[idx]
    bundle = part._get_onnx_bundle()

    print(f"[Phase B] discovering projection taps in {part_name} ...")
    taps, dropped = discover_taps(part, bundle)
    if not taps:
        raise RuntimeError(
            f"No encoded projection taps found in {part_name}; cannot run Phase B."
        )
    print(f"  {len(taps)} taps ({len(dropped)} dropped for lacking encodings).")

    # The last part carries the LM head (split_lm_head=False for these models).
    # Its logits graph-prepare OOMs on the HTP at large seq_len, so prune the
    # graph to only the layer taps -- this deletes the LM head entirely.
    is_final_part = part.part_id == part.num_splits
    if is_final_part:
        print(
            f"  {part_name} holds the LM head; pruning graph to taps "
            "(dropping logits) to avoid HTP graph-prepare OOM."
        )
    tapped_bundle = build_tapped_bundle(
        bundle, taps, part_name, prune_to_taps=is_final_part
    )
    print(f"[Phase B] re-running QuantSim on tapped {part_name} for golden taps ...")
    golden = golden_taps_from_sim(ctx, part, tapped_bundle, cap)

    return _PreparedPart(
        idx,
        part_name,
        part,
        cap,
        taps,
        dropped,
        tapped_bundle,
        golden,
    )


def run(
    ctx: HarnessContext,
    device: hub.Device,
    part_name: str,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    model_cache_mode: CacheMode = CacheMode.DISABLE,
) -> PhaseBResult:
    """Execute Phase B on ONE Part via the per-layer isolated-input path."""
    return run_layers(ctx, device, part_name, target_runtime, model_cache_mode)


def run_layers(
    ctx: HarnessContext,
    device: hub.Device,
    part_name: str,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    model_cache_mode: CacheMode = CacheMode.DISABLE,
) -> PhaseBResult:
    """Per-layer isolated-input Phase B for one Part.

    Split the part into per-layer sub-graphs, add each layer's projection taps,
    compile all sub-graphs (fanned out), inject the GOLDEN previous-layer
    residual + shared mask/pos + this layer's KV, infer, and merge into the grid.
    Each job outputs only one layer's small tensors -> no 2 GB overflow, and a
    failure localizes to a single layer.
    """
    from device import (
        collect_compile_bundle,
        collect_part_inference,
        submit_compile_bundle,
        submit_feed_inference,
    )
    from layer_split import (
        add_projection_taps,
        build_layer_feed,
        split_part_into_layers,
    )

    # Reuse the full-part golden run: it already holds every residual + tap value.
    prep = _prepare_part(ctx, part_name)
    part, cap, golden = prep.part, prep.cap, prep.golden
    base = _first_transformer_layer(part)

    print(f"[Phase B] splitting {part_name} into per-layer sub-graphs ...")
    subs = split_part_into_layers(part, part._get_onnx_bundle(), base)
    for sub in subs:
        add_projection_taps(sub)

    # Map layer index -> that layer's GOLDEN residual output values (list of
    # per-sample arrays), resolved from the full-part golden run. We key by
    # LAYER INDEX, not tensor name: the per-layer split sub-graphs and the
    # full-part graph are independently produced, so their residual tensor names
    # differ (renumbered) and must not be matched by string. The residual taps in
    # `prep.taps` carry the correct layer index and their golden values.
    def _norm_key(x: str) -> str:
        return x.replace("_updated", "").replace("/", "_").replace(".", "_")

    _golden_by_norm = {_norm_key(k): k for k in golden}
    golden_residual_by_layer: dict[int, list] = {}
    for t in prep.taps:
        if t.projection != "residual":
            continue
        gk = _golden_by_norm.get(_norm_key(t.tensor_name))
        if gk is not None:
            golden_residual_by_layer[t.layer] = golden[gk]

    # Compile every layer sub-graph (fanned out).
    print(f"[Phase B] compiling {len(subs)} layer sub-graphs (parallel) ...")
    input_specs = {s.layer: layer_input_spec(ctx, part, s) for s in subs}
    cjobs = {
        s.layer: submit_compile_bundle(
            s.bundle.bundle_path.as_posix(),
            input_specs[s.layer],
            device,
            ctx.presplit,
            ctx.precision,
            part.part_id - 1,
            part.num_splits,
            target_runtime,
            f"debug_{part_name}_L{s.layer}",
            model_cache_mode=model_cache_mode,
        )
        for s in subs
    }
    compiled = {
        s.layer: collect_compile_bundle(cjobs[s.layer], f"{part_name}_L{s.layer}")
        for s in subs
    }

    # Build golden feeds + submit inference (fanned out).
    print(f"[Phase B] submitting {len(subs)} layer inferences (parallel) ...")
    jobs = {}
    for s in subs:
        feed = build_layer_feed(
            s, compiled[s.layer].input_order, cap, golden_residual_by_layer
        )
        jobs[s.layer] = submit_feed_inference(
            compiled[s.layer], feed, device, f"debug_{part_name}_L{s.layer}"
        )

    # Collect + merge into the grid. Both golden (QuantSim session outputs) and
    # device_out (on-device output names) can carry '_updated' suffixes and
    # '/'.'-> '_' sanitization, so match tap tensors by NORMALIZED name rather
    # than exact string (same gap that hit the input feed).
    def _norm(x: str) -> str:
        return x.replace("_updated", "").replace("/", "_").replace(".", "_")

    golden_by_norm = {_norm(k): k for k in golden}

    result = PhaseBResult(part_name=part_name, taps=[], dropped_unencoded=prep.dropped)
    unmatched_reported = False
    for s in subs:
        device_out = collect_part_inference(jobs[s.layer])
        dev_by_norm = {_norm(k): k for k in device_out}
        result.taps.extend(s.taps)
        for tap in s.taps:
            nkey = _norm(tap.tensor_name)
            dkey = dev_by_norm.get(nkey)
            gkey = golden_by_norm.get(nkey)
            if dkey is None or gkey is None:
                if not unmatched_reported:
                    print(
                        f"  [warn] tap {tap.projection}@L{s.layer} '{tap.tensor_name}' "
                        f"unmatched (device={dkey}, golden={gkey}). "
                        f"device keys sample={sorted(device_out)[:6]}"
                    )
                    unmatched_reported = True
                continue
            nn = min(len(golden[gkey]), len(device_out[dkey]))
            if nn == 0:
                continue
            sq, cos, mx = [], [], []
            for i in range(nn):
                m = TensorMetric.compare(
                    tap.tensor_name,
                    golden[gkey][i],
                    np.asarray(device_out[dkey][i]),
                )
                sq.append(m.sqnr_db)
                cos.append(m.cosine)
                mx.append(m.max_abs_diff)
            result.grid.setdefault(s.layer, {})[tap.projection] = TensorMetric(
                tap.tensor_name,
                float(np.mean(sq)),
                float(np.mean(cos)),
                float(np.max(mx)),
            )
    return result


def layer_input_spec(ctx: HarnessContext, part: object, sub: object) -> dict:
    """Input spec for a per-layer sub-graph: reuse part-spec shapes by name;
    the previous-residual input is (1, seq, hidden) float32.
    """
    from qai_hub_models.utils.input_spec import TensorSpec

    full = ctx.presplit.get_input_spec(
        llm_config=ctx.presplit.llm_config.to_dict(),
        sequence_length=ctx.sequence_length,
        context_length=ctx.context_length,
        llm_io_type=ctx.presplit.llm_io_type,
    )
    spec = {}
    for name in sub.input_names:
        if name in full:
            spec[name] = full[name]
        else:
            spec[name] = TensorSpec(
                shape=(1, ctx.sequence_length, ctx.presplit.llm_config.hidden_size),
                dtype="float32",
            )
    return spec


def run_all(
    ctx: HarnessContext,
    device: hub.Device,
    target_runtime: TargetRuntime = TargetRuntime.GENIE,
    skip_embedding: bool = True,
    model_cache_mode: CacheMode = CacheMode.DISABLE,
) -> list[PhaseBResult]:
    """Execute Phase B on ALL transformer Parts, fanning out device jobs.

    Local surgery + golden capture run per Part (cheap, offline). Then every
    Part's compile is submitted before any is awaited, and likewise inference,
    so the device work overlaps across Parts -- for a small model (few Parts)
    this is close to the wall-clock of a single Part.
    """
    part_names = [
        cap.part_name
        for idx, cap in enumerate(ctx.captures)
        if not (skip_embedding and ctx.layers_per_part.get(idx, 0) == 0)
    ]
    # Per-layer path: each part is split into per-layer sub-graphs internally.
    return [
        run_layers(ctx, device, name, target_runtime, model_cache_mode)
        for name in part_names
    ]


# ------------------------------- reporting -------------------------------


def report_dict(result: PhaseBResult, meta: dict) -> dict:
    return {
        "phase": "B",
        "meta": meta,
        "part": result.part_name,
        "dropped_unencoded": result.dropped_unencoded,
        "grid": {
            str(layer): {
                proj: {
                    "sqnr_db": m.sqnr_db,
                    "cosine": m.cosine,
                    "max_abs_diff": m.max_abs_diff,
                }
                for proj, m in projs.items()
            }
            for layer, projs in result.grid.items()
        },
    }


def _crater_threshold(result: PhaseBResult) -> float:
    """DB below which a cell is a 'crater', from the grid's own healthy median.

    Uses the grid median minus a fixed margin so we flag the cliff relative to
    the Part's own healthy taps rather than an absolute number.
    """
    vals = [
        m.sqnr_db
        for projs in result.grid.values()
        for m in projs.values()
        if not np.isnan(m.sqnr_db)
    ]
    if not vals:
        return float("-inf")
    return float(np.median(vals)) - 12.0


def _phase_b_verdict(result: PhaseBResult, thresh: float) -> str:
    """One-line synthesis: which projection craters, from which layer."""
    per_proj_min_layer: dict[str, int] = {}
    for layer in sorted(result.grid):
        for proj, m in result.grid[layer].items():
            if m.sqnr_db < thresh and proj not in per_proj_min_layer:
                per_proj_min_layer[proj] = layer
    if not per_proj_min_layer:
        return "No projection craters; divergence is not at a projection boundary."
    bits = [f"{proj} (from layer {ly})" for proj, ly in per_proj_min_layer.items()]
    return "Craters: " + ", ".join(bits) + " -- inspect that interval's ops."


def print_report(result: PhaseBResult, meta: dict, json_path: Path | None) -> None:
    print("\n" + "=" * 78)
    print(f"PHASE B -- per-layer projection taps in {result.part_name}")
    print("=" * 78)
    header = f"{'layer':>6}" + "".join(f"{p:>12}" for p in _INTERVAL_ORDER)
    print(header)
    thresh = _crater_threshold(result)
    for layer in sorted(result.grid):
        row = f"{layer:>6}"
        for proj in _INTERVAL_ORDER:
            m = result.grid[layer].get(proj)
            if m is None:
                row += f"{'-':>12}"
            else:
                # Trailing '*' marks a crater (>= 12 dB below grid median).
                cell = f"{m.sqnr_db:.1f}{'*' if m.sqnr_db < thresh else ''}"
                row += f"{cell:>12}"
        print(row)

    print("-" * 78)
    print(_phase_b_verdict(result, thresh))
    print("  (* = >= 12 dB below the grid median)")
    if result.dropped_unencoded:
        print(f"Dropped {len(result.dropped_unencoded)} un-encoded candidate tensors.")
    print("=" * 78 + "\n")

    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report_dict(result, meta), indent=2))
        print(f"Phase B metrics written to {json_path}\n")
