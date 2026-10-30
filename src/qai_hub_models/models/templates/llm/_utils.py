# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Generator

import onnx

try:
    from aimet_onnx.common.defs import QuantizationDataType
    from aimet_onnx.qc_quantize_op import (
        GroupedBlockQuantizeDequantize,
        QcQuantizeOp,
    )
    from aimet_onnx.quantsim import QuantizationSimModel as QuantSimOnnx
except (ImportError, ModuleNotFoundError):
    QuantizationDataType = GroupedBlockQuantizeDequantize = QcQuantizeOp = (
        QuantSimOnnx
    ) = None

logger = logging.getLogger(__name__)

# Ops that rearrange or reinterpret data without changing the values flowing
# through them, so both sides of one must share a scale with the KV buffer.
_VALUE_PRESERVING_OPS = frozenset(
    {
        "Cast",
        "Concat",
        "Identity",
        "Reshape",
        "Slice",
        "Split",
        "Squeeze",
        "Transpose",
        "Unsqueeze",
    }
)

# ScatterElements inputs carrying cache values: the buffer written into (0) and
# the values written (2). Input 1 holds indices, which keep their own encoding.
_SCATTER_VALUE_INPUTS = (0, 2)


def _kv_write_path_tensors(sim: QuantSimOnnx, seeds: list[str]) -> set[str]:
    """Names of every tensor that must share the KV buffer's encoding.

    Floods outward from the KV cache graph I/O in both directions, crossing only
    value-preserving ops and the ScatterElements that write into the buffer. The
    flood stops at the first computing op but keeps its output, which is the
    value being written into the cache (a k/v_proj Conv or a RoPE Add/Sub).
    """
    cg = sim.connected_graph
    reached: set[str] = set()
    stack = list(seeds)
    while stack:
        name = stack.pop()
        if name in reached:
            continue
        reached.add(name)
        product = cg.get_product(name)
        if product is None:
            continue

        producer = product.producer
        if producer is not None:
            if producer.type in _VALUE_PRESERVING_OPS:
                stack.extend(t.name for t in producer.inputs)
            elif producer.type == "ScatterElements":
                stack.extend(
                    producer.inputs[i].name
                    for i in _SCATTER_VALUE_INPUTS
                    if i < len(producer.inputs)
                )

        for consumer in product.consumers:
            if consumer.type in _VALUE_PRESERVING_OPS or (
                consumer.type == "ScatterElements"
                and any(
                    consumer.inputs[i].name == name
                    for i in _SCATTER_VALUE_INPUTS
                    if i < len(consumer.inputs)
                )
            ):
                stack.extend(t.name for t in consumer.outputs)

    return reached


def _enable_scatter_output_quantizers(sim: QuantSimOnnx) -> None:
    """Enable ScatterElements output quantizers as 8-bit symmetric.

    AIMET's default config leaves these disabled.  They must be enabled before
    encoding tying so that the traversal can tie them to the KV I/O encoding
    (required for OutputSlice fusion on HTP).
    """
    enabled_count = 0
    for op in sim.connected_graph.ordered_ops:
        if op.type != "ScatterElements":
            continue
        for output in op.outputs:
            qtzr = sim.qc_quantize_op_dict.get(output.name)
            if qtzr and not qtzr.enabled:
                qtzr.enabled = True
                qtzr.set_bitwidth(8)
                qtzr.use_symmetric_encodings = True
                enabled_count += 1
    logger.info(
        "[DEBUG] _enable_scatter_output_quantizers: enabled %d ScatterElements output quantizers",
        enabled_count,
    )


def _tie_quantizers_for_kv_cache(
    quantsim_model: QuantSimOnnx, kv_io_map: dict[str, str]
) -> None:
    """Tie the KV cache input quantizer to its matching output quantizer.

    For native KV, extend the tie across the whole write path: HTP fuses
    ScatterElements into OutputSlice.K/.V and drops the requantize steps along
    it, so the k/v_proj outputs, RoPE outputs, per-head scatters and KV I/O must
    all share one scale. A mismatch is not a small accuracy loss: the fused op
    writes bits that attention reads back at a different scale, and the model
    emits garbage. Legacy Concat-based caches are not fused and keep the narrow
    input-to-output tie they were calibrated with.
    """
    is_native = any("nativekvcache" in k for k in kv_io_map)
    quantizer_mapping: dict[str, QcQuantizeOp] = {}

    for input_name, output_name in kv_io_map.items():
        quantizer = quantsim_model._get_enabled_quantizer(output_name)  # pylint: disable=protected-access
        if not quantizer:
            logger.debug("No enabled quantizer for KV output %s", output_name)
            continue

        quantizer_mapping[input_name] = quantizer
        if not is_native:
            continue

        for tensor_name in _kv_write_path_tensors(
            quantsim_model, [input_name, output_name]
        ):
            qtzr_name = quantsim_model._get_enabled_quantizer_name(tensor_name)  # pylint: disable=protected-access
            if qtzr_name:
                quantizer_mapping[qtzr_name] = quantizer

    logger.info("Tying %d KV write-path quantizers", len(quantizer_mapping))
    quantsim_model.set_quantizers(quantizer_mapping)


def _set_tensors_to_output_8b_sym(
    quantsim_model: QuantSimOnnx, out_tensors: list[str]
) -> None:
    for out_tensor in out_tensors:
        _set_tensor_to_8_bit_symmetric(quantsim_model, out_tensor)


def _is_kv_tensor_name(name: str) -> bool:
    return (
        "past_key" in name
        or "past_value" in name
        or "nativekvcache__key" in name
        or "nativekvcache__value" in name
    )


def _get_kv_io_map(quantsim_model: QuantSimOnnx) -> dict[str, str]:
    inputs = [
        t.name for t in quantsim_model.model.graph().input if _is_kv_tensor_name(t.name)
    ]
    outputs = [
        t.name.replace("_updated", "")
        for t in quantsim_model.model.graph().output
        if _is_kv_tensor_name(t.name)
    ]
    return dict(zip(inputs, outputs, strict=False))


def _set_4bit_weights_to_lpbq(quantsim_model: QuantSimOnnx) -> None:
    # This is largely a copy-paste of
    # set_grouped_blockwise_quantization_for_weights, except adds an op
    # selection criterion based on all ops that already have the target
    # bitwidth. Can be simplified once that function accepts a function
    # argument.
    block_size = 64
    decompressed_bw = 8
    strict = False
    bitwidth = 4
    for op in quantsim_model.connected_graph.ordered_ops:
        _, _, param_quantizers = quantsim_model.get_op_quantizers(op)

        weight_quantizer: QcQuantizeOp = param_quantizers.get("weight")
        bias_quantizer: QcQuantizeOp = param_quantizers.get("bias")

        if not weight_quantizer:
            continue

        if weight_quantizer.bitwidth != bitwidth:
            continue

        try:
            grouped_quantizer = GroupedBlockQuantizeDequantize(
                weight_quantizer.quant_info,
                bitwidth,
                decompressed_bw,
                block_size,
                weight_quantizer.quant_scheme,
                weight_quantizer.op_mode,
                weight_quantizer.tensor_quantizer_params,
            )
        except ValueError:
            if strict:
                raise
        else:
            if bias_quantizer:
                bias_quantizer.enable_per_channel_quantization()
                bias_quantizer.use_symmetric_encodings = True
                bias_quantizer.data_type = QuantizationDataType.int

            for name, quantizer in quantsim_model.qc_quantize_op_dict.items():
                if quantizer is weight_quantizer:
                    quantsim_model.qc_quantize_op_dict[name] = grouped_quantizer


def _set_tensor_to_8_bit_symmetric(
    quantsim_model: QuantSimOnnx, tensor_name: str
) -> None:
    quantizer = quantsim_model._get_enabled_quantizer(tensor_name)  # pylint: disable=protected-access
    if quantizer:
        quantizer.set_bitwidth(8)
        quantizer.use_symmetric_encodings = True


def _set_lm_head_to_8b(quantsim_model: QuantSimOnnx) -> None:
    for weight in _get_lm_head_weights(quantsim_model.model.model):
        quantizer = quantsim_model.qc_quantize_op_dict[weight.name]
        quantizer.set_bitwidth(8)
        quantizer.quant_info.blockSize = 0
        quantizer.quant_info.blockAxis = -1
        quantizer.enable_per_channel_quantization()


def _get_lm_head_weights(
    model: onnx.ModelProto,
) -> Generator[onnx.TensorProto, None, None]:
    vocab_size = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    for weight in model.graph.initializer:
        if any(dim == vocab_size for dim in weight.dims):
            for node in model.graph.node:
                if node.op_type in ("Gemm", "MatMul", "Conv") and node.input[1] in {
                    weight.name,
                    weight.name + "_updated",
                    weight.name + "_qdq",
                }:
                    yield weight


def _get_quantizer_no_split_slice(
    quantsim_model: QuantSimOnnx, tensor_name: str
) -> QcQuantizeOp | None:
    """
    Returns closest enabled quantizer to tensor traversing upwards only
    through invariant ops and no Split/Slice.
    """
    from aimet_onnx.common.onnx._utils import _is_grid_preserving_op

    quantizer = quantsim_model.qc_quantize_op_dict.get(tensor_name, None)
    if quantizer and quantizer.enabled:
        return quantizer

    prod_dict = quantsim_model.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)

    if product is None:
        if tensor_name.endswith(("_updated", "_qdq")):
            raise KeyError(
                f"Could not find quantizer for tensor {tensor_name}. "
                "Input tensor_name must be the name of a tensor in the "
                "original (unquantized) graph"
            )
        raise KeyError(
            f"Could not find quantizer for tensor {tensor_name}. "
            "Tensor name does not exist in the graph"
        )

    producer = product.producer

    if producer is None:
        return None

    if not (_is_grid_preserving_op(producer.type)) or producer.type in {
        "Slice",
        "Split",
        "SplitToSequence",
    }:
        return None

    if len(producer.inputs) == 0:
        return None

    upstream_tensor = producer.inputs[0]
    return _get_quantizer_no_split_slice(quantsim_model, upstream_tensor.name)


_PROPAGATE_8B_OPS = frozenset({"Concat", "Transpose", "Reshape", "Slice", "Div"})


def _propagate_8b_upstream(
    quantsim_model: QuantSimOnnx, tensor_name: str, visited: set[str] | None = None
) -> None:
    """
    Recursively propagate 8-bit quantization upstream through Concat,
    Transpose, Reshape, Slice, and Div ops. For each such op, set its data
    inputs' quantizers to 8-bit and continue recursing.

    Div is also included to accommodate when we swap the order of MatMul and
    Div. We do this for Qwen3. When we do this, the Div is upstream from the
    MatMul and need to be included if we want to get 16x8 MatMuls.

    Concat: all inputs are data inputs.
    Transpose, Reshape, Slice: only input[0] is data.
    """
    if visited is None:
        visited = set()
    if tensor_name in visited:
        return
    visited.add(tensor_name)

    prod_dict = quantsim_model.connected_graph.get_all_products()
    product = prod_dict.get(tensor_name, None)
    if product is None:
        return

    producer = product.producer
    if producer is None:
        return

    if producer.type not in _PROPAGATE_8B_OPS:
        return

    if producer.type == "Concat":
        data_inputs = list(producer.inputs)
    else:
        # Transpose, Reshape, Slice: only first input is data
        data_inputs = [producer.inputs[0]] if producer.inputs else []

    for inp in data_inputs:
        quantizer = quantsim_model.qc_quantize_op_dict.get(inp.name, None)
        if quantizer is not None and quantizer.enabled and quantizer.bitwidth > 8:
            quantizer.set_bitwidth(8)
            quantizer.use_symmetric_encodings = True
        _propagate_8b_upstream(quantsim_model, inp.name, visited)


def _reads_kv_cache(sim: QuantSimOnnx, tensor_name: str, kv_io: set[str]) -> bool:
    """Whether tensor_name is fed from the KV cache through data-movement ops only."""
    cg = sim.connected_graph
    stack = [tensor_name]
    seen: set[str] = set()
    while stack:
        name = stack.pop()
        if name in kv_io:
            return True
        if name in seen:
            continue
        seen.add(name)
        product = cg.get_product(name)
        producer = product.producer if product is not None else None
        if producer is None or not producer.inputs:
            continue
        if producer.type == "Concat":
            stack.extend(t.name for t in producer.inputs)
        elif producer.type == "ScatterElements":
            stack.extend(
                producer.inputs[i].name
                for i in _SCATTER_VALUE_INPUTS
                if i < len(producer.inputs)
            )
        elif producer.type in _VALUE_PRESERVING_OPS | _PROPAGATE_8B_OPS:
            stack.append(producer.inputs[0].name)
    return False


def _set_matmul_second_input_to_8b(
    quantsim_model: QuantSimOnnx, kv_io_map: dict[str, str] | None = None
) -> None:
    """Make MatMul second inputs 8-bit so HTP runs them as 16x8.

    With kv_io_map, only MatMuls whose second input is read from the KV cache.
    """
    cg = quantsim_model.connected_graph
    # Not _kv_write_path_tensors: that flood crosses Slice/Reshape shape inputs,
    # which constant sharing links to nearly every tensor in the graph.
    kv_io = {*kv_io_map.keys(), *kv_io_map.values()} if kv_io_map is not None else None

    for op in reversed(cg.ordered_ops):
        if op.type != "MatMul":
            continue
        if kv_io is not None and not _reads_kv_cache(
            quantsim_model, op.inputs[1].name, kv_io
        ):
            continue

        upper_quantizer = quantsim_model._get_enabled_quantizer(op.inputs[1].name)  # pylint: disable=protected-access

        enabled_quantizer = _get_quantizer_no_split_slice(
            quantsim_model, op.inputs[1].name
        )

        if enabled_quantizer and enabled_quantizer.bitwidth <= 8:
            pass
        elif enabled_quantizer:
            enabled_quantizer.set_bitwidth(8)
            enabled_quantizer.use_symmetric_encodings = True
        elif upper_quantizer:
            if op.inputs[1].name in quantsim_model.qc_quantize_op_dict:
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True
            else:
                quantsim_model._insert_quantizer(op.inputs[1].name, is_param=False)  # pylint: disable=protected-access
                quantsim_model._rebuild_session()  # pylint: disable=protected-access
                quantizer = quantsim_model.qc_quantize_op_dict[op.inputs[1].name]
                quantizer.enabled = True
                quantizer.set_bitwidth(8)
                quantizer.use_symmetric_encodings = True
        else:
            continue

        # Propagate 8-bit upstream through select op types to ensure MatMul
        # 8-bit input is respected during conversion.
        _propagate_8b_upstream(quantsim_model, op.inputs[1].name)


def _apply_int8_kv_cache_tying_and_lm_head(
    sim: QuantSimOnnx,
    kv_io_map: dict[str, str],
    use_16x8_matmuls: bool = True,
) -> QuantSimOnnx:
    is_native = any("nativekvcache" in k for k in kv_io_map)

    if is_native:
        # AIMET's default config leaves ScatterElements output quantizers off;
        # they have to exist before the write path can be tied to them.
        _enable_scatter_output_quantizers(sim)
    else:
        sim._tie_quantizers_for_op_types(["Concat"])  # pylint: disable=protected-access
    sim._rebuild_session()  # pylint: disable=protected-access

    kv_io_list = list(kv_io_map.keys()) + list(kv_io_map.values())
    _set_tensors_to_output_8b_sym(sim, kv_io_list)
    _set_lm_head_to_8b(sim)

    _tie_quantizers_for_kv_cache(sim, kv_io_map)
    sim._rebuild_session()  # pylint: disable=protected-access

    if use_16x8_matmuls:
        _set_matmul_second_input_to_8b(sim, kv_io_map)

    return sim
