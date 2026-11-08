# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared scaffolding for the sim-vs-device debugging phases.

Everything that builds the model and is common to Phase A (per-Part
localization) and Phase B (projection-tap surgery) lives here: model-id
resolution, the decoder-layer-per-Part layout, building the QuantSim split
model, and capturing golden per-Part tensors on the dataset windows. Each phase
imports this and adds only its own logic + report.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass

import torch
from dataset import build_source
from golden import PartCapture, capture_golden

from qai_hub_models import Precision
from qai_hub_models.models.templates.llm.split_onnx_utils.utils import (
    get_split_tensors,
)


def resolve_model_module(model_id: str) -> object:
    """Import a model package by id, e.g. 'qwen3_1_7b'."""
    return importlib.import_module(f"qai_hub_models.models.{model_id}")


@functools.cache
def _layers_in_graph(onnx_graph_path: str) -> int:
    """Number of decoder layers in one split ONNX graph (residual boundaries)."""
    return len(get_split_tensors(onnx_graph_path, include_first_input=False))


def layers_per_part(presplit: object) -> dict[int, int]:
    """Map 0-based part index -> number of decoder layers it holds.

    Counted from each part's actual split ONNX graph (the same residual-add
    topology split_onnx cuts on), NOT re-derived arithmetically -- so it follows
    whatever drove the real split (num_layers_per_split, splitting_points,
    split_embedding / split_lm_head). Embedding / LM-head-only parts count 0.
    """
    return {
        i: _layers_in_graph(
            str(presplit.convert_to_onnx_and_split(part_id=i + 1).onnx_graph_path)
        )
        for i in range(presplit.num_splits)
    }


def build_split_model(
    model_module: object,
    checkpoint: str,
    precision: Precision,
    seq_len: int,
    ctx_len: int,
) -> object:
    """Build the shared QuantSim split-model wrapper (IS-A quantizable presplit).

    The wrapper owns the Part instances (all sharing one presplit, so the ONNX
    is exported/split once). Sequence/context lengths are dynamic-path
    attributes set after construction, as the export pipeline does.
    """
    wrapper_cls = getattr(model_module, "QuantizedSplitModelWrapper", None)
    if wrapper_cls is None:
        raise RuntimeError(
            f"{model_module.__name__} has no QuantizedSplitModelWrapper; "
            "this harness targets the PreSplit/Part split LLMs."
        )
    split_model = wrapper_cls.from_pretrained(
        checkpoint=checkpoint,
        precision=precision,
        _skip_quantsim_creation=False,
    )
    split_model.sequence_length = seq_len
    split_model.context_length = ctx_len
    split_model.eval()
    return split_model


@dataclass
class HarnessContext:
    """Everything the phases need after model build + golden capture."""

    model_id: str
    checkpoint: str
    precision: Precision
    sequence_length: int
    context_length: int
    dataset: str
    num_windows: int
    split_model: object  # QuantizedSplitModelWrapper (also the presplit)
    parts: list  # instantiated Part objects
    captures: list[PartCapture]  # golden per-Part I/O
    layers_per_part: dict[int, int]
    num_splits: int

    @property
    def presplit(self) -> object:
        return self.split_model

    def meta(self) -> dict:
        return {
            "model_id": self.model_id,
            "checkpoint": self.checkpoint,
            "precision": str(self.precision),
            "dataset": self.dataset,
            "num_windows": self.num_windows,
            "sequence_length": self.sequence_length,
            "context_length": self.context_length,
        }


def build_context(
    model_id: str,
    checkpoint: str,
    precision: Precision,
    seq_len: int,
    ctx_len: int,
    dataset: str,
    num_windows: int,
) -> HarnessContext:
    """Build the split model, load dataset windows, capture golden tensors."""
    module = resolve_model_module(model_id)

    print(f"Building QuantSim split model for {model_id} @ {precision} ...")
    split_model = build_split_model(module, checkpoint, precision, seq_len, ctx_len)
    split_model._ensure_parts()
    parts = split_model._parts

    num_splits = module.NUM_SPLITS
    lpp = layers_per_part(split_model)
    print(f"  decoder layers per part: {[lpp[i] for i in range(num_splits)]}")
    if sum(lpp.values()) != module.NUM_LAYERS:
        print(
            f"  WARNING: split graphs hold {sum(lpp.values())} decoder layers, "
            f"but {model_id}.NUM_LAYERS = {module.NUM_LAYERS}."
        )

    print(f"Loading {dataset} prefill windows ...")
    source = build_source(dataset, split_model.tokenizer, ctx_len, seq_len)
    windows = source.windows(num_windows)
    print(f"  {len(windows)} windows.")

    print("Capturing golden per-Part tensors from QuantSim (local) ...")
    with torch.no_grad():
        captures = capture_golden(
            split_model,
            parts,
            windows,
            sequence_length=seq_len,
            context_length=ctx_len,
            layers_per_part=lpp,
        )

    return HarnessContext(
        model_id=model_id,
        checkpoint=checkpoint,
        precision=precision,
        sequence_length=seq_len,
        context_length=ctx_len,
        dataset=dataset,
        num_windows=len(windows),
        split_model=split_model,
        parts=parts,
        captures=captures,
        layers_per_part=lpp,
        num_splits=num_splits,
    )
