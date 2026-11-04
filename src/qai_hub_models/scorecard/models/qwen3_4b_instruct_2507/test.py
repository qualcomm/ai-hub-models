# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from qai_hub_models.models.qwen3_4b_instruct_2507 import Model
from qai_hub_models.models.qwen3_4b_instruct_2507.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_4B_Instruct_2507_Part1_Of_4,
    Qwen3_4B_Instruct_2507_Part4_Of_4,
    Qwen3_4B_Instruct_2507_PartBase,
    Qwen3_4B_Instruct_2507_PreSplit,
    Qwen3_4B_Instruct_2507_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


@pytest.mark.evaluate
@pytest.mark.parametrize(
    "part_cls",
    [Qwen3_4B_Instruct_2507_Part1_Of_4, Qwen3_4B_Instruct_2507_Part4_Of_4],
)
def test_part_quantsim_loads_encodings(
    part_cls: type[Qwen3_4B_Instruct_2507_PartBase],
) -> None:
    """Building a Part's QuantSim must load the migrated encodings.

    Qwen3-4B-Instruct-2507 ties lm_head.weight to the embedding table, so the
    dynamo graph names the single tied initializer ``model.lm_head.weight`` and
    feeds it to both the embedding ``Gather`` (Part1) and the lm_head ``MatMul``
    (Part4). The migrated per-channel lm_head encoding is loadable in Part4 but
    must be relaxed/stripped for the Gather input in Part1 (which has no
    ``tensor_quantizer_params``). This exercises both ends; Part1 regression-
    tests the tied-embedding fix in
    ``Qwen3_4B_Instruct_2507_PartBase._get_quant_sim``.
    """
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    part = part_cls.from_pretrained(
        checkpoint="DEFAULT_W4A16",
        _skip_quantsim_creation=True,
        sequence_lengths=[DEFAULT_SEQUENCE_LENGTH],
        context_lengths=[DEFAULT_CONTEXT_LENGTH],
    )
    # Must not raise (regression: per-channel load on a Gather-fed tied weight).
    quant_sim = part._get_quant_sim()
    assert quant_sim is not None


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext", 9.92, 0),
        ("DEFAULT_W4A16", "mmlu", 0.689, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 9.39, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.74, 0),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    dataset_cls = next(
        d
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_4B_Instruct_2507_PreSplit.release()
    Qwen3_4B_Instruct_2507_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=QuantizedSplitModelWrapper,
        fp_split_cls=FPSplitModelWrapper,
        quantized_presplit_cls=Qwen3_4B_Instruct_2507_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_4B_Instruct_2507_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
    )
