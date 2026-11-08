# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest
import torch

from qai_hub_models.models.qwen3_4b import Model
from qai_hub_models.models.qwen3_4b.demo import qwen3_4b_chat_demo
from qai_hub_models.models.qwen3_4b.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_4B_PreSplit,
    Qwen3_4B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.model import DEFAULT_CONTEXT_LENGTH
from qai_hub_models.utils.checkpoint import CheckpointSpec

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


# qwen3_1_7b is the qwen nightly canary (its SpinQuant R1 path surfaces
# quantization regressions); this model runs the full matrix weekly only.
@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext", 13.49, 0),
        ("DEFAULT_W4A16", "mmlu", 0.668, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 12.41, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.72, 0),
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
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
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
        quantized_presplit_cls=Qwen3_4B_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_4B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
    )


# Full W4A16 quantize + demo runs on qwen3_1_7b as the nightly canary; running it
# here too was expensive and flaky, so it's dropped.
# Weekly-only (no @pytest.mark.nightly); nightly demo coverage is on qwen3_1_7b.
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    qwen3_4b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
