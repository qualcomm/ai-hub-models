# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.qwen3_5_4b.model import (
    HF_REPO_NAME,
    MODEL_ID,
    Qwen3_5_4B_PreSplit,
    Qwen3_5_4B_QuantizablePreSplit,
)
from qai_hub_models.models.templates.llm.demo import llm_chat_demo
from qai_hub_models.models.templates.llm.model import LLM_QNN
from qai_hub_models.models.templates.qwen3_5.model import END_TOKENS
from qai_hub_models.utils.checkpoint import CheckpointSpec

HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"


def qwen3_5_4b_chat_demo(
    default_prompt: str | None = None,
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    llm_chat_demo(
        model_cls=Qwen3_5_4B_QuantizablePreSplit,
        fp_model_cls=Qwen3_5_4B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        model_id=MODEL_ID,
        end_tokens=END_TOKENS,
        hf_repo_name=HF_REPO_NAME,
        hf_repo_url=HF_REPO_URL,
        default_prompt=default_prompt,
        test_checkpoint=test_checkpoint,
    )


if __name__ == "__main__":
    qwen3_5_4b_chat_demo()
