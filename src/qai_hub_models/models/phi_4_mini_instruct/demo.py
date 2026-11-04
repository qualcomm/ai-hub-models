# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models.phi_4_mini_instruct.model import (
    HF_REPO_NAME,
    MODEL_ID,
    Phi4Mini_PreSplit,
    QuantizedSplitModelWrapper,
)
from qai_hub_models.models.templates.llm.demo import llm_chat_demo
from qai_hub_models.models.templates.llm.model import LLM_QNN, LLM_AIMETOnnx, LLMBase
from qai_hub_models.models.templates.phi.model import END_TOKENS
from qai_hub_models.utils.checkpoint import CheckpointSpec

HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"


def phi4_mini_chat_demo(
    model_cls: type[LLM_AIMETOnnx] = QuantizedSplitModelWrapper,
    fp_model_cls: type[LLMBase] = Phi4Mini_PreSplit,
    qnn_model_cls: type[LLM_QNN] = LLM_QNN,
    model_id: str = MODEL_ID,
    end_tokens: set = END_TOKENS,
    hf_repo_name: str = HF_REPO_NAME,
    hf_repo_url: str = HF_REPO_URL,
    default_prompt: str | None = None,
    test_checkpoint: CheckpointSpec | None = None,
) -> None:
    llm_chat_demo(
        model_cls=model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=qnn_model_cls,
        model_id=model_id,
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        default_prompt=default_prompt,
        test_checkpoint=test_checkpoint,
    )


if __name__ == "__main__":
    phi4_mini_chat_demo()
