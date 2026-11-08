# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import pytest
from transformers import AutoConfig, AutoTokenizer

from qai_hub_models.models.phi_4_mini_instruct.model import (
    HF_REPO_NAME,
    MODEL_ID,
    NUM_SPLITS,
)
from qai_hub_models.models.templates.llm.llm_helpers import (
    DEFAULT_MAX_NUM_TOKENS,
    create_genie_config,
)
from qai_hub_models.models.templates.phi.model import (
    END_TOKENS,
    PhiPreSplitCollectionBase,
)


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_NAME)

    # Resolve the <|end|> token ID (the sole EOS stop token for Phi-4-mini).
    # <|endoftext|> (id=199999) is pre-filled in prompt files via
    # genie_generation_prompt_suffix and is NOT in the stop-token list.
    end_token_encoded = tokenizer.encode("<|end|>", add_special_tokens=False)
    assert len(end_token_encoded) == 1, (
        f"Expected '<|end|>' to encode to a single token for Phi-4-mini, "
        f"got {end_token_encoded}"
    )
    end_token_id = int(end_token_encoded[0])

    # Verify END_TOKENS contains only <|end|>.
    assert {"<|end|>"} == END_TOKENS, (
        f"END_TOKENS should contain only '<|end|>', got {END_TOKENS}"
    )

    model_list = [
        f"{MODEL_ID}_part_{i}_of_{NUM_SPLITS}.bin" for i in range(1, NUM_SPLITS + 1)
    ]
    # create_genie_config is called with the raw llm_config (bos = <|endoftext|>
    # id=199999 from HF).  write_supplementary_files then patches the written
    # genie_config.json:
    #   bos-token → PAD_TOKEN_ID (200029)
    #   eos-token → end_token_id (200020, <|end|>)
    # Simulate that same two-step here.
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    actual_config["dialog"]["context"]["bos-token"] = (
        PhiPreSplitCollectionBase.PAD_TOKEN_ID
    )
    actual_config["dialog"]["context"]["eos-token"] = end_token_id
    rope_scaling = llm_config.rope_scaling
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "max-num-tokens": DEFAULT_MAX_NUM_TOKENS,
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 200064,
                "bos-token": 200029,
                "eos-token": end_token_id,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 128,
                        "allow-async-init": False,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                    "positional-encoding": {
                        "type": "rope",
                        # int(head_dim * partial_rotary_factor) // 2 = int(128*0.75)//2.
                        # Must match the 48-wide cos/sin inputs of the compiled graph.
                        "rope-dim": 48,
                        "rope-theta": 10000,
                        "rope-scaling": {
                            "rope-type": "longrope",
                            "factor": 32.0,
                            "original-max-position-embeddings": 4096,
                            "long-factor": rope_scaling["long_factor"],
                            "short-factor": rope_scaling["short_factor"],
                        },
                    },
                },
            },
        }
    }

    assert expected_config == actual_config


@pytest.mark.unmarked
def test_chat_template() -> None:
    """Verify the Phi chat template has the correct turn tokens and Jinja template.

    The assistant_prefix must NOT include <|endoftext|>: that token is
    pre-filled in the prompt file written by _write_eval_prompts_to_dir
    (via PhiPreSplitCollectionBase.genie_generation_prompt_suffix) so that
    genie-t2t-run never sees it as the first generated token.  The
    get_chat_template() return value is consumed by GenieX (metadata.json),
    which handles the suffix separately.

    CHAT_TEMPLATE is the Jinja2 template written into tokenizer_config.json
    and used by _write_eval_prompts_to_dir via tokenizer.apply_chat_template.
    """
    from qai_hub_models.models.templates.phi.model import Phi35Base

    chat_template = Phi35Base.get_chat_template()

    assert chat_template["system_prefix"] == "<|system|>"
    assert chat_template["system_suffix"] == "<|end|>"
    assert chat_template["user_prefix"] == "<|user|>"
    assert chat_template["user_suffix"] == "<|end|>"
    assert chat_template["assistant_prefix"] == "<|assistant|>"
    assert chat_template["assistant_suffix"] == "<|end|>"

    # Verify the Jinja template produces the expected format for a simple
    # user message (no tools, no system prompt).
    jinja = Phi35Base.CHAT_TEMPLATE
    assert "<|assistant|>" in jinja
    assert "<|end|>" in jinja
    assert "add_generation_prompt" in jinja
    # Tool-call branch must be present.
    assert "<|tool|>" in jinja
    assert "<|/tool|>" in jinja

    # Verify the template renders correctly via the HuggingFace tokenizer.
    from jinja2 import Environment

    env = Environment()
    tmpl = env.from_string(jinja)
    rendered = tmpl.render(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        add_generation_prompt=True,
    )
    assert (
        rendered == "<|system|>You are helpful.<|end|><|user|>Hello<|end|><|assistant|>"
    )


@pytest.mark.unmarked
def test_tokenizer_special_tokens() -> None:
    """Verify the special-token IDs and unk_token used in tokenizer patching.

    PhiPreSplitCollectionBase._patch_tokenizer_files writes these values into
    tokenizer_config.json; this test confirms the constants are self-consistent
    with the HuggingFace tokenizer vocabulary.
    """
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_NAME)

    # pad_token_id = 200029 is an unused embedding row past the tokenizer vocab
    assert (
        len(tokenizer)
        <= PhiPreSplitCollectionBase.PAD_TOKEN_ID
        < AutoConfig.from_pretrained(HF_REPO_NAME).vocab_size
    ), f"PAD_TOKEN_ID {PhiPreSplitCollectionBase.PAD_TOKEN_ID} is not a padding row"

    # eos_token_id = 200020 (<|end|>)
    eos_encoded = tokenizer.encode("<|end|>", add_special_tokens=False)
    assert (
        len(eos_encoded) == 1
        and eos_encoded[0] == PhiPreSplitCollectionBase.EOS_TOKEN_ID
    ), (
        f"EOS_TOKEN_ID {PhiPreSplitCollectionBase.EOS_TOKEN_ID} does not match "
        f"<|end|> encoding {eos_encoded}"
    )

    # unk_token = U+FFFD
    assert PhiPreSplitCollectionBase.UNK_TOKEN == "\ufffd", (
        f"UNK_TOKEN should be U+FFFD, got {PhiPreSplitCollectionBase.UNK_TOKEN!r}"
    )
