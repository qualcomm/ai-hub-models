# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from qai_hub_models.models.templates.llm.llm_helpers import create_genie_config

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def test_create_genie_config_preserves_rope_scaling_factor() -> None:
    rope_scaling = {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    # create_genie_config only reads attributes off llm_config, so a stub avoids
    # importing transformers here.
    llm_config = cast(
        "PretrainedConfig",
        SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=32,
            vocab_size=128256,
            bos_token_id=128000,
            eos_token_id=[128001, 128008, 128009],
            rope_theta=500000,
            rope_scaling=rope_scaling,
        ),
    )

    genie_config = create_genie_config(
        context_length=1024,
        llm_config=llm_config,
        embedding_type="rope",
        model_list=["part1.bin", "part2.bin", "part3.bin"],
    )

    output_factor = genie_config["dialog"]["engine"]["model"]["positional-encoding"][
        "rope-scaling"
    ]["factor"]

    assert output_factor == rope_scaling["factor"]


def test_create_genie_config_emits_longrope() -> None:
    long_factor = [1.0 + i / 10 for i in range(48)]
    short_factor = [1.0] * 48
    llm_config = cast(
        "PretrainedConfig",
        SimpleNamespace(
            hidden_size=3072,
            num_attention_heads=24,
            partial_rotary_factor=0.75,
            vocab_size=200064,
            bos_token_id=199999,
            eos_token_id=199999,
            rope_theta=10000.0,
            max_position_embeddings=131072,
            original_max_position_embeddings=4096,
            rope_scaling={
                "type": "longrope",
                "long_factor": long_factor,
                "short_factor": short_factor,
            },
        ),
    )

    genie_config = create_genie_config(
        context_length=4096,
        llm_config=llm_config,
        embedding_type="rope",
        model_list=["part1.bin"],
    )

    engine = genie_config["dialog"]["engine"]
    assert "pos-id-dim" not in engine["backend"]["QnnHtp"]
    assert "rope-theta" not in engine["backend"]["QnnHtp"]
    assert engine["model"]["positional-encoding"] == {
        "type": "rope",
        "rope-dim": 48,
        "rope-theta": 10000,
        "rope-scaling": {
            "rope-type": "longrope",
            "factor": 32.0,
            "original-max-position-embeddings": 4096,
            "long-factor": long_factor,
            "short-factor": short_factor,
        },
    }
