# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from types import SimpleNamespace

from qai_hub_models.models.templates.llm.llm_helpers import create_genie_config


def test_create_genie_config_preserves_rope_scaling_factor() -> None:
    rope_scaling = {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    llm_config = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=32,
        vocab_size=128256,
        bos_token_id=128000,
        eos_token_id=[128001, 128008, 128009],
        rope_theta=500000,
        rope_scaling=rope_scaling,
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
