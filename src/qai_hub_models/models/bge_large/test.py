# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.models.bge_large.app import BGELargeApp
from qai_hub_models.models.bge_large.demo import main as demo_main
from qai_hub_models.models.bge_large.model import (
    EMBEDDING_DIM,
    BGELarge,
)


def test_task() -> None:
    model = BGELarge.from_pretrained()
    model.eval()
    app = BGELargeApp(model)

    embeddings = app.encode(
        [
            "The cat sat on the mat.",
            "A feline rested on the rug.",
            "Stock prices rose sharply today.",
        ]
    )

    assert embeddings.shape == (3, EMBEDDING_DIM)
    norms = embeddings.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    # Pin cosine similarities to golden values (verifies the model produces
    # the expected BAAI/bge-large-en-v1.5 embeddings).
    sim_similar = torch.nn.functional.cosine_similarity(
        embeddings[0:1], embeddings[1:2]
    ).item()
    sim_different_a = torch.nn.functional.cosine_similarity(
        embeddings[0:1], embeddings[2:3]
    ).item()
    sim_different_b = torch.nn.functional.cosine_similarity(
        embeddings[1:2], embeddings[2:3]
    ).item()
    assert abs(sim_similar - 0.8083) < 0.01, f"got {sim_similar:.4f}"
    assert abs(sim_different_a - 0.2460) < 0.01, f"got {sim_different_a:.4f}"
    assert abs(sim_different_b - 0.2384) < 0.01, f"got {sim_different_b:.4f}"


def test_demo() -> None:
    demo_main(is_test=True)
