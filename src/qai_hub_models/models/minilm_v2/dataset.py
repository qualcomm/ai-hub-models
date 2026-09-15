# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Amazon Counterfactual dataset for MiniLM sentence embeddings."""

from __future__ import annotations

from qai_hub_models.models.templates.bert_hf.sentence_embedding import (
    AmazonCounterfactualDataset,
)

HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEQ_LENGTH = 128


class MiniLMAmazonCounterfactualDataset(AmazonCounterfactualDataset):
    """Amazon Counterfactual binary classification dataset for MiniLM evaluation.

    Uses the MiniLM tokenizer (no instruction prefix — unlike Nomic which
    prepends "classification: ").
    """

    tokenizer_name = HF_MODEL_ID
    default_seq_length = DEFAULT_SEQ_LENGTH
