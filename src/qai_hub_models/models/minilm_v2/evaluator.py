# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Linear probe evaluator for sentence embedding models.

Kept as a re-export for backwards compatibility; the implementation now
lives in the shared ``templates/bert_hf`` sentence-embedding helpers.
"""

from qai_hub_models.models.templates.bert_hf.sentence_embedding import (
    SentenceEmbeddingEvaluator,
)

__all__ = ["SentenceEmbeddingEvaluator"]
