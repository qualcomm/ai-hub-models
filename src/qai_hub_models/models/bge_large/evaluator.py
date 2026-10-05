# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Linear probe evaluator for sentence embedding models."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    ACCURACY_TOP1,
    MetricMetadata,
)


class SentenceEmbeddingEvaluator(BaseEvaluator):
    """Evaluator for sentence embedding models using linear probe classification.

    Trains a logistic regression on accumulated embeddings from the training
    split, then evaluates classification accuracy on the test split.
    This follows the MTEB linear probing evaluation protocol.

    The evaluator accumulates all embeddings and labels via add_batch(),
    then splits 80/20 for train/test when get_accuracy_score() is called.
    """

    def __init__(self, max_iter: int = 1000, seed: int = 42) -> None:
        self.max_iter = max_iter
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.embeddings: list[np.ndarray] = []
        self.labels: list[int] = []

    def add_batch(self, output: torch.Tensor, gt: torch.Tensor | int) -> None:
        """Accumulate embeddings and labels.

        Parameters
        ----------
        output
            Sentence embeddings of shape [batch_size, embedding_dim].
        gt
            Ground truth labels of shape [batch_size] or scalar.
        """
        out_np = (
            output.detach().cpu().numpy()
            if isinstance(output, torch.Tensor)
            else output
        )
        gt_np = (
            gt.detach().cpu().numpy()
            if isinstance(gt, torch.Tensor)
            else np.array([gt])
        )

        for i in range(out_np.shape[0]):
            self.embeddings.append(out_np[i])
            self.labels.append(int(gt_np[i]) if i < len(gt_np) else int(gt_np[0]))

    def get_accuracy_score(self) -> float:
        """Train linear probe on first 80% of data, test on last 20%."""
        if len(self.embeddings) < 10:
            return 0.0

        X = np.array(self.embeddings)
        y = np.array(self.labels)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if len(set(y_train)) < 2 or len(X_test) == 0:
            return 0.0

        clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred) * 100

    def formatted_accuracy(self) -> str:
        return f"{self.get_accuracy_score():.1f}% (Top 1)"

    def get_metric_metadata(self) -> MetricMetadata:
        return ACCURACY_TOP1
