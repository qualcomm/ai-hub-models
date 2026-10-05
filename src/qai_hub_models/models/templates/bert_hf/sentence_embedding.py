# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared sentence-embedding evaluation helpers for BERT-family models.

Provides a linear-probe evaluator (MTEB protocol) and the MTEB
``amazon_counterfactual`` classification dataset. Subclasses of
:class:`AmazonCounterfactualDataset` set ``tokenizer_name`` to the
tokenizer used by the embedding model under test.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
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


class AmazonCounterfactualDataset(BaseDataset):
    """MTEB amazon_counterfactual binary classification dataset.

    Subclasses must set ``tokenizer_name`` to the HuggingFace tokenizer
    used by the embedding model being evaluated. Returns tokenized
    (input_ids, attention_mask) pairs with binary labels
    (0=not-counterfactual, 1=counterfactual).
    """

    tokenizer_name: str
    default_seq_length: int = 128

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        seq_len: int | None = None,
    ) -> None:
        if split == DatasetSplit.TRAIN:
            self.ds = load_dataset("mteb/amazon_counterfactual", "en", split="train")
        else:
            self.ds = load_dataset("mteb/amazon_counterfactual", "en", split="test")

        BaseDataset.__init__(self, "non_existent_dir", split)
        self.seq_len = seq_len or self.default_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, model_max_length=self.seq_len
        )

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        """Return (input_ids, attention_mask) and label."""
        text = self.ds[index]["text"]
        label = self.ds[index]["label"]
        tokens = self.tokenizer(
            text, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = cast(torch.Tensor, tokens["input_ids"]).squeeze(0)
        attention_mask = cast(torch.Tensor, tokens["attention_mask"]).squeeze(0)
        return (input_ids, attention_mask), label

    def __len__(self) -> int:
        return len(self.ds)

    def _validate_data(self) -> bool:
        return hasattr(self, "ds")

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 100
