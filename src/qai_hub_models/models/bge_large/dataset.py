# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Amazon Counterfactual dataset for BGE sentence embeddings."""

from __future__ import annotations

from typing import cast

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit

HF_MODEL_ID = "BAAI/bge-large-en-v1.5"
DEFAULT_SEQ_LENGTH = 128


class BGEAmazonCounterfactualDataset(BaseDataset):
    """Amazon Counterfactual binary classification dataset for BGE evaluation.

    Returns tokenized (input_ids, attention_mask) pairs with binary labels
    (0=not-counterfactual, 1=counterfactual). BGE does not use an instruction
    prefix for classification, so the raw text is tokenized directly.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        seq_len: int = DEFAULT_SEQ_LENGTH,
    ) -> None:
        if split == DatasetSplit.TRAIN:
            self.ds = load_dataset("mteb/amazon_counterfactual", "en", split="train")
        else:
            self.ds = load_dataset("mteb/amazon_counterfactual", "en", split="test")

        BaseDataset.__init__(self, "non_existent_dir", split)
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_ID, model_max_length=seq_len
        )
        self.seq_len = seq_len

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
