# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod

import torch
from transformers import AutoTokenizer

from qai_hub_models.datasets.coco import CocoDataset
from qai_hub_models.models.templates.wedetect.app import tokenize_class_names
from qai_hub_models.models.templates.wedetect.constants import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_CLASSES,
)
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.labels import get_class_names


class WeDetectCocoDataset(CocoDataset):
    """COCO detection dataset that also emits pre-tokenized class prompts.

    Each sample yields ``((image, input_ids, attention_mask), ground_truth)``.
    The tokens are identical for every sample (COCO's fixed 80-class
    vocabulary), so tokenization happens once at construction time.
    """

    @classmethod
    @abstractmethod
    def tokenizer_path(cls) -> str:
        """Return the XLM-RoBERTa tokenizer bundle directory."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(split=split, input_spec=input_spec)
        tokenizer = AutoTokenizer.from_pretrained(type(self).tokenizer_path())
        input_ids, attention_mask = tokenize_class_names(
            tokenizer,
            get_class_names("coco"),
            DEFAULT_NUM_CLASSES,
            DEFAULT_MAX_SEQ_LEN,
        )
        self._input_ids = input_ids
        self._attention_mask = attention_mask

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        image, ground_truth = super().__getitem__(index)
        return (image, self._input_ids, self._attention_mask), ground_truth
