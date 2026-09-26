# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoModel
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.models.bge_large.dataset import BGEAmazonCounterfactualDataset
from qai_hub_models.models.bge_large.evaluator import SentenceEmbeddingEvaluator
from qai_hub_models.models.templates.bert_hf.model_patches import (
    patch_get_extended_attention_mask,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
HF_MODEL_ID = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 128


class BGELarge(BaseModel):
    """Sentence embedding model that maps text to a 1024-dim vector.

    Given a tokenized sentence (input_ids, attention_mask), produces a
    normalized 1024-dimensional embedding suitable for semantic similarity,
    search, and clustering. The forward graph includes the transformer,
    [CLS] token pooling, and L2 norm, matching the reference
    BAAI/bge-large-en-v1.5 sentence-transformers model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.get_extended_attention_mask = patch_get_extended_attention_mask  # type: ignore[assignment]

    @classmethod
    def from_pretrained(cls, weights: str = HF_MODEL_ID) -> Self:
        model = AutoModel.from_pretrained(weights)
        return cls(model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Produce sentence embeddings via [CLS] token pooling.

        Parameters
        ----------
        input_ids
            Token IDs of shape [batch_size, seq_len].
        attention_mask
            Attention mask of shape [batch_size, seq_len].

        Returns
        -------
        embeddings : torch.Tensor
            Normalized sentence embeddings of shape [batch_size, 1024].
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

    def get_input_spec(
        self,
        batch_size: int = 1,
        seq_length: int = MAX_SEQ_LENGTH,
    ) -> InputSpec:
        return {
            "input_ids": TensorSpec(
                shape=(batch_size, seq_length),
                dtype="int32",
            ),
            "attention_mask": TensorSpec(
                shape=(batch_size, seq_length),
                dtype="int32",
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {"embeddings": TensorSpec()}

    def get_evaluator(self) -> BaseEvaluator:
        return SentenceEmbeddingEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [BGEAmazonCounterfactualDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return BGEAmazonCounterfactualDataset

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        """Generate realistic sample inputs with proper token IDs and attention mask."""
        if not input_spec:
            input_spec = self.get_input_spec()
        seq_len = (
            input_spec["input_ids"][0][1]
            if isinstance(input_spec["input_ids"], tuple)
            else input_spec["input_ids"].shape[1]
        )
        real_length = seq_len // 2
        input_ids = np.zeros((1, seq_len), dtype=np.int32)
        rng = np.random.default_rng(42)
        input_ids[0, :real_length] = rng.integers(1, 30000, size=real_length)
        attention_mask = np.zeros((1, seq_len), dtype=np.int32)
        attention_mask[0, :real_length] = 1
        return {"input_ids": [input_ids], "attention_mask": [attention_mask]}
