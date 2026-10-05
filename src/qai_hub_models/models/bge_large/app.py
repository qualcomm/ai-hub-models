# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from qai_hub_models.models.bge_large.model import HF_MODEL_ID, MAX_SEQ_LENGTH
from qai_hub_models.protocols import ExecutableModelProtocol


class BGELargeApp:
    """End-to-end sentence embedding application.

    Handles tokenization and produces normalized 1024-dim sentence embeddings.
    The model accepts (input_ids, attention_mask) and returns embeddings —
    works with both PyTorch model and on-device inference.
    """

    def __init__(
        self,
        model: ExecutableModelProtocol[torch.Tensor],
        seq_length: int = MAX_SEQ_LENGTH,
    ) -> None:
        self.model = model
        self.seq_length = seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

    def encode(self, texts: str | list[str]) -> torch.Tensor:
        """Encode one or more sentences into normalized embeddings.

        Parameters
        ----------
        texts
            A single sentence or list of sentences to encode.

        Returns
        -------
        embeddings : torch.Tensor
            Normalized embeddings of shape [num_sentences, 1024].
        """
        if isinstance(texts, str):
            texts = [texts]

        # Encode sentences one at a time so the same code path works for
        # both PyTorch and on-device models (the latter is compiled with a
        # fixed batch size of 1).
        outputs: list[torch.Tensor] = []
        for text in texts:
            tokens = self.tokenizer(
                [text],
                padding="max_length",
                truncation=True,
                max_length=self.seq_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(torch.int32)
            attention_mask = tokens["attention_mask"].to(torch.int32)

            with torch.no_grad():
                outputs.append(self.model(input_ids, attention_mask))

        return torch.cat(outputs, dim=0)

    def predict(self, texts: str | list[str]) -> torch.Tensor:
        """Inference entry point; equivalent to :meth:`encode`."""
        return self.encode(texts)
