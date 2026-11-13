# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""ImageNet validation dataset for zero-shot image-text evaluation.

Supports both image-only and joint image+text modes:

* **Image-only** (default, ``tokenizer_id=None``):
  ``__getitem__`` returns ``(image, label)`` — identical to the previous
  behaviour, so all existing callers (evaluate.py, etc.) are unaffected.

* **Joint image+text** (``tokenizer_id`` provided):
  ``__getitem__`` returns ``(image, input_ids, label)``.
  ``input_ids`` is the tokenized ``"a photo of a {label}"`` prompt for the
  ground-truth class, making the dataset directly usable for calibrating
  *both* the image encoder and the text encoder of a vision-language model
  such as SigLIP2.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenet.imagenet import ImagenetDataset
from qai_hub_models.utils.labels import get_class_names

# Zero-shot prompt template applied to every ImageNet class name.
PROMPT_TEMPLATE = "a photo of a {label}"


def imagenet_prompts() -> list[str]:
    """Zero-shot prompts for all ImageNet-1K classes."""
    return [PROMPT_TEMPLATE.format(label=lbl) for lbl in get_class_names("imagenet")]


def tokenize_prompts(
    prompts: list[str], tokenizer_id: str, text_seq_len: int
) -> torch.Tensor:
    """Tokenize a list of prompts into int32 input_ids [N, text_seq_len]."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    token_out = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=text_seq_len,
    )
    return token_out["input_ids"].to(torch.int32)


class ImagenetZeroshotDataset(ImagenetDataset):
    """ImageNet validation set for zero-shot image(-text) evaluation.

    Parameters
    ----------
    split
        Dataset split. Only ``VAL`` is supported.
    tokenizer_id
        Hugging Face model id whose tokenizer is used to encode class prompts.
        When ``None`` (default) the dataset operates in *image-only* mode and
        ``__getitem__`` returns ``(image, label)``.
        When provided the dataset operates in *joint* mode and ``__getitem__``
        returns ``(image, input_ids, label)``.
    text_seq_len
        Fixed sequence length for tokenized prompts.  Only used when
        ``tokenizer_id`` is not ``None``.

    Notes
    -----
    In joint mode all 1000 class prompts are tokenized once at construction
    time and cached in memory.  The ``input_ids`` returned for a given sample
    correspond to the ground-truth class of that sample, so the same dataset
    can be used to calibrate both the image encoder (use ``image``) and the
    text encoder (use ``input_ids``) of a vision-language model.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        tokenizer_id: str | None = None,
        text_seq_len: int = 64,
    ) -> None:
        super().__init__(split=split)

        # Pre-tokenize all 1000 class prompts when a tokenizer is requested.
        self._class_input_ids: torch.Tensor | None = None
        if tokenizer_id is not None:
            # Shape: [1000, text_seq_len], dtype int32
            self._class_input_ids = tokenize_prompts(
                imagenet_prompts(), tokenizer_id, text_seq_len
            )

    def __getitem__(  # type: ignore[override]
        self, index: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Return a single sample.

        Parameters
        ----------
        index
            Sample index into the dataset.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(image, label)`` in image-only mode, or ``(image, input_ids, label)``
            in joint mode. ``image`` is float32 ``[3, 224, 224]``; ``input_ids``
            is int32 ``[text_seq_len]`` (joint mode only); ``label`` is the
            ground-truth class index in ``[0, 999]``.
        """
        image, label = super().__getitem__(index)
        if self._class_input_ids is not None:
            return image, self._class_input_ids[label], label
        return image, label
