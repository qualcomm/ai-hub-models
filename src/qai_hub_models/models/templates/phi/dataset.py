# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit

if TYPE_CHECKING:
    from qai_hub_models.models.templates.llm.model import LLMBase


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        batch[0]["input_ids"],
        batch[0]["attention_mask"],
        batch[0].get("label", batch[0]["input_ids"]),
    )


class BaseTrickyLLMPrompts(BaseDataset):
    """
    LLM prompts with known problem areas (in the subclass's prompt format). The
    prompts typically include part of a reasonable answer and is cut off
    mid-sentence where known problems tend to appear.

    Because of this, distance metrics (such as KL divergence) on the last token
    are appropriate to quantify these failures.
    """

    MODEL_NAME: str = ""  # Must be set in subclass

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int = 128,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
    ) -> None:
        self.block_size = block_size
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        if split != DatasetSplit.TEST:
            raise ValueError("Tricky LLM prompts only supports `test` split")

        self.prompts = self.tokenizer(
            self.raw_prompts(),
            add_special_tokens=True,
        )

    def raw_prompts(self) -> list[str]:
        """
        Should return a list of raw prompts (including headers) with some part
        of the answer included, but cut off at a challenging token.
        """
        # Note, raw_prompts could also have been a class variable, like
        # MODEL_NAME. However, making it a function allows us to lazy-import
        # models and use their prompt format processors. Global import of
        # models in this file will trigger a circular dependency.
        raise NotImplementedError()

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch[0]["input_ids"],
            batch[0]["attention_mask"],
            batch[0].get("label", batch[0]["input_ids"]),
        )

    def __len__(self) -> int:
        # Each prompt is considered a sample
        max_num = len(self.prompts["input_ids"])
        num = self.num_samples if self.num_samples != 0 else max_num
        return min(num, max_num)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary of tokenized LLM inputs with common prompts.

        Contains:
            - "input_ids": Tokenized input (int32).
            - "attention_mask": 1D attention mask (float32, with zeros and ones).
        """
        return {
            "input_ids": torch.Tensor(self.prompts["input_ids"][index : index + 1]).to(
                torch.int
            ),
            "attention_mask": torch.Tensor(
                self.prompts["attention_mask"][index : index + 1]
            ).to(torch.int),
        }

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1

    @classmethod
    def dataset_name(cls) -> str:
        assert cls.MODEL_NAME, (
            "MODEL_NAME must be set in BaseTrickyLLMPrompts subclass."
        )
        return "tricky_llm_prompts_" + cls.MODEL_NAME


class TrickyLLMPromptsPhi35(BaseTrickyLLMPrompts):
    """Abstract base for Phi-family tricky-prompt datasets (Phi-3.5 and Phi-4).

    The prompts are tokenizer- and prompt-format-specific, so this class carries no
    variant of its own. Concrete subclasses must set:

        MODEL_NAME   -- identifier used in dataset_name() (e.g. "phi4_mini")
        HF_REPO_NAME -- HuggingFace repo supplying the tokenizer
        LLMBaseClass -- model base whose get_input_prompt_with_tags() produces the
                        correct prompt format for this variant
    """

    MODEL_NAME: str = ""  # set in subclass
    HF_REPO_NAME: str = ""  # set in subclass
    LLMBaseClass: type[LLMBase] | None = None  # set in subclass

    def raw_prompts(self) -> list[str]:
        assert self.HF_REPO_NAME, (
            "HF_REPO_NAME must be set in TrickyLLMPromptsPhi35 subclass."
        )
        llm_cls = self.LLMBaseClass
        assert llm_cls is not None, (
            "LLMBaseClass must be set in TrickyLLMPromptsPhi35 subclass."
        )

        tokenizer = AutoTokenizer.from_pretrained(self.HF_REPO_NAME)

        return [
            (
                llm_cls.get_input_prompt_with_tags(
                    user_input_prompt="What is Gravity?",
                    system_context_prompt="You are a helpful AI assistant.",
                    tokenizer=tokenizer,
                )
                + "Gravity is a fundamental force of nature that attracts two bodies with mass towards each other. It is described by Isaac Newton'"
            ),
            (
                llm_cls.get_input_prompt_with_tags(
                    user_input_prompt="What is Gravity?",
                    system_context_prompt="You are a helpful AI assistant.",
                    tokenizer=tokenizer,
                )
                + "Gravity is a fundamental force of nature that attracts two bodies with mass towards each other. It is described by Isaac Newton's theory in the 17th century and is a key component in Albert Einstein'"
            ),
        ]
