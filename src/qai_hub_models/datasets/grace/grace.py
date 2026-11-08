# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from PIL import Image

from qai_hub_models.datasets.device_eval import (
    DeviceEvalDataset,
    DeviceEvalItem,
    register_device_eval_dataset,
)
from qai_hub_models.models.templates.llm.grace_tasks import (
    GRACE_DOC_URL,
    MULTIMODAL_TASK_NAME,
)
from qai_hub_models.models.templates.llm.grader.grace import (
    GRACE_PROMPTS_PATH,
    GRACE_TASK_NAME,
    load_eval_prompts,
    select_balanced,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.metrics import GRACE2_GRADE, MetricMetadata

# Type-only: collect_llm_accuracy_csv imports this module from a venv without transformers.
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

MULTIMODAL_DATASET_ID = MULTIMODAL_TASK_NAME
MULTIMODAL_VERSION = 1
MULTIMODAL_PROMPTS_FILENAME = "multimodal_prompts.yaml"
SAMPLE_IMAGES_SUBDIR = "sample_images"


@dataclass
class PromptLabel:
    """Per-sample metadata used by ``LLMResponseEvaluator``.

    Carried as the dataset ``label`` so it survives the standard
    ``(input_ids, attention_mask, label, ...)`` collate convention.
    """

    index: int
    prompt: str
    image_path: str | None = None
    category: str | None = None


def _format_text_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    is_vlm: bool,
) -> str:
    """Apply the model's chat template to a raw user prompt.

    Thinking is disabled to match the on-device Genie path and the FP baselines.
    A thinking model otherwise spends its whole token budget on a reasoning
    trace, so the graded response is the trace rather than an answer.
    Non-thinking templates ignore the unused variable.
    """
    content: Any = prompt
    if is_vlm:
        # VLM processors (Qwen2.5-VL) require a list-of-content-parts. This form
        # is NOT safe for plain LLMs: their chat templates concatenate content
        # as a string, so the list either raises (Qwen2.5 text) or leaks a
        # literal "[{'type': 'text', ...}]" into the prompt (Llama-3.2). Keep
        # the bare string for non-VLMs.
        content = [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert isinstance(formatted_prompt, str)
    return formatted_prompt


@register_device_eval_dataset
class Grace2(BaseDataset, DeviceEvalDataset):
    """The Grace2 prompt set: 10 categories x 10 text-only prompts.

    Yields items shaped like the other LLM eval datasets so they collate the
    same way and the evaluator can drop them straight into the generator.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
        # processor accepted (and ignored) for API symmetry with VLM datasets
        processor: Any = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if split != DatasetSplit.TEST:
            raise ValueError("Grace2 only supports the `test` split")
        if tokenizer is None:
            raise ValueError("Grace2 requires a tokenizer.")
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.num_samples = num_samples
        self.is_vlm = processor is not None

        super().__init__(GRACE_PROMPTS_PATH, split)
        self.eval_prompts = load_eval_prompts()
        if num_samples and num_samples > 0:
            # Records are grouped by category, so a prefix slice would leave a
            # short smoke run reporting only the first category or two.
            self.eval_prompts = select_balanced(self.eval_prompts, num_samples)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
        item = batch[0]
        return item["input_ids"], item["attention_mask"], item["label"]

    def __len__(self) -> int:
        return len(self.eval_prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.eval_prompts[idx]
        prompt = entry.prompt
        formatted = _format_text_prompt(self.tokenizer, prompt, self.is_vlm)
        tokenized = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
            return_token_type_ids=False,
        )
        input_ids = tokenized["input_ids"][:, -self.context_length :]
        attention_mask = tokenized["attention_mask"][:, -self.context_length :]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            # index is the prompt's own idx, not its position in a subset, so a
            # truncated run's responses still join against the full set.
            "label": PromptLabel(
                index=entry.idx, prompt=prompt, category=entry.category
            ),
        }

    def _download_data(self) -> None:
        # The prompt set ships with the repo; nothing to download.
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link=GRACE_DOC_URL,
            split_description="test split (10 categories x 10 prompts)",
        )

    @classmethod
    def dataset_name(cls) -> str:
        return GRACE_TASK_NAME

    @classmethod
    def device_eval_items(cls) -> list[DeviceEvalItem]:
        return [
            DeviceEvalItem(p.idx, p.prompt, p.category) for p in load_eval_prompts()
        ]

    @classmethod
    def metric_metadata(cls) -> MetricMetadata:
        return GRACE2_GRADE

    @classmethod
    def doc_url(cls) -> str:
        return GRACE_DOC_URL


def _multimodal_image_asset(filename: str) -> CachedWebDatasetAsset:
    """Asset descriptor for one sample image in the multimodal_prompts bucket."""
    return CachedWebDatasetAsset.from_asset_store(
        MULTIMODAL_DATASET_ID,
        MULTIMODAL_VERSION,
        f"{SAMPLE_IMAGES_SUBDIR}/{filename}",
    )


class MultimodalPrompts(BaseDataset):
    """Image + question pairs for qualitative VLM evaluation.

    Tokenization, image preprocessing, and vision token insertion are
    delegated to the VLM processor (same approach as ``AOKVQA``).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        context_length: int = 4096,
        split: DatasetSplit = DatasetSplit.TEST,
        num_samples: int = 0,
        processor: Any = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if split != DatasetSplit.TEST:
            raise ValueError("MultimodalPrompts only supports the `test` split")
        if processor is None:
            raise ValueError(
                "MultimodalPrompts requires a VLM processor "
                "(pass one through the evaluator's vlm_processor argument)."
            )
        self.processor = processor
        self.context_length = context_length
        self.num_samples = num_samples
        self.image_size: tuple[int, int] | None = (
            (int(image_size[0]), int(image_size[1])) if image_size is not None else None
        )

        self._yaml_asset = CachedWebDatasetAsset.from_asset_store(
            MULTIMODAL_DATASET_ID, MULTIMODAL_VERSION, MULTIMODAL_PROMPTS_FILENAME
        )
        super().__init__(self._yaml_asset.path, split)

        with open(self._yaml_asset.path) as f:
            raw = yaml.safe_load(f)
        assert isinstance(raw, list)
        items: list[dict[str, str]] = []
        for entry in raw:
            if (
                not isinstance(entry, dict)
                or "image" not in entry
                or "prompt" not in entry
            ):
                raise TypeError(
                    f"Each {self._yaml_asset.path} entry must be a mapping with "
                    f"'image' and 'prompt'."
                )
            items.append({"image": str(entry["image"]), "prompt": str(entry["prompt"])})
        self.items = items

        # Resolve each unique image filename to its on-disk path. Doing it up
        # front gives a nicer error if S3 access is broken; fetch() is cheap
        # when the file is already cached.
        self._image_paths: dict[str, Path] = {}
        for filename in {item["image"] for item in items}:
            asset = _multimodal_image_asset(filename)
            self._image_paths[filename] = Path(asset.fetch())

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> tuple[Any, ...]:
        item = batch[0]
        result: tuple[Any, ...] = (
            item["input_ids"],
            item["attention_mask"],
            item["label"],
        )
        for key in ("pixel_values", "image_grid_thw"):
            if key in item:
                result = (*result, item[key])
        return result

    def __len__(self) -> int:
        if self.num_samples and self.num_samples > 0:
            return min(self.num_samples, len(self.items))
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        filename = item["image"]
        prompt = item["prompt"]
        image_path = self._image_paths[filename]

        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            image = image.resize(self.image_size)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        assert inputs["input_ids"].shape[1] <= self.context_length
        inputs.pop("mm_token_type_ids", None)
        inputs["label"] = PromptLabel(
            index=idx, prompt=prompt, image_path=str(image_path)
        )
        return inputs

    def _download_data(self) -> None:
        self._yaml_asset.fetch()
        # Image fetches happen in __init__ once the YAML is parsed.

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @classmethod
    def dataset_name(cls) -> str:
        return MULTIMODAL_TASK_NAME


__all__ = [
    "Grace2",
    "MultimodalPrompts",
    "PromptLabel",
]
