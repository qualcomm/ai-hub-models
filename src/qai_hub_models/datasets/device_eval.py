# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Prompt sets that can be run on device and graded.

A device-eval dataset describes its prompts without a tokenizer or processor,
so the Genie and GenieX runners can stage it on the host. Decorate a dataset
class with :func:`register_device_eval_dataset` and the device runners and
``collect_llm_accuracy_csv`` can run and file it by ``dataset_name()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TypeVar

from qai_hub_models.models.templates.llm.grace_tasks import MULTIMODAL_TASK_NAME
from qai_hub_models.models.templates.llm.grader.grace import GRACE_TASK_NAME
from qai_hub_models.utils.metrics import MetricMetadata


@dataclass(frozen=True)
class DeviceEvalItem:
    """One prompt as the device runners and the grader see it.

    ``idx`` is the dataset's own index, stable across subsets, so responses
    from a shortened run still join against the full set.
    """

    idx: int
    prompt: str
    category: str | None = None
    image_paths: tuple[str, ...] = field(default_factory=tuple)


class DeviceEvalDataset(ABC):
    """Mixin for a dataset whose prompts can be staged on device and graded.

    Everything here is class-level, so device runners never build a
    tokenizer or processor to get the prompts.
    """

    # Runners that cannot send images refuse a dataset that sets this.
    requires_images: bool = False

    @classmethod
    @abstractmethod
    def dataset_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def device_eval_items(cls) -> list[DeviceEvalItem]:
        """Every prompt in the set, in ``idx`` order. Device runs always use all of them."""

    @classmethod
    @abstractmethod
    def metric_metadata(cls) -> MetricMetadata:
        """The metric the graded score is filed under."""

    @classmethod
    @abstractmethod
    def doc_url(cls) -> str:
        """Public documentation for the prompt set, published with the score."""


_DatasetT = TypeVar("_DatasetT", bound=type[DeviceEvalDataset])
_REGISTRY: dict[str, type[DeviceEvalDataset]] = {}

# What a device run evaluates when the caller names no dataset.
DEFAULT_DEVICE_EVAL_DATASET = GRACE_TASK_NAME


def register_device_eval_dataset(cls: _DatasetT) -> _DatasetT:
    """Class decorator that makes a dataset runnable on device by name."""
    name = cls.dataset_name()
    existing = _REGISTRY.get(name)
    assert existing is None or existing is cls, (
        f"device-eval dataset {name!r} is registered twice: {existing} and {cls}"
    )
    _REGISTRY[name] = cls
    return cls


def get_device_eval_dataset(name: str) -> type[DeviceEvalDataset]:
    _load_builtin_datasets()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown device-eval dataset {name!r}. "
            f"Registered: {', '.join(sorted(_REGISTRY))}."
        ) from None


def is_device_eval_dataset(name: str) -> bool:
    _load_builtin_datasets()
    return name in _REGISTRY


# Prompt tasks graded like device-eval datasets but not runnable on device.
_HOST_ONLY_PROMPT_TASKS = frozenset({MULTIMODAL_TASK_NAME})


def is_prompt_task(task: str) -> bool:
    """Whether ``task``'s evaluator generates free-form responses and grades them,
    as opposed to the forward-only metrics (wikitext, mmlu, ...).
    """
    return task in _HOST_ONLY_PROMPT_TASKS or is_device_eval_dataset(task)


def load_device_eval_items(
    name: str, *, supports_images: bool = False
) -> list[DeviceEvalItem]:
    """The prompts a device runner stages for ``name``: always the whole set.

    Raises for an image dataset on a runner that cannot send images, rather
    than silently running the questions without their images.
    """
    dataset = get_device_eval_dataset(name)
    if dataset.requires_images and not supports_images:
        raise NotImplementedError(
            f"{name!r} needs images, which this device runner cannot send yet."
        )
    return dataset.device_eval_items()


def label_eval_results(results: list[dict], items: list[DeviceEvalItem]) -> list[dict]:
    """Map device results, keyed by staged position, back onto the dataset's items.

    Device scripts number prompt files 0..N-1 in staging order, so a subset run
    would otherwise record positions instead of the dataset's own ``idx``.
    """
    labeled = []
    for result in results:
        position = result["idx"]
        if not 0 <= position < len(items):
            continue
        item = items[position]
        labeled.append(
            {
                "idx": item.idx,
                "category": item.category,
                "prompt": item.prompt,
                "output": result["output"],
                **({"image_paths": list(item.image_paths)} if item.image_paths else {}),
            }
        )
    return labeled


def _load_builtin_datasets() -> None:
    # Imported here, not at the top: the built-in datasets import this module to register.
    import qai_hub_models.datasets.grace  # noqa: F401
