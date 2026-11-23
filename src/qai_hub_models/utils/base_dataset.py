# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import hashlib
import inspect
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sized
from copy import copy
from enum import Enum, unique
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple, final

from torch.utils.data import DataLoader, Dataset, Subset, default_collate

from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.kwarg_helpers import cli_friendly_class_name
from qai_hub_models.utils.private_asset_loaders import UnfetchableDatasetError
from qai_hub_models.utils.samplers import shuffled_order, strided_order

__all__ = [
    "AugmentedLabelDataset",
    "BaseDataset",
    "DatasetMetadata",
    "DatasetSplit",
    "InterleavedDataset",
    "get_folder_name",
    "instantiate_dataset",
]


@unique
class DatasetSplit(Enum):
    """
    Distinct splits of the dataset should be used for training vs. validation.

    This enum can be set during dataset initialization to indicate which split to use.
    """

    TRAIN = 0
    VAL = 1
    TEST = 2


class AugmentedLabelDataset(Dataset):
    """
    Augment labels to a dataset (making the label a tuple, if labels are
    already present).
    """

    def __init__(self, base_dataset: Dataset, extra_data: Sized) -> None:
        self.base_dataset = base_dataset
        self.extra_data = extra_data
        self.extra_len = len(extra_data)

    def __len__(self) -> int:
        return len(self.base_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict[str, object]:
        item = copy(self.base_dataset[idx])
        extra_item = self.extra_data[idx % self.extra_len]  # type: ignore[index]
        if "label" in item:
            item["label"] = (item["label"], extra_item)
        else:
            item["label"] = extra_item
        return item


class DatasetMetadata(NamedTuple):
    """Metadata about the dataset to publish on the website."""

    # Link to the dataset source
    link: str

    # String describing which split was used for evaluation
    # For example, "validation split" or "partition #3 of the test split"
    split_description: str


def get_folder_name(dataset_name: str, input_spec: InputSpec | None = None) -> str:
    """The name of the folder under which to store this dataset."""
    if input_spec is None:
        return dataset_name
    sha256_hasher = hashlib.sha256()
    for key in sorted(input_spec.keys()):
        spec = input_spec[key]

        # Only include the input name and shape in the hash.
        # The model data type can change for quantized models
        # but we still want to use the same dataset in those cases.
        sha256_hasher.update(f"({key}, {spec[0]})".encode())
    hex_digest = sha256_hasher.hexdigest()
    return f"{dataset_name}_{hex_digest[:6]}"


class BaseDataset(Dataset, Sized, ABC):
    """Base class to be extended by Datasets used in this repo for quantizing models."""

    def __init__(
        self,
        dataset_path: str | Path,
        split: DatasetSplit,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.split_str = split.name.lower()
        self.input_spec = input_spec
        self.download_data()

    @staticmethod
    def collate_fn(batch: list[Any]) -> Any:
        """To be passed into DataLoader(..., collate_fn=...).

        The return is typed ``Any`` because each dataset's collated shape
        differs; subclasses narrow it in their own signatures.
        """
        return default_collate(batch)

    @final
    def download_data(self) -> None:
        if self._validate_data():
            return
        if self.dataset_path.exists():
            # Data is corrupted, delete and re-download
            if self.dataset_path.is_dir():
                shutil.rmtree(self.dataset_path)
            else:
                os.remove(self.dataset_path)

        print("Downloading data")
        try:
            self._download_data()
        except UnfetchableDatasetError as e:
            # Rebuild with the concrete class so the message can name the exact
            # `configure-dataset` command. This is the only place that knows it:
            # the error is constructed at module scope with the asset.
            raise UnfetchableDatasetError(
                e.dataset_name,
                e.installation_steps,
                dataset_cls=type(self),
                configure_files=e.configure_files,
            ) from None
        print("Done downloading")
        if not self._validate_data():
            raise ValueError("Something went wrong during download.")

    @abstractmethod
    def _download_data(self, *args: Any, **kwargs: Any) -> None:
        """Method to download necessary data to disk. To be implemented by subclass."""

    def _validate_data(self) -> bool:
        """Validates data downloaded on disk. By default just checks that folder exists."""
        return self.dataset_path.exists()

    @classmethod
    def dataset_name(cls) -> str:
        """
        CLI-friendly name derived from the class name.

        Strips a trailing "Dataset" suffix (case-insensitive), then splits
        CamelCase into underscore-separated lowercase chunks. Existing
        underscores in the class name are preserved as separators.
        """
        name = cls.__name__
        if name.lower().endswith("dataset"):
            name = name[:-7]
        return cli_friendly_class_name(name)

    @staticmethod
    @abstractmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""

    @staticmethod
    def default_num_calibration_samples() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @cached_property
    def folder_name(self) -> str:
        return get_folder_name(self.dataset_name(), self.input_spec)

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """Metadata about the dataset. Used for publishing on the website."""
        raise NotImplementedError()

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        """Configure the dataset from local files (download/extract/symlink).

        Datasets that require external setup (e.g. files behind a license
        wall, large archives users must download manually) should override
        this. Implementations typically forward `files` to the constructor.

        The expected file ordering and meaning is defined per-dataset; see
        each subclass's override.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not require external configuration. "
            "If this dataset can be auto-downloaded, just instantiate it; "
            "otherwise the dataset author should override `configure()`."
        )

    def eval_order(
        self, num_samples: int, samples_per_job: int, seed: int | None = None
    ) -> list[int]:
        """
        Choose which samples an evaluation uses, and the order they run in.

        Evaluating on every sample is too slow, so an evaluation runs on a subset of
        `num_samples`. The subset is split into consecutive batches of
        `samples_per_job` (the last may be shorter), and each batch is one on-device
        inference job. Evaluation reads samples through this method whether or not
        the dataset cache is used, so both runs see the same samples.

        By default, without a seed, each batch is spread evenly across the whole
        dataset so it is representative; with a seed the order is a seeded shuffle.
        Override this if a batch must keep certain samples together (for example
        all the crops of one image), or if a dataset is cheaper to sample directly.

        Parameters
        ----------
        num_samples
            Number of samples requested. Must already be resolved (not -1).
        samples_per_job
            Samples per batch (inference job).
        seed
            If set, a seeded random selection; otherwise evenly spread.

        Returns
        -------
        order : list[int]
            `num_samples` indices into this dataset.
        """
        if seed is None:
            return strided_order(len(self), samples_per_job, num_samples)
        return shuffled_order(len(self), seed)[:num_samples]

    def get_dataloader(
        self,
        num_samples: int,
        samples_per_job: int | None = None,
        seed: int | None = None,
    ) -> DataLoader:
        """Return a DataLoader over the samples `eval_order` selects."""
        batch_size = samples_per_job or num_samples
        return DataLoader(
            Subset(self, self.eval_order(num_samples, batch_size, seed)),
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )


class InterleavedDataset(BaseDataset):
    """Base class for datasets that interleave multiple source datasets.

    Items are selected in round-robin order: index ``i`` maps to dataset
    ``i % N`` and item ``i // N``. The total length is
    ``min_dataset_len * N`` so every dataset is exhausted evenly.

    Subclasses must implement ``dataset_name`` and ``load_datasets``.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = 0,
        **kwargs: Any,
    ) -> None:
        self._datasets = self.load_datasets(split, **kwargs)
        assert len(self._datasets) > 0
        self._min_len = min(len(ds) for ds in self._datasets)
        self.num_samples = num_samples

    @classmethod
    @abstractmethod
    def dataset_name(cls) -> str:
        """CLI-friendly name for this interleaved dataset."""

    @abstractmethod
    def load_datasets(self, split: DatasetSplit, **kwargs: Any) -> list[BaseDataset]:
        """Construct the source datasets to interleave."""

    @staticmethod
    def collate_fn(
        batch: list[dict[str, Any]],
    ) -> tuple[Any, ...]:
        item = batch[0]
        if (
            isinstance(item, Mapping)
            and "input_ids" in item
            and "attention_mask" in item
        ):
            result: tuple[Any, ...] = (
                item["input_ids"],
                item["attention_mask"],
                item.get("label", item["input_ids"]),
            )
            for key in ("pixel_values", "image_grid_thw"):
                if key in item:
                    result = (*result, item[key])
            return result
        return tuple(batch)

    def __len__(self) -> int:
        total = self._min_len * len(self._datasets)
        if self.num_samples > 0:
            return min(self.num_samples, total)
        return total

    def __getitem__(self, idx: int) -> Any:
        ds_idx = idx % len(self._datasets)
        item_idx = idx // len(self._datasets)
        return self._datasets[ds_idx][item_idx]

    def _download_data(self) -> None:
        pass

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="",
            split_description="Interleaved from multiple source datasets",
        )


def instantiate_dataset(
    dataset_cls: type[BaseDataset],
    split: DatasetSplit,
    input_spec: InputSpec | None = None,
    **kwargs: Any,
) -> BaseDataset:
    init_params = inspect.signature(dataset_cls.__init__).parameters
    if input_spec is not None and "input_spec" in init_params:
        kwargs["input_spec"] = input_spec

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in init_params.values()
    )
    if not has_var_keyword:
        kwargs = {k: v for k, v in kwargs.items() if k in init_params}

    return dataset_cls(split=split, **kwargs)
