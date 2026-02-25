# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import hashlib
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sized
from copy import copy
from enum import Enum, unique
from functools import cached_property
from pathlib import Path
from typing import Any, NamedTuple, final

from torch.utils.data import Dataset, default_collate

from qai_hub_models.utils.asset_loaders import LOCAL_STORE_DEFAULT_PATH
from qai_hub_models.utils.envvars import IsOnCIEnvvar
from qai_hub_models.utils.input_spec import InputSpec


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
    def collate_fn(batch: list[object]) -> list[object]:
        """To be passed into DataLoader(..., collate_fn=...)."""
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
        self._download_data()
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
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return cls.__module__.split(".")[-1]

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


def setup_fiftyone_env() -> None:
    """
    FiftyOne is an external library that provides utilities for downloading and storing
    datasets. We want all of its operations to be done within the ai-hub-models cache
    directory.

    Import within the function so it only happens when the function is called.
    """
    try:
        import fiftyone as fo
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "This dataset requires the `fiftyone` module. Run `pip install fiftyone>=1.0.1,<1.9` to use this dataset."
        ) from None

    fiftyone_dir = os.path.join(LOCAL_STORE_DEFAULT_PATH, "fiftyone")
    fo.config.database_dir = os.path.join(fiftyone_dir, "mongo")
    fo.config.dataset_zoo_dir = fiftyone_dir
    fo.config.default_dataset_dir = fiftyone_dir
    fo.config.model_zoo_dir = os.path.join(fiftyone_dir, "__models__")
    if IsOnCIEnvvar.get():
        fo.config.show_progress_bars = False


class UnfetchableDatasetError(Exception):
    def __init__(self, dataset_name: str, installation_steps: list[str] | None) -> None:
        """
        Create an error for datasets that cannot be automatically fetched in code.
        These datasets often require a login, license agreement acceptance, etc., to download.

        Parameters
        ----------
        dataset_name
            The name of the dataset being fetched.

        installation_steps
            Steps required for a 3rd party user to install this dataset manually.
            If None, the dataset is assumed to be not publicly available.
        """
        self.dataset_name = dataset_name
        self.installation_steps = installation_steps
        if installation_steps is None:
            super().__init__(
                f"Dataset {dataset_name} is for Qualcomm-internal usage only. If you have reached this error message when running an export or evaluate script, please file an issue at https://github.com/quic/ai-hub-models/issues."
            )
        else:
            super().__init__(
                f"To use dataset {dataset_name}, you must download it manually. Follow these steps:\n"
                + "\n".join(
                    [f"{i + 1}. {step}" for i, step in enumerate(installation_steps)]
                )
            )
