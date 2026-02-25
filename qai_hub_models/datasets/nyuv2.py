# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, cast

import cv2
import h5py
import numpy as np
import numpy.typing as npt
import torch
from scipy.io import loadmat

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_nyuv2_files,
    )
except ImportError:
    download_nyuv2_files = None  # type: ignore[assignment]
from qai_hub_models.utils.input_spec import InputSpec

NYUV2_FOLDER_NAME = "nyuv2"
FILE_NAME = "nyu_depth_v2_labeled.mat"
NYUV2_VERSION = 1
SPLIT_ASSET = CachedWebDatasetAsset.from_asset_store(
    NYUV2_FOLDER_NAME,
    NYUV2_VERSION,
    "splits.mat",
)


class NyUv2Dataset(BaseDataset):
    """Wrapper class around NYU_depth_v2 dataset https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html"""

    def __init__(
        self,
        input_spec: InputSpec | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_samples: int = -1,
        source_dataset_file: str | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.dataset_path = SPLIT_ASSET.path().parent / FILE_NAME
        self.source_dataset_file = source_dataset_file
        BaseDataset.__init__(self, str(self.dataset_path), split)
        assert self.split_str in ["train", "val"]

        mat = loadmat(SPLIT_ASSET.fetch())
        f = h5py.File(str(self.dataset_path))

        if self.split_str == "train":
            indices = [ind[0] - 1 for ind in mat["trainNdxs"]]
        elif self.split_str == "val":
            indices = [ind[0] - 1 for ind in mat["testNdxs"]]
        else:
            raise ValueError(f"Split {self.split_str} not found.")

        self.image_list: list[npt.NDArray[np.int8]] = []
        self.depth_list: list[npt.NDArray[np.floating[Any]]] = []
        images = cast(list[npt.NDArray[np.int8]], f["images"])
        depths = cast(list[npt.NDArray[np.floating[Any]]], f["rawDepths"])
        for ind in indices:
            self.image_list.append(np.swapaxes(images[ind], 0, 2))
            self.depth_list.append(np.swapaxes(depths[ind], 0, 1))
            if len(self.image_list) == num_samples:
                break

        input_spec = input_spec or {"image": ((1, 3, 256, 256), "")}
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]

    def _validate_data(self) -> bool:
        """Validates data downloaded on disk. By default just checks that folder exists."""
        try:
            with h5py.File(str(self.dataset_path)) as f:
                images = cast(list[npt.NDArray[np.int8]], f["images"])
                depths = cast(list[npt.NDArray[np.floating[Any]]], f["rawDepths"])
                assert len(images) == 1449
                assert len(depths) == 1449
        except Exception:  # Failed to load data
            return False

        return self.dataset_path.exists()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # image
        image = self.image_list[index]
        scaled_image = image / 255

        # depth
        depth = self.depth_list[index]

        # sample
        resized_image = cv2.resize(
            scaled_image,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        )

        image_tensor = torch.tensor(resized_image).to(torch.float32).permute(2, 0, 1)
        target = torch.tensor(depth).to(torch.float32)
        return image_tensor, target

    def __len__(self) -> int:
        return len(self.image_list)

    def _download_data(self, source_file: str | None = None) -> None:
        if self.dataset_path.exists():
            return

        # Use passed arg if provided, otherwise use instance attribute
        if source_file is None:
            source_file = self.source_dataset_file

        # If no file provided/set, try auto-download
        if source_file is None and download_nyuv2_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                source_file = os.path.join(tmpdir, FILE_NAME)
                download_nyuv2_files(source_file)
                self._download_data(source_file)
            return

        if source_file is None or not source_file.endswith(FILE_NAME):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=[
                    "Download the dataset from https://www.kaggle.com/datasets/rmzhang0526/nyu-depth-v2-labeled",
                    f"Run `python -m qai_hub_models.datasets.configure_dataset --dataset nyuv2 --files /path/to/{FILE_NAME}`",
                ],
            )

        os.makedirs(self.dataset_path.parent, exist_ok=True)
        shutil.copy(source_file, self.dataset_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html",
            split_description="validation split",
        )
