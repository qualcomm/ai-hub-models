# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile

import torch
from torchvision.datasets import ImageFolder

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_human_faces_files,
    )
except ImportError:
    download_human_faces_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

DATASET_VERSION = 1
DATASET_ID = "human_faces_dataset"
DATASET_DIR_NAME = "Humans"


class HumanFacesDataset(BaseDataset):
    """
    Wrapper class for human faces dataset

    https://www.kaggle.com/datasets/ashwingupta3012/human-faces
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        width: int = 256,
        height: int = 256,
    ) -> None:
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            DATASET_ID, DATASET_VERSION, "data"
        )
        self.images_path = self.data_path / DATASET_DIR_NAME
        self.input_data_zip = input_data_zip

        self.img_width = width
        self.img_height = height
        self.scale_width = 1.0 / self.img_width
        self.scale_height = 1.0 / self.img_height
        BaseDataset.__init__(self, self.data_path, split=split)
        self.dataset = ImageFolder(str(self.data_path))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, _ = self.dataset[index]
        image = image.resize((self.img_width, self.img_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, 0

    def __len__(self) -> int:
        return len(self.dataset)

    def _validate_data(self) -> bool:
        return self.images_path.exists() and len(os.listdir(self.images_path)) >= 100

    def _download_data(self, zip_path: str | None = None) -> None:
        # Use passed arg if provided, otherwise use instance attribute
        if zip_path is None:
            zip_path = self.input_data_zip

        # If no file provided/set, try auto-download
        if zip_path is None and download_human_faces_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "faces.zip")
                download_human_faces_files(zip_path)
                self._download_data(zip_path)
            return

        if zip_path is None or not zip_path.endswith(".zip"):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=[
                    "Download the dataset from https://www.kaggle.com/datasets/ashwingupta3012/human-face",
                    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset human_faces --files /path/to/zip`",
                ],
            )

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(zip_path, self.data_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000


class HumanFaces192Dataset(HumanFacesDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
    ) -> None:
        super().__init__(split, input_data_zip, 192, 192)

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "human_faces_192"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/datasets/ashwingupta3012/human-faces",
            split_description="validation split",
        )
