# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
from torchvision.datasets import ImageFolder

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

DATASET_VERSION = 2
DATASET_ID = "human_faces_dataset"

HUMAN_FACES_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/human_faces/faces.zip",
    DATASET_ID,
    DATASET_VERSION,
    "data.zip",
    installation_steps=[
        "Download the dataset from https://www.kaggle.com/datasets/ashwingupta3012/human-face",
        "Rename or copy the download so it is named human_faces.zip, e.g.:\n"
        "    cp archive.zip human_faces.zip",
    ],
    local_cache_extracted_path="data/Humans",
    configure_files=["human_faces.zip"],
)


class HumanFacesDataset(BaseDataset):
    """
    Wrapper class for human faces dataset

    https://www.kaggle.com/datasets/ashwingupta3012/human-faces
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.images_path = HUMAN_FACES_PRIVATE_ASSET.extracted_path
        self.data_path = self.images_path.parent
        self.input_data_zip = input_data_zip

        if input_spec is not None:
            self.img_height = input_spec["image"][0][2]
            self.img_width = input_spec["image"][0][3]
        else:
            self.img_height = 256
            self.img_width = 256
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

    def _download_data(self) -> None:
        HUMAN_FACES_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        cls(input_data_zip=str(files[0]))

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000
