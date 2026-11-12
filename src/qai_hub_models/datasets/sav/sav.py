# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn
from torchvision.transforms import Resize

from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.utils.asset_loaders import (
    ASSET_CONFIG,
    load_image,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

SAV_FOLDER_NAME = "sav"
SAV_VERSION = 2
SAV_DIR_NAME = "sav_val"
SEED = 42

SAV_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/sav/sav_val.tar",
    SAV_FOLDER_NAME,
    SAV_VERSION,
    f"{SAV_DIR_NAME}.tar",
    installation_steps=[
        "Download sav_val.tar from https://ai.meta.com/datasets/segment-anything-video-downloads/",
    ],
    configure_files=[f"{SAV_DIR_NAME}.tar"],
)


class SaVDataset(BaseDataset):
    def __init__(
        self,
        input_tar: str | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.input_tar = input_tar
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 1024, 1024))}
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            SAV_FOLDER_NAME, SAV_VERSION, "sav_val"
        )
        BaseDataset.__init__(self, self.data_path, split=split)

        self.sample = []
        images_path = os.path.join(self.data_path, "JPEGImages_24fps")
        for root, _dirs, files in os.walk(images_path):
            if files:
                for file_name in files:
                    image_path = os.path.join(root, file_name)
                    self.sample.append(
                        {
                            "img_path": image_path,
                        }
                    )

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], list[None]]:
        """
        Returns a tuple of input data and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_input : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            input_image:
                Raw floating point pixel values for encoder consumption.
                3-channel Color Space: RGB, range [0, 1]
            point_coords:
                Point coordinates from input image for segmentation,
                mapped to the resized image with shape [N, 2]
            point_labels:
                Point Labels to select/de-select given point for segmentation

        ground_truth : list[None]
            Empty list; no ground truth data.
        """
        image_path = self.sample[index]["img_path"]
        image = np.array(load_image(image_path))
        resize_transform = nn.Sequential(Resize((self.input_height, self.input_width)))
        image_tensor = resize_transform(numpy_image_to_torch(image)).squeeze(0)
        torch.manual_seed(SEED)
        point_coords = torch.rand((2, 2))
        point_labels = torch.ones(2)
        return (image_tensor, point_coords, point_labels), []

    def __len__(self) -> int:
        return len(self.sample)

    def _download_data(self) -> None:
        SAV_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_tar)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        cls(input_tar=str(files[0]))

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100
