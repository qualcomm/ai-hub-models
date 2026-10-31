# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
import shutil
import subprocess

import torch
from torchvision.datasets import ImageNet

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.image_processing import IMAGENET_TRANSFORM

IMAGENET_FOLDER_NAME = "imagenet"
IMAGENET_VERSION = 2

IMAGENET_ASSET = CachedWebDatasetAsset(
    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    IMAGENET_FOLDER_NAME,
    IMAGENET_VERSION,
    "ILSVRC2012_img_val.tar",
    private_s3_key="qai-hub-models/datasets/imagenet/ILSVRC2012_img_val.tar",
)
DEVKIT_NAME = "ILSVRC2012_devkit_t12.tar.gz"
DEVKIT_ASSET = CachedWebDatasetAsset(
    f"https://image-net.org/data/ILSVRC/2012/{DEVKIT_NAME}",
    IMAGENET_FOLDER_NAME,
    IMAGENET_VERSION,
    DEVKIT_NAME,
    private_s3_key="qai-hub-models/datasets/imagenet/ILSVRC2012_devkit_t12.tar.gz",
)
VAL_PREP_ASSET = CachedWebDatasetAsset(
    "https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh",
    IMAGENET_FOLDER_NAME,
    IMAGENET_VERSION,
    "valprep.sh",
)


class ImagenetDataset(BaseDataset, ImageNet):
    """Wrapper class for using the Imagenet validation dataset: https://www.image-net.org/"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        transform: object = IMAGENET_TRANSFORM,
    ) -> None:
        """
        A direct download link for the validation set is not available.
        Users should download the validation dataset manually and pass the local filepath
        as an argument here. After this is done once, it will be symlinked to an
        internal location and doesn't need to be passed again.

        input_data_path: Local filepath to imagenet validation set.
        """
        if split != DatasetSplit.VAL:
            raise ValueError("Imagenet dataset currently only supports `val` split")
        BaseDataset.__init__(self, DEVKIT_ASSET.extracted_path.parent, split)
        ImageNet.__init__(
            self,
            root=str(self.dataset_path),
            split=self.split_str,
            transform=transform,
        )

    def _validate_data(self) -> bool:
        val_path = self.dataset_path / self.split_str
        if not (self.dataset_path / DEVKIT_NAME).exists():
            print("Missing Devkit.")
            return False

        if not val_path.exists():
            print("Missing images.")
            return False

        subdirs = [filepath for filepath in val_path.iterdir() if filepath.is_dir()]
        if len(subdirs) != 1000:
            print(f"Expected 1000 subdirectories but got {len(subdirs)}")
            return False

        total_images = 0
        for subdir in subdirs:
            total_images += len(list(subdir.iterdir()))

        if total_images != 50000:
            print(f"Expected 50000 images but got {total_images}")
            return False
        return True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return ImageNet.__len__(self)

    def _download_data(self) -> None:
        # Fetch data
        IMAGENET_ASSET.fetch(extract=True)
        DEVKIT_ASSET.fetch()
        VAL_PREP_ASSET.fetch()

        # Prep images
        subprocess.call(
            f"sh {VAL_PREP_ASSET.path}", shell=True, cwd=IMAGENET_ASSET.extracted_path
        )

        # Move images to <root>/val
        dst_folder = self.dataset_path / self.split_str
        if not dst_folder.exists():
            if os.name == "nt":
                shutil.move(IMAGENET_ASSET.extracted_path, dst_folder)
                # Leave breadcrumbs so the CachedWebDatasetAsset won't try to re-download the imagenet devkit
                IMAGENET_ASSET.extracted_path.mkdir()
                (IMAGENET_ASSET.extracted_path / "exists.txt").touch()
            else:
                os.symlink(IMAGENET_ASSET.extracted_path, dst_folder)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 2500

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.image-net.org/",
            split_description="validation split",
        )
