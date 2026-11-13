# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

CITYSCAPES_VERSION = 2
CITYSCAPES_DATASET_ID = "cityscapes"

CITYSCAPES_INSTALLATION_STEPS = [
    "Go to https://www.cityscapes-dataset.com/ and make an account",
    "Go to https://www.cityscapes-dataset.com/downloads/ and download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`",
]

CITYSCAPES_CONFIGURE_FILES = [
    "leftImg8bit_trainvaltest.zip",
    "gtFine_trainvaltest.zip",
]

CITYSCAPES_IMAGES_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/cityscapes/partial_leftImg8bit_trainvaltest.zip",
    CITYSCAPES_DATASET_ID,
    CITYSCAPES_VERSION,
    "data/leftImg8bit_trainvaltest.zip",
    installation_steps=CITYSCAPES_INSTALLATION_STEPS,
    configure_files=CITYSCAPES_CONFIGURE_FILES,
)

CITYSCAPES_GT_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/cityscapes/gtFine_trainvaltest.zip",
    CITYSCAPES_DATASET_ID,
    CITYSCAPES_VERSION,
    "data/gtFine_trainvaltest.zip",
    installation_steps=CITYSCAPES_INSTALLATION_STEPS,
    configure_files=CITYSCAPES_CONFIGURE_FILES,
)

# Map dataset class ids to model class ids
# https://github.com/mcordts/cityscapesScripts/blob/9f0aa8d3fa937c42bd5f21e0180a6546f077539f/cityscapesscripts/helpers/labels.py#L62
CLASS_MAP = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}

HEIGHT = 1024
WIDTH = 2048


def class_map_lookup(key: int) -> int:
    return CLASS_MAP.get(key, -1)


class CityscapesDataset(BaseDataset):
    """Wrapper class around Cityscapes dataset https://www.cityscapes-dataset.com/"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_images_zip: str | None = None,
        input_gt_zip: str | None = None,
        make_lowres: bool = False,
    ) -> None:
        self.images_path = CITYSCAPES_IMAGES_ASSET.extracted_path
        self.gt_path = CITYSCAPES_GT_ASSET.extracted_path

        self.input_images_zip = input_images_zip
        self.input_gt_zip = input_gt_zip
        self.make_lowres = make_lowres
        BaseDataset.__init__(self, self.images_path.parent, split=split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(image_path)
        gt_img = Image.open(gt_path)
        if self.make_lowres:
            new_size = (WIDTH // 2, HEIGHT // 2)
            image = image.resize(new_size)
        gt = np.vectorize(class_map_lookup)(np.array(gt_img))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, torch.tensor(gt)

    def __len__(self) -> int:
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / self.split_str
        self.gt_path = self.gt_path / "gtFine" / self.split_str
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        img_count = 0
        # Sort by path name to ensure deterministic ordering
        for subdir in sorted(self.images_path.iterdir(), key=lambda item: item.name):
            for img_path in sorted(subdir.iterdir(), key=lambda item: item.name):
                if not img_path.name.endswith("leftImg8bit.png"):
                    print(f"Invalid file: {img_path!s}")
                    return False
                if Image.open(img_path).size != (WIDTH, HEIGHT):
                    raise ValueError(Image.open(img_path).size)
                img_count += 1
                gt_filename = img_path.name.replace(
                    "leftImg8bit.png", "gtFine_labelIds.png"
                )
                gt_path = self.gt_path / subdir.name / gt_filename
                if not gt_path.exists():
                    print(f"Ground truth file not found: {gt_path!s}")
                    return False
                self.image_list.append(img_path)
                self.gt_list.append(gt_path)
        return True

    def _download_data(self) -> None:
        CITYSCAPES_IMAGES_ASSET.fetch(extract=True, local_path=self.input_images_zip)
        CITYSCAPES_GT_ASSET.fetch(extract=True, local_path=self.input_gt_zip)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 2:
            raise ValueError(
                f"{cls.__name__}.configure expects 2 file(s), got {len(files)}."
            )
        cls(input_images_zip=str(files[0]), input_gt_zip=str(files[1]))

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.cityscapes-dataset.com/",
            split_description="validation split",
        )
