# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_cityscapes_files,
    )
except ImportError:
    download_cityscapes_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

CITYSCAPES_VERSION = 1
CITYSCAPES_DATASET_ID = "cityscapes"
IMAGES_DIR_NAME = "leftImg8bit_trainvaltest"
GT_DIR_NAME = "gtFine_trainvaltest"

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
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            CITYSCAPES_DATASET_ID, CITYSCAPES_VERSION, "data"
        )
        self.images_path = self.data_path / IMAGES_DIR_NAME
        self.gt_path = self.data_path / GT_DIR_NAME

        self.input_images_zip = input_images_zip
        self.input_gt_zip = input_gt_zip
        self.make_lowres = make_lowres
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(image_path)
        gt_img = Image.open(gt_path)
        if self.make_lowres:
            new_size = (WIDTH // 2, HEIGHT // 2)
            image = image.resize(new_size)
            gt_img = gt_img.resize(new_size)
        gt = np.vectorize(class_map_lookup)(np.array(gt_img))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, torch.tensor(gt)

    def __len__(self) -> int:
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / "leftImg8bit" / self.split_str
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

    def _download_data(
        self, images_zip: str | None = None, gt_zip: str | None = None
    ) -> None:
        # Use passed args if provided, otherwise use instance attributes
        if images_zip is None:
            images_zip = self.input_images_zip
        if gt_zip is None:
            gt_zip = self.input_gt_zip

        # If no files provided/set, try auto-download
        if images_zip is None and download_cityscapes_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                images_zip = os.path.join(tmpdir, f"{IMAGES_DIR_NAME}.zip")
                gt_zip = os.path.join(tmpdir, f"{GT_DIR_NAME}.zip")
                download_cityscapes_files(images_zip, gt_zip)
                self._download_data(images_zip, gt_zip)
            return

        if (
            images_zip is None
            or not images_zip.endswith(IMAGES_DIR_NAME + ".zip")
            or gt_zip is None
            or not gt_zip.endswith(GT_DIR_NAME + ".zip")
        ):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=[
                    "Go to https://www.cityscapes-dataset.com/ and make an account",
                    "Go to https://www.cityscapes-dataset.com/downloads/ and download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip`",
                    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset cityscapes --files /path/to/leftImg8bit_trainvaltest.zip /path/to/gtFine_trainvaltest.zip`",
                ],
            )

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(images_zip, self.images_path)
        extract_zip_file(gt_zip, self.gt_path)

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
