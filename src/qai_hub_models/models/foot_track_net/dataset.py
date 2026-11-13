# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

FOOTTRACK_DATASET_VERSION = 4
FOOTTRACK_DATASET_ID = "foottrack_dataset"
FOOTTRACK_DATASET_DIR_NAME = "foottrackv3_trainvaltest"

FOOTTRACK_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/foottrack/foottrackv3_trainvaltest.zip",
    FOOTTRACK_DATASET_ID,
    FOOTTRACK_DATASET_VERSION,
    f"data/{FOOTTRACK_DATASET_DIR_NAME}.zip",
    configure_files=[f"{FOOTTRACK_DATASET_DIR_NAME}.zip"],
)

CLASS_STR2IDX = {"face": "0", "person": "1", "hand": "2"}


class FootTrackDataset(BaseDataset):
    """Wrapper class for foot_track_net private dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        max_boxes: int = 100,
    ) -> None:
        self.data_path = FOOTTRACK_PRIVATE_ASSET.extracted_path
        self.images_path = self.data_path
        self.gt_path = self.data_path

        self.input_data_zip = input_data_zip
        self.max_boxes = max_boxes

        self.img_width = 640
        self.img_height = 480
        self.scale_width = 1.0 / self.img_width
        self.scale_height = 1.0 / self.img_height
        BaseDataset.__init__(self, self.data_path, split=split)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        image_path = self.image_list[index]
        gt_path = self.gt_list[index]
        image = Image.open(image_path)
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)

        labels_gt = np.genfromtxt(gt_path, delimiter=" ", dtype="str")
        for key, value in CLASS_STR2IDX.items():
            labels_gt = np.char.replace(labels_gt, key, value)
        labels_gt = labels_gt.astype(np.float32)
        labels_gt = np.reshape(labels_gt, (-1, 5))

        boxes = torch.tensor(labels_gt[:, 1:5])
        labels = torch.tensor(labels_gt[:, 0])

        # Pad the number of boxes to a standard value
        num_boxes = len(labels)
        if num_boxes == 0:
            boxes = torch.zeros((100, 4))
            labels = torch.zeros(100)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
            labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)

        image_id = abs(hash(str(image_path.name[:-4]))) % (10**8)

        return image_tensor, (
            image_id,
            self.img_height,
            self.img_width,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )

    def __len__(self) -> int:
        return len(self.image_list)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False

        self.images_path = self.images_path / "images" / self.split_str
        self.gt_path = self.gt_path / "labels" / self.split_str
        self.image_list: list[Path] = []
        self.gt_list: list[Path] = []
        for img_path in self.images_path.iterdir():
            if Image.open(img_path).size != (self.img_width, self.img_height):
                raise ValueError(Image.open(img_path).size)
            gt_filename = img_path.name.replace(".jpg", ".txt")
            gt_path = self.gt_path / gt_filename
            if not gt_path.exists():
                print(f"Ground truth file not found: {gt_path!s}")
                return False
            self.image_list.append(img_path)
            self.gt_list.append(gt_path)
        return True

    def _download_data(self) -> None:
        FOOTTRACK_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

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
