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
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_foot_track_files,
    )
except ImportError:
    download_foot_track_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

FOOTTRACK_DATASET_VERSION = 3
FOOTTRACK_DATASET_ID = "foottrack_dataset"
FOOTTRACK_DATASET_DIR_NAME = "foottrackv3_trainvaltest"

CLASS_STR2IDX = {"face": "0", "person": "1", "hand": "2"}


class FootTrackDataset(BaseDataset):
    """Wrapper class for foot_track_net private dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_data_zip: str | None = None,
        max_boxes: int = 100,
    ) -> None:
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            FOOTTRACK_DATASET_ID, FOOTTRACK_DATASET_VERSION, "data"
        )
        self.images_path = self.data_path / FOOTTRACK_DATASET_DIR_NAME
        self.gt_path = self.data_path / FOOTTRACK_DATASET_DIR_NAME

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

    def _download_data(self, zip_path: str | None = None) -> None:
        # Use passed arg if provided, otherwise use instance attribute
        if zip_path is None:
            zip_path = self.input_data_zip

        # If no file provided/set, try auto-download
        if zip_path is None and download_foot_track_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, f"{FOOTTRACK_DATASET_DIR_NAME}.zip")
                download_foot_track_files(zip_path)
                self._download_data(zip_path)
            return

        if zip_path is None or not zip_path.endswith(
            FOOTTRACK_DATASET_DIR_NAME + ".zip"
        ):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=None,
            )

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(zip_path, self.images_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000
