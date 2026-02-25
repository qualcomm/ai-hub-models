# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tempfile

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
        download_carvana_files,
    )
except ImportError:
    download_carvana_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

CARVANA_VERSION = 1
CARVANA_DATASET_ID = "carvana"
IMAGES_DIR_NAME = "train"
GT_DIR_NAME = "train_masks"


class CarvanaDataset(BaseDataset):
    """Wrapper class around carvana dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_images_zip: str | None = None,
        input_gt_zip: str | None = None,
    ) -> None:
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            CARVANA_DATASET_ID, CARVANA_VERSION, "data"
        )
        self.images_path = self.data_path / IMAGES_DIR_NAME
        self.gt_path = self.data_path / GT_DIR_NAME
        self.input_images_zip = input_images_zip
        self.input_gt_zip = input_gt_zip

        BaseDataset.__init__(self, self.data_path, split=split)

        self.input_height = 640
        self.input_width = 1280

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor : torch.Tensor
            Normalized image tensor [C, H, W]
        mask_tensor : torch.Tensor
            Binary mask tensor [H, W] (0=background, 1=car)
        """
        orig_image = Image.open(self.images[index]).convert("RGB")
        image = orig_image.resize((self.input_width, self.input_height), Image.BILINEAR)

        _, img_tensor = app_to_net_image_inputs(image)
        img_tensor = img_tensor.squeeze(0)

        # Load and process mask
        orig_mask = Image.open(self.masks[index])
        mask = orig_mask.resize((self.input_width, self.input_height), Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float()

        return img_tensor, mask_tensor

    def __len__(self) -> int:
        return len(self.images)

    def _validate_data(self) -> bool:
        if not self.images_path.exists() or not self.gt_path.exists():
            return False
        self.im_ids = []
        self.images = []
        self.masks = []
        # Match images with their corresponding masks
        for image_path in sorted(self.images_path.glob("*.jpg")):
            im_id = image_path.stem
            mask_path = self.gt_path / f"{im_id}_mask.gif"
            if mask_path.exists():
                self.im_ids.append(im_id)
                self.images.append(image_path)
                self.masks.append(mask_path)

        if not self.images:
            raise ValueError(
                f"No valid image-mask pairs found in {self.images_path} and {self.gt_path}"
            )

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
        if images_zip is None and download_carvana_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                images_zip = os.path.join(tmpdir, f"{IMAGES_DIR_NAME}.zip")
                gt_zip = os.path.join(tmpdir, f"{GT_DIR_NAME}.zip")
                download_carvana_files(images_zip, gt_zip, CARVANA_VERSION)
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
                    "Go to https://www.kaggle.com/c/carvana-image-masking-challenge and make an account",
                    "Go to https://www.kaggle.com/c/carvana-image-masking-challenge/data and download `train.zip` and `train_masks.zip`",
                    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset carvana --files /path/to/train.zip ",
                ],
            )

        os.makedirs(self.images_path.parent, exist_ok=True)
        extract_zip_file(images_zip, self.images_path.parent)
        extract_zip_file(gt_zip, self.gt_path.parent)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/competitions/carvana-image-masking-challenge",
            split_description="train split",
        )
