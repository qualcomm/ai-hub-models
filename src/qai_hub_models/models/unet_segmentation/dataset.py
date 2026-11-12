# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.coco import CocoDetection

from qai_hub_models.datasets.coco.coco import CocoDatasetBase
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

# COCO category names segmented as "vehicle"; collapsed to UNet's single
# foreground class to mirror Carvana's car-vs-background masks.
COCO_VEHICLE_CLASSES = ["car", "bus", "truck"]

CARVANA_VERSION = 2
CARVANA_DATASET_ID = "carvana"
IMAGES_DIR_NAME = "train"
GT_DIR_NAME = "train_masks"

CARVANA_INSTALLATION_STEPS = [
    "Go to https://www.kaggle.com/c/carvana-image-masking-challenge and make an account",
    "Go to https://www.kaggle.com/c/carvana-image-masking-challenge/data and download `train.zip` and `train_masks.zip`",
]

CARVANA_CONFIGURE_FILES = [
    "train.zip",
    "train_masks.zip",
]

CARVANA_IMAGES_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/carvana/v{CARVANA_VERSION}/train.zip",
    CARVANA_DATASET_ID,
    CARVANA_VERSION,
    f"data/{IMAGES_DIR_NAME}.zip",
    installation_steps=CARVANA_INSTALLATION_STEPS,
    configure_files=CARVANA_CONFIGURE_FILES,
)

CARVANA_GT_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/carvana/v{CARVANA_VERSION}/train_masks.zip",
    CARVANA_DATASET_ID,
    CARVANA_VERSION,
    f"data/{GT_DIR_NAME}.zip",
    installation_steps=CARVANA_INSTALLATION_STEPS,
    configure_files=CARVANA_CONFIGURE_FILES,
)


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

    def _download_data(self) -> None:
        CARVANA_IMAGES_ASSET.fetch(extract=True, local_path=self.input_images_zip)
        CARVANA_GT_ASSET.fetch(extract=True, local_path=self.input_gt_zip)

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
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/competitions/carvana-image-masking-challenge",
            split_description="train split",
        )


class UNetCalibrationDataset(CocoDatasetBase):
    """
    Calibration-only dataset for UNet, sourced from MS COCO (CC-BY 4.0).

    Carvana (the eval dataset) is Kaggle-competition licensed and non-commercial,
    so it must not ship as calibration data. COCO's car/bus/truck instance masks
    are collapsed into a single binary vehicle mask, matching UNet's 2-class
    (foreground / background) output and Carvana's `__getitem__` contract.
    Only images containing at least one vehicle are kept.

    Calibration consumes only the image tensor (see utils/quantization.py); the
    mask is returned solely to satisfy the (image, mask) dataloader contract.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        max_train_samples: int = 2000,
    ) -> None:
        # UNet has no train download; calibrate from val2017 (auto-downloadable,
        # no license wall) to keep `qai-hub-models export` self-service.
        super().__init__(
            split=DatasetSplit.VAL,
            input_spec=input_spec,
            max_train_samples=max_train_samples,
        )
        spec = input_spec or {}
        self.input_height = spec["image"][0][2] if "image" in spec else 640
        self.input_width = spec["image"][0][3] if "image" in spec else 1280

        vehicle_cat_ids = set(self.coco.getCatIds(catNms=COCO_VEHICLE_CLASSES))
        self.ids = [
            img_id
            for img_id in self.ids
            if vehicle_cat_ids
            & {
                ann["category_id"]
                for ann in self.coco.loadAnns(
                    self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
                )
            }
        ]
        if not self.ids:
            raise ValueError("No COCO images with vehicle annotations found.")
        self.vehicle_cat_ids = vehicle_cat_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, target = CocoDetection.__getitem__(self, index)
        image = image.convert("RGB").resize(
            (self.input_width, self.input_height), Image.BILINEAR
        )
        _, img_tensor = app_to_net_image_inputs(image)
        img_tensor = img_tensor.squeeze(0)

        mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
        for annotation in target:
            if annotation["category_id"] not in self.vehicle_cat_ids:
                continue
            ann_mask = cv2.resize(
                self.coco.annToMask(annotation),
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = np.maximum(mask, ann_mask)

        return img_tensor, torch.from_numpy(mask).float()

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 vehicle subset (car/bus/truck)",
        )
