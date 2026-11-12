# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from hdf5storage import loadmat
from numpy.typing import NDArray

from qai_hub_models.datasets.coco import CocoBodyDatasetBase
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_cs,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

COFW_VERSION = 2

# originally from the https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms
COFW_ASSET = CachedWebDatasetAsset.from_asset_store(
    "cofw",
    COFW_VERSION,
    "cofw.zip",
)


class COFWDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, 256, 256), dtype="float32")
        }
        self.target_h = int(input_spec["image"][0][2])
        self.target_w = int(input_spec["image"][0][3])

        self.dataset_path = COFW_ASSET.extracted_path
        BaseDataset.__init__(self, self.dataset_path, split)

        # Select .mat file based on split
        if split == DatasetSplit.TRAIN:
            self.mat_path = self.dataset_path / "COFW_train_color.mat"
            img_key, pts_key = "IsTr", "phisTr"
        else:
            self.mat_path = self.dataset_path / "COFW_test_color.mat"
            img_key, pts_key = "IsT", "phisT"

        # Check if .mat file exists
        if not self.mat_path.exists():
            raise FileNotFoundError(f"COFW .mat not found: {self.mat_path}.")
        self.mat = loadmat(str(self.mat_path))
        self.images = self.mat[img_key]
        self.pts = self.mat[pts_key]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, idx: int
    ) -> tuple[
        torch.Tensor,
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    ]:
        """
        Retrieve a preprocessed image and ground truth at index.

        Parameters
        ----------
        idx
            Index of the dataset item.

        Returns
        -------
        image : torch.Tensor
            Input image resized for the network. RGB, floating point range [0-1].

        ground_truth : tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
            center
                Bounding box center (x, y).
            scale
                Bounding box scale (w, h).
            pts
                Landmarks (29, 2) as (x, y).
        """
        # Load image
        img = self.images[idx][0]
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        # Load and reshape landmarks
        pts = np.array(self.pts[idx][0:58]).reshape(2, -1).T.astype(np.float32)

        (xmin, ymin), (xmax, ymax) = pts.min(0), pts.max(0)
        center, scale = box_xywh_to_cs(
            [xmin, ymin, xmax - xmin, ymax - ymin],
            self.target_w / self.target_h,
            padding_factor=1.25,
        )

        # Preprocess image with affine warping
        image_tensor = pre_process_with_affine(
            img, center, scale, 0, (self.target_h, self.target_w)
        ).squeeze(0)

        return image_tensor, (center, scale, pts)

    def _validate_data(self) -> bool:
        """Validate dataset by checking if the dataset directory and .mat file exist."""
        return self.dataset_path.exists()

    def _download_data(self) -> None:
        COFW_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 507

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """
        Return metadata for the COFW dataset.

        Returns
        -------
        DatasetMetadata
            Contains dataset URL and split description (train or test).
        """
        return DatasetMetadata(
            link="https://1drv.ms/u/s!AiWjZ1LamlxzdmYbSkHpPYhI8Ms",
            split_description="COFW train or test split",
        )


class CocoFaceCalibrationDataset(CocoBodyDatasetBase):
    """COCO-WholeBody face crops used only for HRNet-Face calibration."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        num_samples: int = 100,
    ) -> None:
        if split != DatasetSplit.TRAIN:
            raise ValueError("CocoFaceCalibrationDataset only supports TRAIN.")
        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, 256, 256), dtype="float32")
        }
        super().__init__(DatasetSplit.VAL, input_spec, num_samples)

    def _load_kpt_db(self) -> list[tuple[Path, np.ndarray, np.ndarray]]:
        face_db: list[tuple[Path, np.ndarray, np.ndarray]] = []
        ratio = self.target_w / self.target_h
        for img_id in self.img_ids:
            img_info = self.cocoGt.loadImgs(img_id)[0]
            ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
            for annotation in self.cocoGt.loadAnns(ann_ids):
                if not annotation.get("face_valid", False):
                    continue
                x, y, width, height = annotation["face_box"]
                if width <= 0 or height <= 0 or x < 0 or y < 0:
                    continue
                img_width, img_height = img_info["width"], img_info["height"]
                x2 = min(img_width - 1, x + width)
                y2 = min(img_height - 1, y + height)
                if x2 <= x or y2 <= y:
                    continue
                center, scale = box_xywh_to_cs(
                    [x, y, x2 - x, y2 - y], ratio, padding_factor=1.25
                )
                face_db.append((self.image_dir / img_info["file_name"], center, scale))
                break
            if self.samples > 0 and len(face_db) >= self.samples:
                break
        return face_db

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[()]]:
        image_path, center, scale = self.kpt_db[index]
        image = cv2.imread(
            str(image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
        assert image is not None, f"Image not found at {image_path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = pre_process_with_affine(
            image, center, scale, 0, (self.target_h, self.target_w)
        ).squeeze(0)
        return image_tensor, ()

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/jin-s13/COCO-WholeBody",
            split_description="COCO-WholeBody val2017 samples with valid face boxes",
        )
