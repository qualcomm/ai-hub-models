# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from os import path as osp

import cv2
import lmdb
import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
    resize_pad,
)
from qai_hub_models.models.templates.nafnet.lmdb_helpers import (
    get_image_from_lmdb,
    paired_paths_from_lmdb,
)
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

REDS_FOLDER_NAME = "reds"
REDS_VERSION = 1
REDS_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{REDS_FOLDER_NAME}/v{REDS_VERSION}/REDS-val300-lmdb.zip",
    REDS_FOLDER_NAME,
    REDS_VERSION,
    "REDS-val300-lmdb.zip",
    installation_steps=[
        "Download the REDS-val300-lmdb.zip file from https://drive.google.com/file/d/1_WPxX6mDSzdyigvie_OlpI-Dknz7RHKh/view",
    ],
    configure_files=["REDS-val300-lmdb.zip"],
)


class REDSDataset(BaseDataset):
    """REDS (Realistic and Dynamic Scenes Dataset)"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_data_zip: str | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.images_path = REDS_PRIVATE_ASSET.extracted_path
        self.gt_folder = str(self.images_path / "REDS" / "val" / "sharp_300.lmdb")
        self.lq_folder = str(self.images_path / "REDS" / "val" / "blur_300.lmdb")
        self.input_data_zip = input_data_zip

        BaseDataset.__init__(self, self.images_path, split, input_spec)
        # Initialize LMDB environments and transactions directly
        self.lq_env = lmdb.open(
            self.lq_folder, readonly=True, lock=False, readahead=False, meminit=False
        )
        self.gt_env = lmdb.open(
            self.gt_folder, readonly=True, lock=False, readahead=False, meminit=False
        )

        # Create persistent read transactions
        self.lq_txn = self.lq_env.begin(write=False)
        self.gt_txn = self.gt_env.begin(write=False)

        # Get paired paths from LMDB
        self.paths = paired_paths_from_lmdb(
            [self.lq_folder, self.gt_folder], ["lq", "gt"]
        )

        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 360, 640))}
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        gt_path = self.paths[index]["gt_path"]
        img_gt = get_image_from_lmdb(self.gt_txn, gt_path, "gt", float32=True)

        lq_path = self.paths[index]["lq_path"]
        img_lq = get_image_from_lmdb(self.lq_txn, lq_path, "lq", float32=True)

        img_gt_tensor = numpy_image_to_torch(
            cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB), to_float=False
        )
        img_lq_tensor = numpy_image_to_torch(
            cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB), to_float=False
        )

        resized_gt_tensor, _, _ = resize_pad(
            img_gt_tensor, (self.input_height, self.input_width)
        )

        resized_lq_tensor, _, _ = resize_pad(
            img_lq_tensor, (self.input_height, self.input_width)
        )

        return resized_lq_tensor.squeeze(0), resized_gt_tensor.squeeze(0)

    def __del__(self) -> None:
        """Clean up LMDB resources"""
        if hasattr(self, "lq_txn"):
            self.lq_txn.abort()
        if hasattr(self, "gt_txn"):
            self.gt_txn.abort()
        if hasattr(self, "lq_env"):
            self.lq_env.close()
        if hasattr(self, "gt_env"):
            self.gt_env.close()

    def _validate_data(self) -> bool:
        if not self.images_path.exists():
            return False
        # Fast path: expected structure
        if osp.exists(self.gt_folder) and osp.exists(self.lq_folder):
            return True
        # Fallback: search recursively for the sharp_300.lmdb directory.
        # Handles zips with different top-level directory structures.
        for gt_dir in sorted(self.images_path.rglob("sharp_300.lmdb")):
            lq_candidate = gt_dir.parent / "blur_300.lmdb"
            if gt_dir.is_dir() and lq_candidate.exists():
                self.gt_folder = str(gt_dir)
                self.lq_folder = str(lq_candidate)
                return True
        return False

    def _download_data(self) -> None:
        REDS_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

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
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://seungjunnah.github.io/Datasets/reds.html",
            split_description="REDS dataset for image deblurring",
        )
