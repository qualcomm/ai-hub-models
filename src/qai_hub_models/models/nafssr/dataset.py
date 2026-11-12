# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
    resize_pad,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

FLICKR_FOLDER_NAME = "flickr1024"
FLICKR_VERSION = 1
FLICKR_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{FLICKR_FOLDER_NAME}/v{FLICKR_VERSION}/Flickr1024.zip",
    FLICKR_FOLDER_NAME,
    FLICKR_VERSION,
    "Flickr1024.zip",
    installation_steps=[
        "Download the Flickr1024.zip file from https://drive.google.com/file/d/1LQDUclNtNZWTT41NndISLGvjvuBbxeUs/view",
    ],
    configure_files=["Flickr1024.zip"],
)


class Flickr1024Dataset(BaseDataset):
    """Flickr1024 (Flickr Stereo Image Dataset for Super Resolution)"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        input_data_zip: str | None = None,
        scaling_factor: int = 2,
    ) -> None:
        self.images_path = FLICKR_PRIVATE_ASSET.extracted_path

        if scaling_factor not in (2, 4):
            raise ValueError(f"scaling_factor must be 2 or 4, got {scaling_factor!r}")

        self.scaling_factor = scaling_factor
        self.gt_folder = str(self.images_path / "hr")
        self.lq_folder = str(self.images_path / f"lr_x{self.scaling_factor}")
        self.input_data_zip = input_data_zip

        # BaseDataset.__init__ triggers download_data() which calls _download_data()
        # to fetch and extract the zip. lq_files/gt_files must be populated after.
        BaseDataset.__init__(self, self.images_path, split, input_spec)

        self.lq_files = sorted(os.listdir(self.lq_folder))
        self.gt_files = sorted(os.listdir(self.gt_folder))

        input_spec = input_spec or {"l_image": TensorSpec(shape=(1, 3, 128, 128))}
        self.target_h = input_spec["l_image"][0][2]
        self.target_w = input_spec["l_image"][0][3]

    def __len__(self) -> int:
        return len(self.gt_files)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        gt_path_l = os.path.join(self.gt_folder, self.gt_files[index], "hr0.png")
        gt_path_r = os.path.join(self.gt_folder, self.gt_files[index], "hr1.png")

        img_bytes = load_image(gt_path_l)
        img_gt_l = preprocess_PIL_image(img_bytes, to_float=True)

        img_bytes = load_image(gt_path_r)
        img_gt_r = preprocess_PIL_image(img_bytes, to_float=True)

        lq_path_l = os.path.join(self.lq_folder, self.lq_files[index], "lr0.png")
        lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], "lr1.png")

        img_bytes = load_image(lq_path_l)
        img_lq_l = preprocess_PIL_image(img_bytes, to_float=True)

        img_bytes = load_image(lq_path_R)
        img_lq_r = preprocess_PIL_image(img_bytes, to_float=True)

        resized_lq_l_tensor = resize_pad(img_lq_l, (self.target_h, self.target_w))[
            0
        ].squeeze(0)

        resized_lq_r_tensor = resize_pad(img_lq_r, (self.target_h, self.target_w))[
            0
        ].squeeze(0)

        resized_gt_l_tensor = resize_pad(
            img_gt_l,
            (
                self.target_h * self.scaling_factor,
                self.target_w * self.scaling_factor,
            ),
        )[0].squeeze(0)

        resized_gt_r_tensor = resize_pad(
            img_gt_r,
            (
                self.target_h * self.scaling_factor,
                self.target_w * self.scaling_factor,
            ),
        )[0].squeeze(0)

        return (resized_lq_l_tensor, resized_lq_r_tensor), (
            resized_gt_l_tensor,
            resized_gt_r_tensor,
        )

    def _download_data(self) -> None:
        FLICKR_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        cls(input_data_zip=str(files[0]))

    def _validate_data(self) -> bool:
        # Only check folder existence here; lq_files/gt_files are populated
        # after BaseDataset.__init__ completes (i.e., after download).
        if not self.images_path.exists():
            return False
        # Fast path: expected structure (zip created with relative paths)
        if os.path.exists(self.gt_folder) and os.path.exists(self.lq_folder):
            return True
        # Fallback: search recursively for the hr/ directory.
        # Handles zips created with absolute paths (e.g. zip /abs/path/Flickr1024/).
        for hr_dir in sorted(self.images_path.rglob("hr")):
            lq_candidate = hr_dir.parent / f"lr_x{self.scaling_factor}"
            if hr_dir.is_dir() and lq_candidate.exists():
                self.gt_folder = str(hr_dir)
                self.lq_folder = str(lq_candidate)
                return True
        return False

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://yingqianwang.github.io/Flickr1024/",
            split_description="Flickr1024 dataset for stereo image super resolution",
        )
