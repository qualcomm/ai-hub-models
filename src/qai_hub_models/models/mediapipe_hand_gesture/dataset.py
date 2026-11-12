# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import pandas as pd
import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_image
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec

DATASET_ID = "hagrid"
DATASET_VERSION = 2
IMAGES_DIR_NAME = "images"  # full frames (palm detector)
ROI_DIR_NAME = "rois"  # cropped hand images (landmarks)
ANNOTATIONS_NAME = "landmarks_dump.csv"

HAGRID_CLEAN_ASSET = CachedWebDatasetAsset.from_asset_store(
    DATASET_ID,
    DATASET_VERSION,
    "data.zip",
)


# ===========================================================
# Palm detection dataset (shared base)
# ===========================================================
class PalmDetectorDataset(BaseDataset):
    """
    Wrapper class for mediapipe palm detection dataset

    https://huggingface.co/datasets/cj-mills/hagrid-sample-500k-384p
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.data_path = HAGRID_CLEAN_ASSET.extracted_path
        self.images_path = self.data_path / IMAGES_DIR_NAME
        self.roi_path = self.data_path / ROI_DIR_NAME
        self.annotations_path = self.data_path / ANNOTATIONS_NAME
        if input_spec is not None:
            self.input_height = input_spec["image"][0][2]
            self.input_width = input_spec["image"][0][3]
        else:
            self.input_height = 256
            self.input_width = 256
        BaseDataset.__init__(self, self.data_path, split)
        # Load annotations CSV
        self.annotations_db = self._load_annotations_db()

    # ---- shared helpers ----
    def _load_annotations_db(self) -> pd.DataFrame:
        df = pd.read_csv(str(self.annotations_path))
        required_cols = ["source_image_name", "roi_filename", "lr"] + [
            f"lm_{i}_{a}" for i in range(21) for a in ("x", "y", "z")
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"CSV missing required columns: {missing}")
        return df

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load a full-frame image by index, resize, and convert to tensor.

        Parameters
        ----------
        idx
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor : torch.Tensor
            Image tensor of shape (C, H, W), where:
            C = number of channels (3 for RGB)
            H = input_height (256)
            W = input_width (256)
        label : int
            Placeholder label (currently always 0).
        """
        row = self.annotations_db.iloc[idx]
        img_fn = str(row["source_image_name"]).strip()
        img_path = os.path.join(self.images_path, img_fn)
        image = load_image(img_path)
        image = image.resize((self.input_width, self.input_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return (image_tensor, 0)

    def __len__(self) -> int:
        try:
            return len(self.annotations_db)
        except Exception:
            return 0

    def _download_data(self) -> None:
        """Download and set up dataset."""
        HAGRID_CLEAN_ASSET.fetch(extract=True)

    def _validate_data(self) -> bool:
        return self.images_path.exists() and len(os.listdir(self.images_path)) >= 100

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "hagrid_palmdetector"
