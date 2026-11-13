# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import asyncio
import csv
import re
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image

from qai_hub_models.datasets.common import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.models.templates.corevision.image_processing import (
    IMAGENET_TRANSFORM,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, download_file
from qai_hub_models.utils.download import download_many_urls
from qai_hub_models.utils.input_spec import InputSpec

# Open Images V7 images are hosted on the public CVDF S3 mirror, keyed by split
# and image id. We calibrate on the validation split (used here as a "training"
# set for calibration purposes); its full image-id list is fetched from the
# official Open Images metadata bucket at download time, then an evenly-spaced
# subset is pulled from CVDF in parallel.
OPENIMAGES_DATASET_ID = "open_images_v7"
OPENIMAGES_DATASET_ASSET_VERSION = 1
OPENIMAGES_SPLIT = "validation"
OPENIMAGES_CVDF_URL = "https://s3.amazonaws.com/open-images-dataset"
OPENIMAGES_IMAGE_LIST_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
# Bounded, evenly-sampled subset size: calibration only ever needs
# default_num_calibration_samples() (100), so there's no reason to pull the
# full ~41.6k-image validation split.
MAX_DOWNLOAD_IMAGES = 1000
# Open Images ids are 16 hex chars; reject anything else so an unexpected CSV
# row can't steer a write or fetch outside the image id namespace.
_IMAGE_ID_RE = re.compile(r"^[0-9a-fA-F]{16}$")


class OpenImagesV7Dataset(BaseDataset):
    """Open Images V7 validation images for post-training quantization calibration.

    Downloads an evenly-sampled subset of the validation split directly from the
    public CVDF mirror (https://github.com/cvdfoundation/open-images-dataset).
    The validation split is used here as the "training" split (``DatasetSplit.TRAIN``)
    since it's the split fed to calibration, not for held-out evaluation. Labels are
    irrelevant for calibration — only the natural-image input distribution matters —
    so ``__getitem__`` returns a dummy target.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        transform: Callable[..., torch.Tensor] = IMAGENET_TRANSFORM,
    ) -> None:
        if split != DatasetSplit.TRAIN:
            raise ValueError(
                "OpenImagesV7Dataset only supports DatasetSplit.TRAIN: it wraps "
                "the Open Images validation split, but that split is used here as "
                "the calibration (training) set, not for held-out evaluation."
            )
        self.transform = transform
        dataset_path = ASSET_CONFIG.get_local_store_dataset_path(
            OPENIMAGES_DATASET_ID, OPENIMAGES_DATASET_ASSET_VERSION, ""
        )
        BaseDataset.__init__(self, dataset_path, split, input_spec)
        self.image_files = sorted(str(p) for p in self.images_dir.glob("*.jpg"))

    @property
    def images_dir(self) -> Path:
        return self.dataset_path / OPENIMAGES_SPLIT

    def _validate_data(self) -> bool:
        if not self.images_dir.is_dir():
            return False
        num_images = len(list(self.images_dir.glob("*.jpg")))
        return num_images >= MAX_DOWNLOAD_IMAGES

    def _download_data(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.dataset_path / "validation-images.csv"
        download_file(OPENIMAGES_IMAGE_LIST_URL, str(csv_path))
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            image_ids = [row[0] for row in reader if row and _IMAGE_ID_RE.match(row[0])]

        # Evenly subsample to a bounded, deterministic subset instead of
        # materializing the full ~41.6k-image split.
        stride = max(1, len(image_ids) // MAX_DOWNLOAD_IMAGES)
        image_ids = image_ids[::stride][:MAX_DOWNLOAD_IMAGES]

        # download_data() rmtrees dataset_path before calling this, so images_dir
        # always starts empty; fetch the whole (bounded) list.
        print(f"Downloading {len(image_ids)} Open Images validation images...")
        samples = [
            {
                "fname": f"{i}.jpg",
                "url": f"{OPENIMAGES_CVDF_URL}/{OPENIMAGES_SPLIT}/{i}.jpg",
            }
            for i in image_ids
        ]
        asyncio.run(download_many_urls(samples, self.images_dir, "fname", "url"))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_files[index]).convert("RGB")
        return self.transform(image), 0

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://storage.googleapis.com/openimages/web/index.html",
            split_description=f"{MAX_DOWNLOAD_IMAGES} images from validation split",
        )
