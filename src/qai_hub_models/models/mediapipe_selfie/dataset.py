# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import base64
import json
import zlib
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit

HUMAN_MATTING_VERSION = 2

# orginally from the https://datasetninja.com/aisegmentcom-human-matting

HUMAN_MATTING_ASSET = CachedWebDatasetAsset.from_asset_store(
    "human_matting",
    HUMAN_MATTING_VERSION,
    "human_matting.tar",
)


class HumanMattingDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_height: int = 256,
        input_width: int = 256,
    ) -> None:
        BaseDataset.__init__(self, HUMAN_MATTING_ASSET.extracted_path, split)

        self.input_height = input_height
        self.input_width = input_width

    def _validate_data(self) -> bool:
        """Validate dataset structure and load image/annotation pairs."""
        self.image_dir = self.dataset_path / "ds" / "img"
        self.ann_dir = self.dataset_path / "ds" / "ann"

        if not self.image_dir.exists() or not self.ann_dir.exists():
            return False

        # Clear any existing data
        self.im_ids = []
        self.images = []
        self.annotations = []

        # Load image/annotation pairs
        for image_path in sorted(self.image_dir.glob("*.jpg")):
            im_id = image_path.stem
            ann_path = self.ann_dir / f"{im_id}.jpg.json"
            if ann_path.exists():
                self.im_ids.append(im_id)
                self.images.append(image_path)
                self.annotations.append(ann_path)

        if not self.images:
            raise ValueError(
                f"No valid image-annotation pairs found in {self.image_dir} and {self.ann_dir}"
            )

        return True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of input data and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_tensor : torch.Tensor
            Input image resized to (input_height, input_width).
            RGB, floating point range [0-1], shape (3, H, W).
        target : torch.Tensor
            Segmentation mask resized to (input_height, input_width).
            Binary values as long integers, shape (H, W).
        """
        orig_image = Image.open(self.images[index]).convert("RGB")
        with open(self.annotations[index]) as f:
            ann_data = json.load(f)

        # Reconstruct full mask from bitmap objects
        full_mask = Image.new("L", orig_image.size, 0)
        for obj in ann_data["objects"]:
            bitmap_data = obj["bitmap"]["data"]
            origin = obj["bitmap"]["origin"]  # [x, y]
            compressed_mask_bytes = base64.b64decode(bitmap_data)
            mask_png_bytes = zlib.decompress(compressed_mask_bytes)
            obj_mask = Image.open(BytesIO(mask_png_bytes))
            # Paste into full mask. We use the mask itself for the paste
            # to ensure we only paste foreground areas.
            full_mask.paste(1, tuple(origin), obj_mask.convert("L"))

        image = orig_image.resize((self.input_width, self.input_height), Image.BILINEAR)
        mask = full_mask.resize((self.input_width, self.input_height), Image.NEAREST)

        img_tensor = preprocess_PIL_image(image).squeeze(0)
        target = torch.from_numpy(np.array(mask)).to(torch.long)
        return img_tensor, target

    def __len__(self) -> int:
        return len(self.images)

    def _download_data(self) -> None:
        HUMAN_MATTING_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 250

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://datasetninja.com/aisegmentcom-human-matting",
            split_description="validation split",
        )
