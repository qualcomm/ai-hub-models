# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

eg1880_VERSION = 2

# originally from https://github.com/clovaai/ext_portrait_segmentation
eg1880_ASSET = CachedWebDatasetAsset.from_asset_store(
    "eg1800",
    eg1880_VERSION,
    "Portrait.zip",
)


class EG1800SegmentationDataset(BaseDataset):
    """Wrapper class around eg1800 dataset"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.eg1800_path = eg1880_ASSET.extracted_path

        BaseDataset.__init__(self, self.eg1800_path, split)

        if input_spec is not None:
            self.input_height = input_spec["image"][0][2]
            self.input_width = input_spec["image"][0][3]
        else:
            self.input_height = 256
            self.input_width = 256

    def _validate_data(self) -> bool:
        self.image_dir = self.eg1800_path / "images_data_crop"
        self.category_dir = self.eg1800_path / "GT_png"

        if not self.image_dir.exists() or not self.category_dir.exists():
            return False

        self.im_ids = []
        self.images = []
        self.categories = []

        # Match images with their corresponding masks
        for image_path in sorted(self.image_dir.glob("*.jpg")):
            im_id = image_path.stem
            annot_path = self.category_dir / f"{im_id}_mask.png"
            if annot_path.exists():
                self.im_ids.append(im_id)
                self.images.append(image_path)
                self.categories.append(annot_path)

        if not self.images:
            raise ValueError(
                f"No valid image-annotation pairs found in {self.image_dir} and {self.category_dir}"
            )

        return True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        orig_image = Image.open(self.images[index]).convert("RGB")
        orig_gt = Image.open(self.categories[index]).convert("L")

        # Resize to model input size
        image = orig_image.resize((self.input_width, self.input_height))
        gt_image = orig_gt.resize((self.input_width, self.input_height))
        img_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        gt_array = np.array(gt_image)

        # convert GT image to binary mask
        target = torch.from_numpy((gt_array > 0).astype(np.float32))
        return img_tensor, target

    def __len__(self) -> int:
        return len(self.images)

    def _download_data(self) -> None:
        eg1880_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 500

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/clovaai/ext_portrait_segmentation",
            split_description="validation split",
        )
