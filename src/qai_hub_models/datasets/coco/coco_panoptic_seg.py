# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import math

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.coco.coco import COCO_VAL_DATASET, DATASET_ASSET_VERSION
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, extract_zip_file
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

COCO_FOLDER_NAME = "coco-panoptic"

# Dataset assets
COCO_ANNOTATIONS_ASSET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
    COCO_FOLDER_NAME,
    DATASET_ASSET_VERSION,
    "panoptic_annotations_trainval2017.zip",
    private_s3_key="qai-hub-models/datasets/coco_panoptic/panoptic_annotations_trainval2017.zip",
)


class CocoPanopticSegmentationDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        """
        Parameters
        ----------
        split
            Dataset split (VAL only supported).
        input_spec
            Model input spec; the letterbox target H/W are read from its
            "image" tensor shape. Defaults to 800x800 when omitted (callers
            that only read annotations, never iterate samples).
        num_samples
            Number of samples to use. -1 means all available samples (5000 for val2017).
        """
        assert split == DatasetSplit.VAL, (
            f"{type(self).__name__} only supports DatasetSplit.VAL; "
            "images are only fetched for the val2017 split."
        )
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 800, 800))}
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]
        self.num_samples = num_samples

        # Load dataset paths
        self.image_dir = COCO_VAL_DATASET.extracted_path
        self.annotation_path = (
            COCO_ANNOTATIONS_ASSET.extracted_path
            / "annotations"
            / f"panoptic_{split.name.lower()}2017.json"
        )
        self.panoptic_dir = (
            self.annotation_path.parent
            / f"panoptic_{split.name.lower()}2017"
            / f"panoptic_{split.name.lower()}2017"
        )
        BaseDataset.__init__(self, self.annotation_path, split, input_spec)

        # Load annotations
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {self.annotation_path}"
            )
        with open(self.annotation_path) as f:
            self.annotations = json.load(f)

        self.img_dict = {img["id"]: img for img in self.annotations["images"]}
        self.ann_dict = {
            ann["image_id"]: ann for ann in self.annotations["annotations"]
        }
        self.img_ids = sorted(self.img_dict.keys())
        # Filter to only images that exist on disk
        self.img_ids = [
            img_id
            for img_id in self.img_ids
            if (self.image_dir / self.img_dict[img_id]["file_name"]).exists()
        ]
        # num_samples < 0 means use all available images
        if self.num_samples > 0:
            self.img_ids = self.img_ids[: self.num_samples]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, int, float, torch.Tensor]]:
        """Return image tensor and panoptic target for given index.

        Parameters
        ----------
        idx
            Dataset index.

        Returns
        -------
        torch.Tensor
            Image of shape (3, H, W) float32 RGB [0, 1], letterbox-padded.
        tuple[torch.Tensor, int, float, torch.Tensor]
            (target mask (H, W, 3), img_id, scale factor, pad tensor
            [left_pad, top_pad]). ``pad`` is a tensor (not a plain tuple) so
            that DataLoader's default_collate stacks it into a (B, 2) tensor
            instead of transposing it into two length-B tensors.
        """
        img_id = self.img_ids[idx]

        img_info = self.img_dict[img_id]
        ann = self.ann_dict[img_id]

        image_path = self.image_dir / img_info["file_name"]
        panoptic_file_path = self.panoptic_dir / ann["file_name"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = Image.open(image_path).convert("RGB")

        panoptic = Image.open(panoptic_file_path)

        # Letterbox resize: preserve aspect ratio, pad to input size
        image_tensor_raw = app_to_net_image_inputs(image)[1]  # (1, 3, H, W)
        image_tensor, scale, pad = resize_pad(
            image_tensor_raw, (self.input_height, self.input_width)
        )
        image_tensor = image_tensor.squeeze(0)  # (3, H, W)

        # Apply same letterbox to panoptic mask using nearest-neighbour.
        # Must match resize_pad's math.floor rounding exactly, or the mask
        # and image resize to different pixel dimensions.
        orig_w, orig_h = panoptic.size
        new_w = math.floor(orig_w * scale)
        new_h = math.floor(orig_h * scale)
        panoptic_resized = panoptic.resize((new_w, new_h), Image.Resampling.NEAREST)
        panoptic_padded = Image.new(
            "RGB", (self.input_width, self.input_height), (0, 0, 0)
        )
        panoptic_padded.paste(panoptic_resized, (pad[0], pad[1]))
        target = torch.from_numpy(np.array(panoptic_padded, dtype=np.int32))

        return image_tensor, (target, img_id, scale, torch.tensor(pad))

    def __len__(self) -> int:
        return len(self.img_ids)

    def _validate_data(self) -> bool:
        return (
            COCO_VAL_DATASET.extracted_path.exists()
            and COCO_ANNOTATIONS_ASSET.extracted_path.exists()
            and self.annotation_path.exists()
            and self.panoptic_dir.exists()
        )

    def _download_data(self) -> None:
        """Download and extract COCO dataset assets."""
        COCO_VAL_DATASET.fetch(extract=True)
        COCO_ANNOTATIONS_ASSET.fetch(extract=True)

        extract_zip_file(
            str(
                self.annotation_path.parent
                / f"panoptic_{self.split.name.lower()}2017.zip"
            ),
        )

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @staticmethod
    def default_num_calibration_samples() -> int:
        """Default number of calibration samples for quantization."""
        return 25

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/#home",
            split_description="val2017 split",
        )
