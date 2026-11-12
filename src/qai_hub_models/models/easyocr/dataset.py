# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.download import download_many_urls
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

ICDAR2015_FOLDER_NAME = "icdar2015"

ICDAR2015_URL = "http://rrc.cvc.uab.es/downloads/ch4_training_images.zip"
ICDAR2015_GT_URL = (
    "http://rrc.cvc.uab.es/downloads/ch4_training_localization_transcription_gt.zip"
)
ICDAR2015_VERSION = 2
ICDAR2015_IMGS_ASSET = CachedWebDatasetAsset(
    ICDAR2015_URL,
    ICDAR2015_FOLDER_NAME,
    ICDAR2015_VERSION,
    "ch4_training_images.zip",
)
ICDAR2015_GT_ASSET = CachedWebDatasetAsset(
    ICDAR2015_GT_URL,
    ICDAR2015_FOLDER_NAME,
    ICDAR2015_VERSION,
    "ch4_training_localization_transcription_gt.zip",
)

COCOTEXT_FOLDER_NAME = "coco_text"
COCOTEXT_VERSION = 1
COCOTEXT_ANNOTATIONS_ASSET = CachedWebDatasetAsset(
    "https://github.com/bgshih/cocotext/releases/download/dl/cocotext.v2.zip",
    COCOTEXT_FOLDER_NAME,
    COCOTEXT_VERSION,
    "cocotext.v2.zip",
)
COCOTEXT_NUM_CALIBRATION_IMAGES = 500
COCO_IMAGE_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org"
CocoTextPolygon = tuple[
    tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
]


class ICDAR2015Dataset(BaseDataset):
    """Wrapper class around ICDAR2015 dataset (text detection task)"""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.dataset_path = ICDAR2015_IMGS_ASSET.extracted_path.parent
        BaseDataset.__init__(self, self.dataset_path, split)

        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, 640, 640), dtype="float32")
        }
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]

    def _validate_data(self) -> bool:
        imgs_dir = self.dataset_path / "ch4_training_images"
        gt_dir = imgs_dir.parent / "ch4_training_localization_transcription_gt"

        if not imgs_dir.exists() or not gt_dir.exists():
            return False

        self.images = []
        self.gt_files = []

        for img_path in sorted(imgs_dir.glob("*.jpg")):
            gt_path = gt_dir / f"gt_{img_path.stem}.txt"
            if gt_path.exists():
                self.images.append(img_path)
                self.gt_files.append(gt_path)

        if not self.images:
            raise ValueError("No valid image/gt pairs found in ICDAR2015 dataset")

        return True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple]:
        """
        Returns a tuple of input data and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image_input : torch.Tensor
            input_image:
                Raw floating point pixel values for encoder consumption.
                3-channel Color Space: RGB, range [0, 1]

        ground_truth : tuple
            Empty tuple; no ground truth data.
        """
        image_path = self.images[index]
        orig_image = Image.open(image_path).convert("RGB")

        # Resize to model input size
        image = orig_image.resize((self.input_width, self.input_height))
        img_tensor = app_to_net_image_inputs(image)[1].squeeze(0)

        return img_tensor, ()

    def __len__(self) -> int:
        return len(self.images)

    def _download_data(self) -> None:
        ICDAR2015_IMGS_ASSET.fetch(extract=True)
        ICDAR2015_GT_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        """The default value for how many samples to run in each inference job."""
        return DatasetMetadata(
            link="https://rrc.cvc.uab.es/?ch=4&com=downloads",
            split_description="ICDAR2015 training split (1000 images)",
        )


class CocoTextCalibrationDataset(BaseDataset):
    """COCO-Text scene images used only for EasyOCR quantization calibration."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        if split != DatasetSplit.TRAIN:
            raise ValueError("CocoTextCalibrationDataset only supports TRAIN.")

        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, 640, 640), dtype="float32")
        }
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]
        self.annotation_path = (
            COCOTEXT_ANNOTATIONS_ASSET.extracted_path / "cocotext.v2.json"
        )
        self.images_dir = COCOTEXT_ANNOTATIONS_ASSET.extracted_path / "images"
        self.metadata: dict[str, Any] = {}
        self.image_metadata: list[dict[str, Any]] = []
        super().__init__(COCOTEXT_ANNOTATIONS_ASSET.extracted_path, split, input_spec)
        self.image_files = [
            self.images_dir / image["file_name"] for image in self.image_metadata
        ]

    def _get_image_metadata(self) -> list[dict[str, Any]]:
        eligible_image_ids = {
            int(annotation["image_id"])
            for annotation in self.metadata["anns"].values()
            if annotation.get("legibility") == "legible"
            and annotation.get("language") == "english"
            and annotation.get("utf8_string", "").strip()
        }
        images = [
            {**image, "file_name": Path(image["file_name"]).name}
            for image in self.metadata["imgs"].values()
            if image.get("set") == "train" and int(image["id"]) in eligible_image_ids
        ]
        if not images:
            raise ValueError("No eligible COCO-Text calibration images found.")
        images.sort(key=lambda image: int(image["id"]))
        step = max(1, len(images) // COCOTEXT_NUM_CALIBRATION_IMAGES)
        return images[::step][:COCOTEXT_NUM_CALIBRATION_IMAGES]

    def get_recognizer_calibration_sample(
        self, index: int
    ) -> tuple[np.ndarray, list[CocoTextPolygon]]:
        """Return a grayscale scene and its usable ground-truth word polygons."""
        image_metadata = self.image_metadata[index]
        image_id = str(image_metadata["id"])
        image = np.asarray(Image.open(self.image_files[index]).convert("L"))
        polygons = []
        for annotation_id in self.metadata["imgToAnns"][image_id]:
            annotation = self.metadata["anns"][str(annotation_id)]
            bbox = annotation["bbox"]
            if (
                annotation.get("legibility") != "legible"
                or annotation.get("language") != "english"
                or not annotation.get("utf8_string", "").strip()
                or max(bbox[2], bbox[3]) <= 20
            ):
                continue
            mask = annotation["mask"]
            if len(mask) != 8:
                continue
            polygons.append(
                (
                    (round(mask[0]), round(mask[1])),
                    (round(mask[2]), round(mask[3])),
                    (round(mask[4]), round(mask[5])),
                    (round(mask[6]), round(mask[7])),
                )
            )
        return image, polygons

    def _validate_data(self) -> bool:
        if not self.annotation_path.exists() or not self.images_dir.is_dir():
            return False
        if not self.metadata:
            with open(self.annotation_path) as annotation_file:
                self.metadata = json.load(annotation_file)
        if not self.image_metadata:
            self.image_metadata = self._get_image_metadata()
        return all(
            (self.images_dir / image["file_name"]).is_file()
            for image in self.image_metadata
        )

    def _download_data(self) -> None:
        COCOTEXT_ANNOTATIONS_ASSET.fetch(extract=True)
        with open(self.annotation_path) as annotation_file:
            self.metadata = json.load(annotation_file)
        self.image_metadata = self._get_image_metadata()
        samples = []
        for image in self.image_metadata:
            filename = image["file_name"]
            coco_split = "train2014" if "train2014" in filename else "val2014"
            samples.append(
                {
                    "file_name": filename,
                    "url": f"{COCO_IMAGE_BASE_URL}/{coco_split}/{filename}",
                }
            )

        asyncio.run(download_many_urls(samples, self.images_dir, "file_name", "url"))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple]:
        image = Image.open(self.image_files[index]).convert("RGB")
        image = image.resize((self.input_width, self.input_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, ()

    def __len__(self) -> int:
        return len(self.image_files)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://bgshih.github.io/cocotext/",
            split_description=(
                "COCO-Text v2 training images containing legible English text"
            ),
        )
