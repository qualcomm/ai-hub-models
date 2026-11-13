# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from qai_hub_models.datasets.coco import COCO_VAL_DATASET
from qai_hub_models.datasets.coco.coco import COCO_ANNOTATIONS
from qai_hub_models.datasets.coco.cocobody import CocoBodyDataset
from qai_hub_models.datasets.common import DatasetMetadata, DatasetSplit
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_cs,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.input_spec import InputSpec

DATASET_ASSET_VERSION = 1
DATASET_ID = "coco_pose"

# Person detection results for COCO val2017.
# Sourced from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
COCO_PERSON_DETECTION_RESULTS = CachedWebDatasetAsset.from_asset_store(
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "COCO_val2017_detections_AP_H_56_person.json",
)


class CocoDetectorKeypointsDataset(CocoBodyDataset):
    """COCO val2017 top-down pose dataset using person detector bounding boxes."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        super().__init__(split, input_spec, num_samples)
        self.kpt_db: list[tuple[str, int, int, np.ndarray, np.ndarray, float, float]]

    def _get_annotation_path(self) -> Path:
        return COCO_ANNOTATIONS.extracted_path / "person_keypoints_val2017.json"

    def _load_kpt_db(
        self,
    ) -> list[tuple[str, int, int, np.ndarray, np.ndarray, float, float]]:
        det_file = COCO_PERSON_DETECTION_RESULTS.fetch()
        with open(det_file) as f:
            all_boxes: list[dict] = json.load(f)

        aspect_ratio = self.target_w / self.target_h
        kpt_db: list[tuple[str, int, int, np.ndarray, np.ndarray, float, float]] = []
        self._image_to_indices: dict[int, list[int]] = {}
        self._image_ids_ordered: list[int] = []

        for det in all_boxes:
            if det.get("category_id") != 1 or float(det["score"]) <= 0:
                continue

            image_id = int(det["image_id"])
            img_info = self.cocoGt.loadImgs(image_id)[0]
            x, y, w, h = det["bbox"]
            center, scale = box_xywh_to_cs(
                [x, y, w, h], aspect_ratio, padding_factor=1.25
            )

            idx = len(kpt_db)
            kpt_db.append(
                (
                    img_info["file_name"],
                    image_id,
                    1,
                    center,
                    scale,
                    float(det["score"]),
                    float(w * h),
                )
            )
            if image_id not in self._image_to_indices:
                self._image_to_indices[image_id] = []
                self._image_ids_ordered.append(image_id)
            self._image_to_indices[image_id].append(idx)

            if self.samples != -1 and len(kpt_db) >= self.samples:
                break

        return kpt_db

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[int, int, np.ndarray, np.ndarray, float, float]]:
        """
        Get item in this dataset.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Input image resized for the network. RGB, floating point range [0-1].
        ground_truth : tuple[int, int, np.ndarray, np.ndarray, float, float]
            image_id, category_id, center, scale, box_score, area
        """
        file_name, image_id, category_id, center, scale, box_score, area = self.kpt_db[
            index
        ]
        data_numpy = cv2.imread(
            str(self.image_dir / file_name),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        assert data_numpy is not None, (
            f"Image not found at {self.image_dir / file_name}"
        )
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        image = pre_process_with_affine(
            data_numpy, center, scale, 0, (self.target_h, self.target_w)
        ).squeeze(0)
        return image, (image_id, category_id, center, scale, box_score, area)

    def _validate_data(self) -> bool:
        return (
            COCO_VAL_DATASET.extracted_path.exists()
            and self._get_annotation_path().exists()
            and COCO_PERSON_DETECTION_RESULTS.local_cache_path.exists()
        )

    def _download_data(self) -> None:
        """Download COCO val images, keypoint annotations, and detector results."""
        COCO_VAL_DATASET.fetch(extract=True)
        # Re-extract if the annotations dir exists but person_keypoints_val2017.json
        # is missing (can happen if a previous extraction only produced instance files).
        ann_dir = COCO_ANNOTATIONS.extracted_path
        if ann_dir.exists() and not self._get_annotation_path().exists():
            shutil.rmtree(ann_dir)
        COCO_ANNOTATIONS.fetch(extract=True)
        COCO_PERSON_DETECTION_RESULTS.fetch()

    def get_dataloader(
        self, num_samples: int, samples_per_job: int | None = None
    ) -> DataLoader:
        """Return a DataLoader with ~num_samples crops, rounded up to complete images.

        Strides over images so samples are spread across the dataset, stopping
        once the accumulated crop count reaches num_samples. The last image is
        always included in full so OKS-NMS has the complete detection set.
        """
        total_crops = len(self.kpt_db)
        crop_stride = max(1, total_crops // num_samples)
        # Identify images by striding over crops, preserving dataset order
        seen_images: dict[int, None] = {}
        for i in range(0, total_crops, crop_stride):
            seen_images[self.kpt_db[i][1]] = None
        # Collect all crops per selected image, stopping once num_samples is reached
        selected: list[int] = []
        for img_id in seen_images:
            selected.extend(self._image_to_indices[img_id])
            if len(selected) >= num_samples:
                break
        return DataLoader(
            Subset(self, selected),
            batch_size=samples_per_job or self.default_samples_per_job(),
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="http://images.cocodataset.org/",
            split_description="val2017 split",
        )
