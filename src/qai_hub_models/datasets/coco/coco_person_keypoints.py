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
from qai_hub_models.utils.samplers import (
    grouped_shuffled_order,
    grouped_strided_order,
)

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

    def _crops_by_image(self) -> list[list[int]]:
        return [self._image_to_indices[i] for i in self._image_ids_ordered]

    def eval_order(
        self, num_samples: int, samples_per_job: int, seed: int | None = None
    ) -> list[int]:
        """Keep all the detections of one image together in the same batch.

        Each sample here is one detected person (a crop) rather than a whole
        image, and a single image can have several. Keypoint accuracy is scored per
        image, so a batch that ends in the middle of an image's people leaves that
        image only partly scored in that batch, which costs a little accuracy. The
        order lists whole images in turn, spread across the dataset the way the
        default does. `default_samples_per_job` is picked so the first batch ends
        exactly at an image boundary.
        """
        if seed is None:
            order = grouped_strided_order(self._crops_by_image(), samples_per_job)
        else:
            order = grouped_shuffled_order(self._crops_by_image(), seed)
        return order[:num_samples]

    @staticmethod
    def default_samples_per_job() -> int:
        # The largest value under 1000 whose first batch ends on an image boundary.
        return 942

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="http://images.cocodataset.org/",
            split_description="val2017 split",
        )
