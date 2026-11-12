# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch

from qai_hub_models.datasets.coco.coco import COCO_VAL_DATASET
from qai_hub_models.extern.xtcocotools.coco import COCO
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_cs,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.printing import suppress_stdout

COCO_FOLDER_NAME = "coco-wholebody"
COCO_VERSION = 2

COCO_VAL_ANNOTATIONS_ASSET = CachedWebDatasetAsset.from_asset_store(
    COCO_FOLDER_NAME,
    COCO_VERSION,
    "coco_wholebody_val_v1.0.json",
)


class CocoBodyDatasetBase(BaseDataset):
    """
    Shared machinery for COCO-WholeBody-derived datasets (download, keypoint
    db construction, ``cocoGt`` access). Subclasses implement ``__getitem__``
    to return the ground truth shape they need (pose keypoints, face crops,
    wholebody keypoints, etc.).

    http://images.cocodataset.org/

    COCO keypoints::
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 256, 192))}
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]
        self.samples = num_samples

        # Load dataset paths
        self.image_dir = COCO_VAL_DATASET.extracted_path
        self.annotation_path = self._get_annotation_path()
        BaseDataset.__init__(self, self.annotation_path, split)

        # Load annotations
        if not self.annotation_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {self.annotation_path}"
            )
        with suppress_stdout():
            self.cocoGt = COCO(self.annotation_path)
        self.img_ids: list[int] = sorted(self.cocoGt.getImgIds())
        self.kpt_db = self._load_kpt_db()

    def _get_annotation_path(self) -> Path:
        """Return the path to the annotation file. Override in subclasses."""
        return COCO_VAL_ANNOTATIONS_ASSET.path

    def _load_kpt_db(self) -> list[Any]:
        kpt_db = []
        for img_id in self.img_ids:
            img_info = self.cocoGt.loadImgs(img_id)[0]
            width, height = img_info["width"], img_info["height"]
            ann_ids = self.cocoGt.getAnnIds(imgIds=img_id, catIds=[1], iscrowd=False)
            annotations = self.cocoGt.loadAnns(ann_ids)
            ratio = self.target_w / self.target_h

            # keep only single person objects
            person_anns = list(annotations)
            if len(person_anns) != 1:
                continue
            ann = person_anns[0]

            # Store cleaned bound boxes
            x, y, w, h = ann["bbox"]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x1 + max(0, w - 1))
            y2 = min(height - 1, y1 + max(0, h - 1))
            if ann.get("area", 0) > 0:
                bbox = [x1, y1, x2 - x1, y2 - y1]
                center, scale = box_xywh_to_cs(bbox, ratio, padding_factor=1.25)
                kpt_db.append(
                    (
                        img_info["file_name"],
                        img_id,
                        ann.get("category_id", 0),
                        center,
                        scale,
                    )
                )

            if len(kpt_db) == self.samples:
                break
        return kpt_db

    def __len__(self) -> int:
        return len(self.kpt_db)

    def _validate_data(self) -> bool:
        return (
            COCO_VAL_DATASET.extracted_path.exists()
            and COCO_VAL_ANNOTATIONS_ASSET.path.exists()
        )

    def _download_data(self) -> None:
        """Download and extract COCO-WholeBody dataset assets."""
        # Download and extract images
        COCO_VAL_DATASET.fetch(extract=True)
        COCO_VAL_ANNOTATIONS_ASSET.fetch()

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="http://images.cocodataset.org/",
            split_description="val2017 split",
        )


class CocoBodyDataset(CocoBodyDatasetBase):
    """COCO-WholeBody human pose dataset returning cropped person instances."""

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
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

        ground_truth : Any
            image_id
                Image ID within the original dataset
            category_id
                Ground truth prediction category ID.
            center
                The center coordinates of the bounding box, with shape(2,) -- (x. y).
            scale
                Bounding box scaling factor, with shape(2,) -- (x, y).
        """
        (
            file_name,
            image_id,
            category_id,
            center,
            scale,
        ) = self.kpt_db[index]
        rotate = 0
        img_path = self.image_dir / file_name

        data_numpy = cv2.imread(
            str(img_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        assert data_numpy is not None, f"Image not found at {img_path}"
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        image = pre_process_with_affine(
            data_numpy, center, scale, rotate, (self.target_h, self.target_w)
        ).squeeze(0)

        return image, (image_id, category_id, center, scale)
