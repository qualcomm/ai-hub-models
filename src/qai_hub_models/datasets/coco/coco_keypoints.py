# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import shutil

import torch
from PIL import Image

from qai_hub_models.datasets.coco.coco import COCO_VAL_DATASET
from qai_hub_models.extern.xtcocotools.coco import COCO
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.printing import suppress_stdout

COCO_KPT_FOLDER_NAME = "coco"
COCO_KPT_VERSION = 3

# Reuse the standard COCO val images and annotations (person_keypoints_val2017.json
# is bundled inside annotations_trainval2017.zip).
COCO_KPT_ANNOTATIONS_ASSET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    COCO_KPT_FOLDER_NAME,
    COCO_KPT_VERSION,
    "annotations_trainval2017.zip",
    private_s3_key="qai-hub-models/datasets/coco/annotations_trainval2017.zip",
)
COCO_KPT_PERSON_ANNOTATIONS_PATH = (
    COCO_KPT_ANNOTATIONS_ASSET.extracted_path / "person_keypoints_val2017.json"
)

# COCO 17-keypoint skeleton connections (0-indexed)
COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 11),
    (6, 12),  # torso
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]


class CocoKeypointsDataset(BaseDataset):
    """
    COCO val2017 dataset for person keypoint / pose estimation evaluation.

    Each sample is a full scene image (resized + padded to the model's input
    resolution) together with the COCO image-id, category-id, and the
    resize/pad parameters needed to map model-space predictions back to the
    original image coordinate system for COCO evaluation.

    Unlike ``CocoBodyDataset`` (which crops individual person instances for
    top-down pose models), this dataset feeds the whole image to a one-stage
    detector-pose model such as YOLOv11-Pose.

    COCO keypoints (0-indexed)::

        0:  nose          1:  left_eye      2:  right_eye
        3:  left_ear      4:  right_ear     5:  left_shoulder
        6:  right_shoulder 7: left_elbow    8:  right_elbow
        9:  left_wrist    10: right_wrist   11: left_hip
        12: right_hip     13: left_knee     14: right_knee
        15: left_ankle    16: right_ankle
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        input_spec = input_spec or {
            "image": TensorSpec(shape=(1, 3, 640, 640), dtype="float32")
        }
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]

        self.image_dir = COCO_VAL_DATASET.extracted_path
        self.annotation_path = COCO_KPT_PERSON_ANNOTATIONS_PATH
        BaseDataset.__init__(self, self.annotation_path, split)

        with suppress_stdout():
            self.cocoGt = COCO(self.annotation_path)

        # Build image-id -> file_name map.
        self._id_to_filename: dict[int, str] = {
            img["id"]: img["file_name"] for img in self.cocoGt.dataset["images"]
        }

        # Keep only images that have at least one person annotation with keypoints.
        seen: set[int] = set()
        self._samples: list[tuple[int, int]] = []  # (image_id, category_id)
        for ann in self.cocoGt.dataset["annotations"]:
            img_id = ann["image_id"]
            if img_id not in seen and ann.get("num_keypoints", 0) > 0:
                seen.add(img_id)
                self._samples.append((img_id, ann["category_id"]))

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Get an item in this dataset.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Preprocessed image tensor, shape [3, H, W], float32 in [0, 1].
        ground_truth : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of (image_id, category_id, scale, pad) tensors.
        """
        image_id, category_id = self._samples[index]
        file_name = self._id_to_filename[image_id]
        img_path = self.image_dir / file_name

        pil_image = Image.open(img_path).convert("RGB")
        # Convert to NCHW float32 [0, 1]
        nchw = app_to_net_image_inputs(pil_image)[1]
        # Resize + letterbox-pad to model input size; capture scale and padding
        resized, scale, pad = resize_pad(nchw, (self.target_h, self.target_w))
        image = resized.squeeze(0)  # [3, H, W]

        return image, (
            torch.tensor(image_id, dtype=torch.int64),
            torch.tensor(category_id, dtype=torch.int64),
            torch.tensor(scale, dtype=torch.float32),
            torch.tensor(list(pad), dtype=torch.int64),
        )

    def _validate_data(self) -> bool:
        return self.image_dir.exists() and self.annotation_path.exists()

    def _download_data(self) -> None:
        COCO_VAL_DATASET.fetch(extract=True)
        ann_dir = COCO_KPT_ANNOTATIONS_ASSET.extracted_path
        if ann_dir.exists() and not COCO_KPT_PERSON_ANNOTATIONS_PATH.exists():
            shutil.rmtree(ann_dir)
        COCO_KPT_ANNOTATIONS_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 300

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 split - images with at least one person keypoint annotation",
        )
