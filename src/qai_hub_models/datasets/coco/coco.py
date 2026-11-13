# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import os
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.coco import CocoDetection

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
    transform_resize_pad_normalized_coordinates,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.download import download_many_urls
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec


class CocoDatasetClass(Enum):
    ALL_CLASSES = 91
    SUBSET_CLASSES = 80


DATASET_ID = "coco"
DATASET_ASSET_VERSION = 3

COCO_VAL_DATASET = CachedWebDatasetAsset(
    "http://images.cocodataset.org/zips/val2017.zip",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "val2017.zip",
    private_s3_key="qai-hub-models/datasets/coco/val2017.zip",
)
COCO_ANNOTATIONS = CachedWebDatasetAsset(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "annotations_trainval2017.zip",
    private_s3_key="qai-hub-models/datasets/coco/annotations_trainval2017.zip",
)

DEFAULT_NUM_TRAIN_SAMPLES = 2000
TOTAL_VAL_SAMPLES = 5000


def collate_fn(
    batch: list[torch.Tensor],
) -> list[torch.Tensor] | tuple[list[torch.Tensor], tuple[list[torch.Tensor], ...]]:
    try:
        image, gt = batch[0][0], batch[0][1]
        image_id, height, width, boxes, labels = gt
        new_list = []
        new_list.append(default_collate([i for i in image if torch.is_tensor(i)]))
        target = (
            torch.tensor(image_id),
            torch.tensor(height),
            torch.tensor(width),
            default_collate([i for i in boxes if torch.is_tensor(i)]),
            default_collate([i for i in labels if torch.is_tensor(i)]),
        )
        new_list.append(target)
        return new_list
    except Exception:
        return [], ([], [], [], [], [], [])


class CocoDatasetBase(BaseDataset, CocoDetection):
    """
    Shared machinery for COCO 2017 datasets (download, indexing, class
    remapping). Subclasses implement ``__getitem__`` to return the ground
    truth shape they need (detection boxes, segmentation masks, etc.).

    This wrapper supports the train and val splits of the 2017 version.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        max_boxes: int = 100,
        num_classes: CocoDatasetClass = CocoDatasetClass.SUBSET_CLASSES,
        max_train_samples: int = DEFAULT_NUM_TRAIN_SAMPLES,
    ) -> None:
        """
        Parameters
        ----------
        split
            Whether to use the train or val split of the dataset.

        input_spec
            Model input spec; determines shapes for model input produced by this dataset.

        max_boxes
            The maximum number of boxes for a given sample. Used so that
            when loading multiple samples in a batch via a dataloader, this will
            be the tensor dimension.

            If a sample has fewer than this many boxes, the tensor of boxes
            will be zero padded up to this amount.

            If a sample has more than this many boxes, an exception is thrown.

        num_classes
            Number of classes this model detects.

        max_train_samples
            Downloading the whole training set is too big, so we choose a reasonably
            large number to download by default that is enough for most use cases.
        """
        self.num_classes = num_classes
        self.train_samples: list[dict[str, Any]] = []
        self.max_train_samples = max_train_samples
        self.coco_base = COCO_VAL_DATASET.extracted_path.parent

        anno_file = (
            "instances_val2017.json"
            if split == DatasetSplit.VAL
            else "instances_train2017.json"
        )
        self.root = (
            COCO_VAL_DATASET.extracted_path
            if split == DatasetSplit.VAL
            else self.coco_base / "train2017"
        )
        BaseDataset.__init__(self, self.root, split, input_spec)
        CocoDetection.__init__(
            self,
            root=self.root,
            annFile=str(COCO_ANNOTATIONS.extracted_path / anno_file),
        )
        if split == DatasetSplit.TRAIN:
            if self.train_samples == []:
                # These are resolved during download, but if data is already downloaded
                # It should be done here.
                self._resolve_train_samples()
            self.ids = [int(sample["id"]) for sample in self.train_samples]

        # Full coco dataset has 91 classes but some models use an 80 class subset
        # This creates a mapping from the 91 class index to the 80 class index
        self.class80_label_map = {}
        if self.num_classes == CocoDatasetClass.SUBSET_CLASSES:
            categories = self.coco.loadCats(self.coco.getCatIds())
            categories.sort(key=lambda x: x["id"])
            for i, cat in enumerate(categories):
                self.class80_label_map[cat["id"]] = i

        # input_spec is (h, w) and target_image_size is (w, h)
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 640, 640))}
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]
        self.max_boxes = max_boxes

    def _validate_data(self) -> bool:
        # Check validation data exists
        if not self.root.exists():
            return False

        # Check annotations exist
        if not COCO_ANNOTATIONS.extracted_path.exists():
            return False

        if self.split == DatasetSplit.TRAIN:
            # Ensure there are enough training samples
            return len(os.listdir(self.root)) >= self.max_train_samples
        return len(os.listdir(self.root)) == TOTAL_VAL_SAMPLES

    def _download_data(self) -> None:
        COCO_VAL_DATASET.fetch(extract=True)
        COCO_ANNOTATIONS.fetch(extract=True)

        if self.split == DatasetSplit.TRAIN:
            self._resolve_train_samples()
            asyncio.run(
                download_many_urls(
                    self.train_samples, self.root, "file_name", "coco_url"
                )
            )

    def _resolve_train_samples(self) -> None:
        if self.split == DatasetSplit.TRAIN:
            with open(
                COCO_ANNOTATIONS.extracted_path / "instances_train2017.json"
            ) as f:
                train_metadata = json.loads(f.read())
            all_samples = sorted(train_metadata["images"], key=lambda k: k["id"])
            step_size = (len(all_samples)) // self.max_train_samples
            self.train_samples = all_samples[
                : step_size * self.max_train_samples : step_size
            ]

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 300

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 split",
        )


class CocoDataset(CocoDatasetBase):
    """
    COCO object detection dataset spanning 80 or 91 classes.

    Contains object detection samples and labels for the train and val splits
    of the 2017 version.
    """

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
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
            Input image resized for the network. RGB, floating point range [0-1].

        ground_truth : tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
            image_id
                Image ID within the original dataset
            target_height
                Returns image height.
            target_width
                Returned image width.
            bboxes
                bounding box data with shape (self.max_boxes, 4)
                The 4 are (x1, y1, x2, y2), and are normalized [0, 1] fp values.
                Box coordinates are normalized to reflect their locations in the input image
                    (which has been resized / padded to fit the network input shape).
            labels
                Ground truth predicted category IDs with shape (self.max_boxes)
            num_boxes
                Number of boxes present in the ground truth data. Shape [1].
        """
        image, target = CocoDetection.__getitem__(self, index)
        boxes_list = []
        labels_list = []
        src_image_w, src_image_h = image.size

        # Convert to torch (NCHW, range [0, 1]) tensor.
        torch_image = app_to_net_image_inputs(image)[1]
        # Scale and center-pad image to user-requested target image shape.
        scaled_padded_torch_image, scale_factor, pad = resize_pad(
            torch_image, (self.target_h, self.target_w)
        )

        for annotation in target:
            bbox = annotation.get("bbox")
            coords = torch.Tensor(
                [
                    [
                        bbox[0] / src_image_w,
                        bbox[1] / src_image_h,
                    ],
                    [
                        (bbox[0] + bbox[2]) / src_image_w,
                        (bbox[1] + bbox[3]) / src_image_h,
                    ],
                ]
            )
            transformed_coords = transform_resize_pad_normalized_coordinates(
                coords,
                (src_image_w, src_image_h),
                scaled_padded_torch_image.shape[2:4][::-1],
                scale_factor,
                pad,
            )
            boxes_list.append(list(transformed_coords.flatten()))
            label = int(annotation.get("category_id"))
            if self.num_classes == CocoDatasetClass.SUBSET_CLASSES:
                label = self.class80_label_map[label]
            labels_list.append(label)
        boxes = torch.tensor(boxes_list)
        labels = torch.tensor(labels_list)

        # Pad the number of boxes to a standard value
        num_boxes = len(labels)
        if num_boxes == 0:
            boxes = torch.zeros((self.max_boxes, 4))
            labels = torch.zeros(self.max_boxes)
        elif num_boxes > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - num_boxes), value=0)
            labels = F.pad(labels, (0, self.max_boxes - num_boxes), value=0)
        return scaled_padded_torch_image.squeeze(0), (
            self.ids[index],
            self.target_h,
            self.target_w,
            boxes,
            labels,
            torch.tensor([num_boxes]),
        )


class Coco180Dataset(CocoDataset):
    """Wrapper class around CocoDataset for 180 samples_per_job"""

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 180

    @staticmethod
    def dataset_name() -> str:
        return "coco_180"
