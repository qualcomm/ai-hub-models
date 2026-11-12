# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import logging
from pathlib import Path

import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
    resize_pad,
    transform_resize_pad_normalized_coordinates,
)
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset, load_image
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

DATASET_ID = "roboflow_car_plate"
DATASET_VERSION = 1

# Roboflow car-plate-k6xij v26, UAE licence plates.
# Hosted at: https://universe.roboflow.com/zara-hara/car-plate-k6xij
# The zip bundles images + YOLO-format labels for train/valid/test splits.
DATASET_ASSET = CachedWebDatasetAsset.from_asset_store(
    DATASET_ID,
    DATASET_VERSION,
    "car_plate_v26.zip",
)

# 51 class names matching data.yaml from the Roboflow export.
CLASS_NAMES = get_class_names("roboflow_car_plate")

# Emirate identifier class indices (used by the evaluator to report plate state).
EMIRATE_CLASS_IDS = {
    CLASS_NAMES.index("new_DUBAI"),
    CLASS_NAMES.index("new_RAK"),
    CLASS_NAMES.index("new_abudabi"),
    CLASS_NAMES.index("new_ajman"),
    CLASS_NAMES.index("new_am"),
    CLASS_NAMES.index("new_fujairah"),
    CLASS_NAMES.index("old_DUBAI"),
    CLASS_NAMES.index("old_RAK"),
    CLASS_NAMES.index("old_abudabi"),
    CLASS_NAMES.index("old_ajman"),
    CLASS_NAMES.index("old_am"),
    CLASS_NAMES.index("old_fujira"),
    CLASS_NAMES.index("old_sharka"),
}

PLATE_CLASS_ID = CLASS_NAMES.index("plate")

# Maximum number of annotation boxes per image (for fixed-size tensors).
MAX_BOXES = 80


class RoboflowCarPlateDataset(BaseDataset):
    """YOLO-format UAE licence-plate dataset from Roboflow (car-plate-k6xij v26).

    Each sample returns a preprocessed image tensor and a ground-truth tuple
    matching the format expected by DetectionEvaluator / LicensePlateEvaluator:

        (image_id, height, width, boxes, labels, num_boxes)

    Boxes are in normalised (x1, y1, x2, y2) coordinates after resize_pad
    to the model input size (same transform as the App/on-device path).
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        if input_spec is None:
            input_spec = {"image": TensorSpec(shape=(1, 3, 640, 640))}
        self.target_h: int = input_spec["image"][0][2]
        self.target_w: int = input_spec["image"][0][3]

        split_dir = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VAL: "valid",
            DatasetSplit.TEST: "test",
        }[split]

        # extracted_path is <cache>/roboflow_car_plate/v1/car_plate_v26
        dataset_root = DATASET_ASSET.extracted_path
        split_path = dataset_root / split_dir

        # Set images_dir / labels_dir before super().__init__() because the
        # base class calls _validate_data() during __init__.
        self.images_dir = split_path / "images"
        self.labels_dir = split_path / "labels"

        super().__init__(dataset_root, split, input_spec)

        # Populate sample list after download (super().__init__ may trigger it).
        self.samples: list[str] = sorted(p.stem for p in self.images_dir.glob("*.jpg"))

    def _download_data(self) -> None:
        DATASET_ASSET.fetch(extract=True)

    def _validate_data(self) -> bool:
        return (
            self.images_dir.exists()
            and self.labels_dir.exists()
            and len(list(self.images_dir.glob("*.jpg"))) > 0
            and len(list(self.labels_dir.glob("*.txt"))) > 0
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """Return (image, gt) for one sample.

        image
            Preprocessed float32 tensor, shape (3, target_h, target_w), range [0, 1].
        gt
            Tuple of (image_id, target_h, target_w, boxes, labels, num_boxes) where
            boxes is (MAX_BOXES, 4) in normalised xyxy coords and labels is (MAX_BOXES,).
        """
        stem = self.samples[index]
        img_path = self.images_dir / f"{stem}.jpg"

        pil_img = load_image(img_path)
        src_w, src_h = pil_img.size

        # Same resize_pad transform as the App / on-device path ensures GT boxes
        # are in the same coordinate space as model predictions.
        torch_image = preprocess_PIL_image(pil_img)
        scaled_image, scale, pad = resize_pad(
            torch_image, (self.target_h, self.target_w)
        )

        label_path = self.labels_dir / f"{stem}.txt"
        boxes, labels = _load_yolo_labels(
            label_path, src_w, src_h, scaled_image, scale, pad
        )

        num_boxes = len(boxes)
        if num_boxes > MAX_BOXES:
            logging.warning(
                "Image %s has %d GT boxes; truncating to MAX_BOXES=%d.",
                stem,
                num_boxes,
                MAX_BOXES,
            )
        boxes_padded = torch.zeros(MAX_BOXES, 4)
        labels_padded = torch.zeros(MAX_BOXES, dtype=torch.long)
        if num_boxes > 0:
            n = min(num_boxes, MAX_BOXES)
            boxes_padded[:n] = boxes[:n]
            labels_padded[:n] = labels[:n]

        gt = (
            index,
            self.target_h,
            self.target_w,
            boxes_padded,
            labels_padded,
            torch.tensor(min(num_boxes, MAX_BOXES)),
        )
        return scaled_image.squeeze(0), gt

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://universe.roboflow.com/zara-hara/car-plate-k6xij/dataset/26",
            split_description="valid split (Roboflow car-plate-k6xij v26)",
        )


def _load_yolo_labels(
    label_path: Path,
    src_w: int,
    src_h: int,
    scaled_image: torch.Tensor,
    scale: float,
    pad: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse YOLO-format label file and return (boxes, labels) in normalised [0,1] coords
    matching the resize_pad output image.
    """
    if not label_path.exists():
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

    dst_w = scaled_image.shape[-1]
    dst_h = scaled_image.shape[-2]
    boxes: list[list[float]] = []
    labels: list[int] = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = (
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )

            # Convert YOLO normalised cxcywh to x1y1x2y2, then apply the same
            # resize_pad transform so GT coords match predicted box coordinates.
            coords = torch.tensor(
                [[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy + bh / 2]],
                dtype=torch.float32,
            )
            transformed = transform_resize_pad_normalized_coordinates(
                coords, (src_w, src_h), (dst_w, dst_h), scale, pad
            )
            x1, y1 = transformed[0].tolist()
            x2, y2 = transformed[1].tolist()
            boxes.append([max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)])
            labels.append(cls)

    if not boxes:
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)

    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(
        labels, dtype=torch.long
    )
