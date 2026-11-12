# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.models.templates.corevision.labels import get_class_names
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

DATASET_ID = "dota128"
DATASET_ASSET_VERSION = 1

DOTA128_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/dota128.zip"
)

DOTA128_IMAGES_ASSET = CachedWebDatasetAsset(
    DOTA128_URL,
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "dota128.zip",
)

DOTA_V1_CLASSES = get_class_names("dota_v1")


class Dota128Dataset(BaseDataset):
    """
    Ultralytics DOTA128 dataset (debug subset of DOTAv1).

    Ground truth is converted to cxcywhr format (radians), normalized to the
    resized and padded model input space.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        max_boxes: int = 1200,
        num_samples: int = 128,
    ) -> None:
        self.max_boxes = int(max_boxes)
        self.num_samples = int(num_samples)

        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 640, 640))}
        self.target_h = input_spec["image"][0][2]
        self.target_w = input_spec["image"][0][3]

        self._images: list[Path] = []
        self._labels: list[Path] = []

        super().__init__(
            DOTA128_IMAGES_ASSET.extracted_path, split=split, input_spec=input_spec
        )

    def __len__(self) -> int:
        return len(self._images)

    def _load_entries(self) -> bool:
        # The public DOTA128 release is a small training subset only. Reuse it
        # for every split so calibration and evaluation behave consistently.
        images_dir = self.dataset_path / "images" / "train"
        labels_dir = self.dataset_path / "labels" / "train"
        if not images_dir.exists() or not labels_dir.exists():
            return False

        img_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
        if self.num_samples > 0:
            img_files = img_files[: min(self.num_samples, len(img_files))]

        self._images = img_files
        self._labels = [
            labels_dir / f"{image_path.stem}.txt" for image_path in img_files
        ]
        return len(self._images) > 0

    def _validate_data(self) -> bool:
        if not self.dataset_path.exists():
            return False
        return self._load_entries()

    def _download_data(self) -> None:
        DOTA128_IMAGES_ASSET.fetch(extract=True)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        img_path = self._images[index]
        lbl_path = self._labels[index]

        image = Image.open(img_path).convert("RGB")
        src_w, src_h = image.size

        torch_image = app_to_net_image_inputs(image)[1]
        scaled, scale, pad = resize_pad(torch_image, (self.target_h, self.target_w))

        boxes_list = []
        labels_list = []

        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 9:
                    continue

                try:
                    cls = int(parts[0])
                    quad_raw = [float(x) for x in parts[1:]]
                except ValueError:
                    continue

                if cls < 0 or cls >= len(DOTA_V1_CLASSES):
                    continue

                if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in quad_raw):
                    continue

                quad = torch.tensor(quad_raw, dtype=torch.float32).view(4, 2)
                quad[:, 0] *= src_w
                quad[:, 1] *= src_h

                cx = quad[:, 0].mean()
                cy = quad[:, 1].mean()
                w = torch.linalg.norm(quad[1] - quad[0])
                h = torch.linalg.norm(quad[2] - quad[1])
                r = math.atan2(
                    float((quad[1] - quad[0])[1]),
                    float((quad[1] - quad[0])[0]),
                )

                cx = (cx * scale + pad[0]) / self.target_w
                cy = (cy * scale + pad[1]) / self.target_h
                norm = max(self.target_w, self.target_h)
                w = (w * scale) / norm
                h = (h * scale) / norm

                boxes_list.append([cx, cy, w, h, r])
                labels_list.append(cls)

        boxes = (
            torch.tensor(boxes_list, dtype=torch.float32)
            if boxes_list
            else torch.zeros((0, 5), dtype=torch.float32)
        )
        labels = (
            torch.tensor(labels_list, dtype=torch.int64)
            if labels_list
            else torch.zeros((0,), dtype=torch.int64)
        )

        n = labels.numel()
        if n == 0:
            boxes = torch.zeros((self.max_boxes, 5))
            labels = torch.zeros((self.max_boxes,), dtype=torch.int64)
        elif n > self.max_boxes:
            raise ValueError(
                f"Sample has more boxes than max boxes {self.max_boxes}. "
                "Re-initialize the dataset with a larger value for max_boxes."
            )
        else:
            boxes = F.pad(boxes, (0, 0, 0, self.max_boxes - n))
            labels = F.pad(labels, (0, self.max_boxes - n))

        return scaled.squeeze(0), (
            self._safe_image_id(img_path),
            self.target_h,
            self.target_w,
            boxes,
            labels,
            torch.tensor([n], dtype=torch.int64),
        )

    @staticmethod
    def default_samples_per_job() -> int:
        return 128

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://docs.ultralytics.com/datasets/obb/dota128/",
            split_description="training subset (128 samples)",
        )

    @staticmethod
    def _safe_image_id(path: Path) -> int:
        try:
            return int(path.stem)
        except ValueError:
            return abs(hash(path.stem)) % (2**31)
