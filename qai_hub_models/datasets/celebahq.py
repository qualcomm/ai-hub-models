# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import os
import tempfile
from glob import glob

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.models._shared.repaint.utils import preprocess_inputs
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_celebahq_files,
    )
except ImportError:
    download_celebahq_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import app_to_net_image_inputs

CELEBAHQ_VERSION = 1
CELEBAHQ_DATASET_ID = "celebahq"
IMAGES_DIR_NAME = "celeba_hq"


class CelebAHQDataset(BaseDataset):
    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_images_zip: str | None = None,
        input_height: int = 512,
        input_width: int = 512,
        mask_type: str | None = "random_stroke",
        random_seed: int = 42,
    ) -> None:
        """Initialize CelebA-HQ dataset for inpainting tasks."""
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            CELEBAHQ_DATASET_ID, CELEBAHQ_VERSION, "data"
        )
        self.input_images_zip = input_images_zip
        split_name = "val" if split.name.lower() == "train" else split.name.lower()
        self.image_dir = self.data_path / IMAGES_DIR_NAME / split_name / "female"
        self.mask_dir = self.data_path / "mask"
        self.random_seed = random_seed
        BaseDataset.__init__(self, self.data_path, split)
        self.random_gen = np.random.default_rng(self.random_seed)
        self.input_height = input_height
        self.input_width = input_width
        self.mask_type = mask_type

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        # Load image
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = image.resize((self.input_height, self.input_width))
        if self.mask_type == "random_stroke":
            mask_array = self.random_stroke(self.input_width, self.input_height)
        else:
            mask_array = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
            # create center mask
            mask_array[
                self.input_height // 4 : self.input_width // 4 * 3,
                self.input_height // 4 : self.input_width // 4 * 3,
            ] = 255
        mask = Image.fromarray(mask_array).convert("L")

        gt = app_to_net_image_inputs(image)[1].squeeze(0)
        inputs = preprocess_inputs(image, mask)
        img_tensor, mask_tensor = inputs["image"].squeeze(0), inputs["mask"].squeeze(0)
        return (img_tensor, mask_tensor), (gt, mask_tensor)

    def random_stroke(self, img_width: int, img_height: int) -> NDArray:
        """
        Creates random brush stroke patterns for image editing.

        Parameters
        ----------
        img_width
            Width of the image
        img_height
            Height of the image

        Returns
        -------
        strokes : NDArray
            Numpy array (0=background, 255=stroke) with shape (height, width)
        """
        min_num_vertex = 4
        max_num_vertex = 12
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 12  # Thinner strokes
        max_width = 30
        average_radius = (
            math.sqrt(img_height * img_height + img_width * img_width) / 8
        )  # Smaller radius
        mask = Image.new("L", (img_width, img_height), 0)
        steps = 10  # Fewer strokes
        for _ in range(self.random_gen.integers(2, steps + 1)):
            num_vertex = self.random_gen.integers(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - self.random_gen.uniform(0, angle_range)
            angle_max = mean_angle + self.random_gen.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(
                        2 * math.pi - self.random_gen.uniform(angle_min, angle_max)
                    )
                else:
                    angles.append(self.random_gen.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append(
                (
                    int(self.random_gen.integers(0, w)),
                    int(self.random_gen.integers(0, h)),
                )
            )
            for i in range(num_vertex):
                r = np.clip(
                    self.random_gen.normal(
                        loc=average_radius, scale=average_radius // 2
                    ),
                    0,
                    2 * average_radius,
                )
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(self.random_gen.uniform(min_width, max_width))
            draw.line(vertex, fill=255, width=width)
            for v in vertex:
                draw.ellipse(
                    (
                        v[0] - width // 2,
                        v[1] - width // 2,
                        v[0] + width // 2,
                        v[1] + width // 2,
                    ),
                    fill=255,
                )

        if self.random_gen.normal() > 0:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.random_gen.normal() > 0:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return np.asarray(mask, np.uint8)

    def _validate_data(self) -> bool:
        if not self.image_dir.exists():
            return False
        self.image_paths = []
        self.mask_paths = []
        # Populate image and mask paths ()
        for ext in ["*.jpg", "*.png"]:
            self.image_paths.extend(sorted(glob(os.path.join(self.image_dir, ext))))
            self.mask_paths.extend(sorted(glob(os.path.join(self.mask_dir, ext))))

        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

        return True

    def _download_data(self, images_zip: str | None = None) -> None:
        # Use passed arg if provided, otherwise use instance attribute
        if images_zip is None:
            images_zip = self.input_images_zip

        # If no file provided/set, try auto-download
        if images_zip is None and download_celebahq_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                images_zip = os.path.join(tmpdir, f"{IMAGES_DIR_NAME}.zip")
                download_celebahq_files(images_zip, CELEBAHQ_VERSION)
                self._download_data(images_zip)
            return

        if images_zip is None or not images_zip.endswith(IMAGES_DIR_NAME + ".zip"):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=[
                    "Download `image.zip` from the Google Drive: https://www.kaggle.com/datasets/lamsimon/celebahq",
                    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset celebahq --files /path/to/celeba_hq.zip",
                ],
            )

        os.makedirs(self.data_path, exist_ok=True)
        extract_zip_file(images_zip, self.data_path)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/IIGROUP/MM-CelebA-HQ-Dataset",
            split_description="validation split",
        )
