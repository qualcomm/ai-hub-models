# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.bsd.bsd300 import BSD300Dataset
from qai_hub_models.models.templates.corevision.image_processing import (
    pil_resize_pad,
    preprocess_PIL_image,
)
from qai_hub_models.utils.base_dataset import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_NOISE_SIGMA = 25
DEFAULT_HEIGHT = 256
DEFAULT_WIDTH = 256


class BSD300DenoisingDataset(BSD300Dataset):
    """
    Denoising evaluation dataset based on BSDS300.

    Reuses the BSDS300 images with synthetic Gaussian noise.
    Returns (noisy, clean) grayscale pairs normalized to [0, 1].
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
        noise_sigma: int = DEFAULT_NOISE_SIGMA,
    ) -> None:
        self.noise_sigma = noise_sigma
        if input_spec is not None and "image" in input_spec:
            shape = input_spec["image"][0]
            self.target_height = shape[2]
            self.target_width = shape[3]
        else:
            self.target_height = DEFAULT_HEIGHT
            self.target_width = DEFAULT_WIDTH
        super().__init__(split=split, input_spec=input_spec)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a (noisy, clean) grayscale image pair.

        Parameters
        ----------
        item
            Index of the image in the dataset.

        Returns
        -------
        noisy_tensor : torch.Tensor
            Noisy grayscale image, shape [1, H, W], float32, range [0, 1].
            Gaussian noise with sigma=self.noise_sigma/255 added to clean image.
        clean_tensor : torch.Tensor
            Clean grayscale image, shape [1, H, W], float32, range [0, 1].
        """
        img = Image.open(os.path.join(self.images_path, self.image_files[item]))
        gray = img.convert("L")

        resized, _, _ = pil_resize_pad(gray, (self.target_height, self.target_width))
        clean_tensor = preprocess_PIL_image(resized)[0]  # [1, H, W], float [0,1]

        rng = np.random.default_rng(seed=item)
        noise = torch.from_numpy(
            rng.normal(0, self.noise_sigma / 255.0, clean_tensor.shape).astype(
                np.float32
            )
        )
        noisy_tensor = torch.clamp(clean_tensor + noise, 0.0, 1.0)
        return noisy_tensor, clean_tensor

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
            split_description="test split (100 grayscale images with Gaussian noise sigma=25)",
        )
