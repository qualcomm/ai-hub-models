# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

import torch
from transformers import MobileViTForImageClassification
from typing_extensions import Self

from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.image_processing import make_imagenet_transform

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "apple/mobilevit-small"
MODEL_ASSET_VERSION = 1

MOBILEVIT_DIM = 256
MOBILEVIT_TRANSFORM = make_imagenet_transform(MOBILEVIT_DIM, resize_size=MOBILEVIT_DIM)


class ImagenetMobileVITDataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=MOBILEVIT_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_256"

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 1000


class OpenImagesMobileVITDataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=MOBILEVIT_TRANSFORM)


class MobileVIT(ImagenetClassifier):
    """Exportable MobileVIT model, end-to-end."""

    INPUT_DIM = MOBILEVIT_DIM
    INPUT_TRANSFORM = MOBILEVIT_TRANSFORM

    def __init__(
        self,
        net: MobileViTForImageClassification,
    ) -> None:
        super().__init__(net, transform_input=False, normalize_input=False)

    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        net = MobileViTForImageClassification.from_pretrained(ckpt_name)
        assert isinstance(net, MobileViTForImageClassification)
        return cls(net)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = torch.flip(image_tensor, dims=[1])  # RGB -> BGR
        return self.net(image_tensor, return_dict=False)[0]

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetMobileVITDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesMobileVITDataset
