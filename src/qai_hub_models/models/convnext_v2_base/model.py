# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from transformers import ConvNextV2ForImageClassification
from typing_extensions import Self

from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.templates.corevision.image_processing import (
    make_imagenet_transform,
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.imagenet_classifier.model import (
    TEST_IMAGENET_IMAGE,
    ImagenetClassifier,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "facebook/convnextv2-base-22k-224"
CONVNEXT_V2_BASE_DIM = 224

# facebook/convnextv2-base-22k-224 ConvNextImageProcessor config: crop_pct=0.875 -> resize=256, BICUBIC
CONVNEXT_V2_BASE_TRANSFORM = make_imagenet_transform(
    crop_size=CONVNEXT_V2_BASE_DIM,
    resize_size=256,
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


class ImagenetConvNextV2BaseDataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=CONVNEXT_V2_BASE_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_convnext_v2_base"


class OpenImagesConvNextV2BaseDataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=CONVNEXT_V2_BASE_TRANSFORM)


class ConvNextV2Base(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, ckpt_name: str = DEFAULT_WEIGHTS) -> Self:
        net = ConvNextV2ForImageClassification.from_pretrained(ckpt_name)
        assert isinstance(net, ConvNextV2ForImageClassification)
        return cls(net)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = normalize_image_torchvision(image_tensor)
        return self.net(image_tensor, return_dict=False)[0]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesConvNextV2BaseDataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetConvNextV2BaseDataset]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGENET_IMAGE)
        tensor = CONVNEXT_V2_BASE_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])
