# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from typing_extensions import Self

from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.repvit.external_repos.repvit.model.repvit import (
    repvit_m2_3,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    make_imagenet_transform,
)
from qai_hub_models.models.templates.imagenet_classifier.model import (
    TEST_IMAGENET_IMAGE,
    ImagenetClassifier,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import (
    InputSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 4
DEFAULT_NUM_OF_CLASSES = 1000
REPVIT_IMAGE_DIM = 224

# Weights taken from https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_distill_300e.pth
WEIGHTS_NAME = "repvit_m2_3_distill_300e.pth"
DEFAULT_WEIGHTS = CachedWebModelAsset(
    "https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m2_3_distill_300e.pth",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    WEIGHTS_NAME,
)

# Standard ImageNet normalization
REPVIT_TRANSFORM = make_imagenet_transform(
    crop_size=REPVIT_IMAGE_DIM,
    resize_size=REPVIT_IMAGE_DIM + 32,
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


class ImagenetRepViTDataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=REPVIT_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_repvit"


class OpenImagesRepViTDataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=REPVIT_TRANSFORM)


class RepViT(ImagenetClassifier):
    model_asset_version = MODEL_ASSET_VERSION

    @classmethod
    def from_pretrained(
        cls, weights: CachedWebModelAsset | str = DEFAULT_WEIGHTS
    ) -> Self:
        weights_path = str(
            weights.fetch() if isinstance(weights, CachedWebModelAsset) else weights
        )
        # distillation=False — clean single output tensor for export
        net = repvit_m2_3(
            pretrained=False, num_classes=DEFAULT_NUM_OF_CLASSES, distillation=False
        )
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        net.load_state_dict(
            state_dict, strict=False
        )  # distillation checkpoint has classifier_dist keys not present in model
        return cls(net)

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesRepViTDataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetRepViTDataset]

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGENET_IMAGE)
        tensor = REPVIT_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])
