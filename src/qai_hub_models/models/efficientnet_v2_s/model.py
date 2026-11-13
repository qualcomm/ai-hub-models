# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torchvision.models as tv_models
import torchvision.transforms as T
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.templates.corevision.image_processing import (
    make_imagenet_transform,
)
from qai_hub_models.models.templates.imagenet_classifier.model import (
    TEST_IMAGENET_IMAGE,
    ImagenetClassifier,
)
from qai_hub_models.utils.asset_loaders import load_image
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "IMAGENET1K_V1"
EFFICIENTNET_V2_S_DIM = 384

# Official torchvision EfficientNet_V2_S_Weights.IMAGENET1K_V1:
# resize=384 (no crop), BILINEAR, antialias=True
EFFICIENTNET_V2_S_TRANSFORM = make_imagenet_transform(
    crop_size=EFFICIENTNET_V2_S_DIM,
    resize_size=EFFICIENTNET_V2_S_DIM,
    interpolation=T.InterpolationMode.BILINEAR,
    antialias=True,
)


class ImagenetEfficientNetV2SDataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=EFFICIENTNET_V2_S_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_efficientnet_v2_s"


class OpenImagesEfficientNetV2SDataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=EFFICIENTNET_V2_S_TRANSFORM)


class EfficientNetV2s(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS) -> Self:
        net = tv_models.efficientnet_v2_s(weights=weights)
        return cls(net)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesEfficientNetV2SDataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetEfficientNetV2SDataset]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, EFFICIENTNET_V2_S_DIM, EFFICIENTNET_V2_S_DIM),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            )
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGENET_IMAGE)
        tensor = EFFICIENTNET_V2_S_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])
