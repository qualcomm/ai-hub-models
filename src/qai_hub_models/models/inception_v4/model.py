# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import timm
import torchvision.transforms as T
from typing_extensions import Self

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
MODEL_ASSET_VERSION = 1
INCEPTION_V4_DIM = 299

# Official huggingface Inception_v4_Weights:
# resize=341, crop=299, BICUBIC, antialias=True
INCEPTION_V4_TRANSFORM = make_imagenet_transform(
    resize_size=341,
    crop_size=INCEPTION_V4_DIM,
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


class ImagenetInceptionV4Dataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=INCEPTION_V4_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_inception_v4"


class OpenImagesInceptionV4Dataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=INCEPTION_V4_TRANSFORM)


class InceptionNetV4(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls) -> Self:
        net = timm.create_model("inception_v4.tf_in1k", pretrained=True)
        return cls(net, transform_input=True)

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesInceptionV4Dataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetInceptionV4Dataset]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, INCEPTION_V4_DIM, INCEPTION_V4_DIM),
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
        tensor = INCEPTION_V4_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])
