# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import timm
import torchvision.transforms as T
from timm.models.nasnet import CellStem1, FirstCell
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.nasnet.model_patches import (
    CellStem1_forward,
    FirstCell_forward,
)
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
DEFAULT_WEIGHTS = "nasnetalarge.tf_in1k"
NASNET_DIM = 331

# timm nasnetalarge.tf_in1k: crop_pct=0.911 -> resize=floor(331/0.911)=363, BICUBIC, antialias=True
NASNET_TRANSFORM = make_imagenet_transform(
    crop_size=NASNET_DIM,
    resize_size=363,
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


class ImagenetNASNetDataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=NASNET_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_nasnet"


class OpenImagesNASNetDataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=NASNET_TRANSFORM)


class NASNet(ImagenetClassifier):
    @classmethod
    def from_pretrained(cls, checkpoint_path: str = DEFAULT_WEIGHTS) -> Self:
        # Make Functional in QNN and reduce inference latency for quantized variant
        CellStem1.forward = CellStem1_forward
        FirstCell.forward = FirstCell_forward

        model = timm.create_model(checkpoint_path, pretrained=True)
        return cls(model, transform_input=True)

    def get_hub_litemp_percentage(self, _: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 1

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesNASNetDataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetNASNetDataset]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, NASNET_DIM, NASNET_DIM),
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
        tensor = NASNET_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])
