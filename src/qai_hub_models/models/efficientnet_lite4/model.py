# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.efficientnet_lite4.external_repos.efficientnet_lite4.efficientnet_lite import (
    build_efficientnet_lite,
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
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_NUM_OF_CLASSES = 1000
EFFICIENTNET_LITE4_DIM = 300

# Weights taken from https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite4.pth
WEIGHTS_NAME = "efficientnet_lite4.pth"
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, WEIGHTS_NAME
)

MEAN_RGB = [0.498, 0.498, 0.498]
STDDEV_RGB = [0.502, 0.502, 0.502]

# Images are delivered in [0,1] range by the DataLoader; normalization is applied
# inside EfficientNetLite4.forward using the model-specific MEAN_RGB/STDDEV_RGB.
EFFICIENTNET_LITE4_TRANSFORM = make_imagenet_transform(
    crop_size=EFFICIENTNET_LITE4_DIM,
    resize_size=EFFICIENTNET_LITE4_DIM + 32,
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


class ImagenetEfficientNetLite4Dataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=EFFICIENTNET_LITE4_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_efficientnet_lite4"


class OpenImagesEfficientNetLite4Dataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=EFFICIENTNET_LITE4_TRANSFORM)


class EfficientNetLite4(ImagenetClassifier):
    def __init__(self, net: torch.nn.Module) -> None:
        # normalize_input=False: we apply model-specific normalization in forward()
        # instead of the base class's torchvision constants (0.485/0.456/0.406).
        super().__init__(net, normalize_input=False)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image_tensor
            A [B, 3, 300, 300] image tensor with pixel values in [0, 1].
            3-channel Color Space: RGB

        Returns
        -------
        class_log_likelihoods : torch.Tensor
            A [B, 1000] tensor of class logits.
        """
        shape = [1, -1, 1, 1]
        mean = torch.tensor(MEAN_RGB).reshape(shape).to(image_tensor.device)
        std = torch.tensor(STDDEV_RGB).reshape(shape).to(image_tensor.device)
        image_tensor = (image_tensor - mean) / std
        return self.net(image_tensor)

    @classmethod
    def from_pretrained(
        cls, weights: CachedWebModelAsset | str = DEFAULT_WEIGHTS
    ) -> Self:
        weights_path = str(
            weights.fetch() if isinstance(weights, CachedWebModelAsset) else weights
        )
        net = build_efficientnet_lite(MODEL_ID, DEFAULT_NUM_OF_CLASSES)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint, strict=True)
        net.eval()
        return cls(net)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesEfficientNetLite4Dataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetEfficientNetLite4Dataset]

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, EFFICIENTNET_LITE4_DIM, EFFICIENTNET_LITE4_DIM),
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
        tensor = EFFICIENTNET_LITE4_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])
