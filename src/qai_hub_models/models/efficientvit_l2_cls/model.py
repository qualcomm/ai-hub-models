# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torchvision.transforms as T
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.templates.corevision.image_processing import (
    make_imagenet_transform,
)
from qai_hub_models.models.templates.efficientvit.external_repos.efficientvit.efficientvit.cls_model_zoo import (
    create_cls_model,
)
from qai_hub_models.models.templates.efficientvit.litemla_patch import (
    apply_attention_denominator_floor,
)
from qai_hub_models.models.templates.imagenet_classifier.model import ImagenetClassifier
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]

DEFAULT_WEIGHTS = "l2-r384.pt"
MODEL_ASSET_VERSION = 1
EFFICIENTVIT_L2_DIM = 384

# timm config: crop_pct=1.0, interpolation=bicubic → resize directly to 384, no extra crop
EFFICIENTVIT_L2_TRANSFORM = make_imagenet_transform(
    crop_size=EFFICIENTVIT_L2_DIM,
    resize_size=EFFICIENTVIT_L2_DIM,
    interpolation=T.InterpolationMode.BICUBIC,
    antialias=True,
)


class ImagenetEfficientViTL2Dataset(ImagenetDataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.VAL) -> None:
        super().__init__(split=split, transform=EFFICIENTVIT_L2_TRANSFORM)

    @classmethod
    def dataset_name(cls) -> str:
        return "imagenet_efficientvit_l2"


class OpenImagesEfficientViTL2Dataset(OpenImagesV7Dataset):
    def __init__(self, split: DatasetSplit = DatasetSplit.TRAIN) -> None:
        super().__init__(split=split, transform=EFFICIENTVIT_L2_TRANSFORM)


ATTENTION_EPS = 90


class EfficientViT(ImagenetClassifier):
    """Exportable EfficientViT Image classifier, end-to-end."""

    @classmethod
    def from_pretrained(
        cls, weights: str | None = None, precision: Precision = Precision.float
    ) -> Self:
        """Load EfficientViT from a weightfile created by the source repository.

        The LiteMLA attention-normalization denominator is floored (see
        ATTENTION_EPS) so it cannot quantize to integer 0 and trigger a
        divide-by-zero on the HTP fixed-point reciprocal. The floor is applied
        unconditionally rather than only for quantized precisions: the scorecard
        quantize path traces the model at float precision and then quantizes that
        ONNX graph, so the Clip must already be present in the float-traced graph
        for the downstream quantized model to inherit it. The float accuracy cost
        is small (~0.8% top-1).
        """
        if not weights:
            weights = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        efficientvit_model = create_cls_model(name="l2", weight_url=weights)
        efficientvit_model.to(torch.device("cpu"))
        efficientvit_model.eval()
        apply_attention_denominator_floor(efficientvit_model, ATTENTION_EPS)
        return cls(efficientvit_model)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, EFFICIENTVIT_L2_DIM, EFFICIENTVIT_L2_DIM),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            )
        }

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesEfficientViTL2Dataset

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetEfficientViTL2Dataset]

    def get_hub_litemp_percentage(self, precision: Precision) -> float:
        """Returns the Lite-MP percentage value for the specified mixed precision quantization."""
        return 10
