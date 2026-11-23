# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------


from __future__ import annotations

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from qai_hub_models.datasets.common import DatasetSplit
from qai_hub_models.datasets.imagenet.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_IMAGE_SIZE = 224


def make_clip_preprocessor(input_spec: InputSpec | None = None) -> Compose:
    if input_spec is not None:
        shape = next(iter(input_spec.values())).shape
        image_size = shape[-1]
    else:
        image_size = DEFAULT_IMAGE_SIZE
    return Compose(
        [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            lambda img: img.convert("RGB"),
            ToTensor(),
        ]
    )


# CLIP's official image preprocessing recipe: Resize -> CenterCrop with
# BICUBIC interpolation.
CLIP_IMAGE_PREPROCESSOR = make_clip_preprocessor()


class ClipImagenetZeroshotDataset(ImagenetDataset):
    """ImageNet validation set for zero-shot image(-text) evaluation."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(split=split, transform=make_clip_preprocessor(input_spec))


class ClipOpenImagesZeroshotDataset(OpenImagesV7Dataset):
    """Open Images dataset for zero-shot image(-text) calibration."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(
            split=split,
            input_spec=input_spec,
            transform=make_clip_preprocessor(input_spec),
        )
