# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.cityscapes import CityscapesDataset
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.corevision.labels import write_labels_file
from qai_hub_models.models.templates.segmentation.segmentation_evaluator import (
    SegmentationOutputEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]
CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_LABEL = 255

# This image showcases the Cityscapes classes (but is not from the dataset)
TEST_CITYSCAPES_LIKE_IMAGE_NAME = "cityscapes_like_demo_2048x1024.jpg"
TEST_CITYSCAPES_LIKE_IMAGE_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, TEST_CITYSCAPES_LIKE_IMAGE_NAME
)


class CityscapesSegmentor(BaseModel):
    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(CITYSCAPES_NUM_CLASSES, resize_to_gt=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic segmentation an input `image`.

        Parameters
        ----------
        image
            A [1, 3, height, width] RGB image, with range [0, 1].
            Assumes image has been resized and normalized using the
            Cityscapes preprocesser (in cityscapes_segmentation/app.py).

        Returns
        -------
        logits : torch.Tensor
            Raw logit probabilities as a tensor of shape
            [1, num_classes, modified_height, modified_width],
            where the modified height and width will be some factor smaller
            than the input image.
        """
        return self.model(normalize_image_torchvision(image))

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 1024,
        width: int = 2048,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "mask": TensorSpec(
                io_type=IoType.TENSOR,
                description="Semantic segmentation mask",
                labels_file="cityscapes_labels.txt",
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(TEST_CITYSCAPES_LIKE_IMAGE_ASSET)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CityscapesDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CityscapesDataset

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("cityscapes", output_dir, metadata)
