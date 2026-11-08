# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
import torch
from torchvision import transforms
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.imagenet import ImagenetDataset
from qai_hub_models.datasets.open_images import OpenImagesV7Dataset
from qai_hub_models.models.templates.imagenet_classifier.classification_evaluator import (
    ClassificationEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import (
    IMAGENET_DIM,
    IMAGENET_TRANSFORM,
    normalize_image_torchvision,
)
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)
from qai_hub_models.utils.labels import write_labels_file

MODEL_ASSET_VERSION = 1
MODEL_ID = __name__.split(".")[-2]

TEST_IMAGENET_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "dog.jpg"
)


class ImagenetClassifier(BaseModel):
    """Base class for all Imagenet Classifier models within QAI Hub Models."""

    INPUT_DIM: int = IMAGENET_DIM
    INPUT_TRANSFORM: transforms.Compose = IMAGENET_TRANSFORM

    def __init__(
        self,
        net: torch.nn.Module,
        transform_input: bool = False,
        normalize_input: bool = True,
    ) -> None:
        """
        Basic initializer which takes in a pretrained classifier network.
        Subclasses can choose to implement their own __init__ and forward methods.

        Parameters
        ----------
        net
            Imagenet classifier network.
        transform_input
            If True, preprocesses the input according to the method with which it was trained on ImageNet.
        normalize_input
            Normalize input of the imagenet classifier inside the network
            instead of requiring it to be done beforehand in a preprocessing step. If set to true, the dynamic
            range of the image input is [0, 1], which is the standard mapping for floating point images.
        """
        super().__init__()
        self.normalize_input = normalize_input
        self.transform_input = transform_input
        self.net = net

    # Type annotation on image_tensor causes aimet onnx export failure
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for an input `image`.

        Parameters
        ----------
        image_tensor
            A [1, 3, 224, 224] image.
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1] if self.normalize_input, else ~[-2.5, 2.5]
            3-channel Color Space: RGB

        Returns
        -------
        class_log_likelihoods : torch.Tensor
            A [1, 1000] where each value is the log-likelihood of
            the image belonging to the corresponding Imagenet class.
        """
        if self.normalize_input and self.transform_input:
            # Combining the norm and transform is mathematically equivalent to: 2(image_tensor) - 1
            image_tensor = image_tensor * 2 - 1
        elif self.normalize_input:
            # Image normalization required before images of range [0, 1] are passed
            # to a torchvision model.
            image_tensor = normalize_image_torchvision(image_tensor)
        elif self.transform_input:
            # Some torchvision models set parameter transform_input to true by default when they are initialized.
            #
            # This is mathematically equivalent to the parameter, but converts better than the built-in.
            # transform_input should be turned off in torchvision model if this transform is used.
            shape = (1, 3, 1, 1)
            scale = torch.tensor([0.229 / 0.5, 0.224 / 0.5, 0.225 / 0.5]).reshape(shape)
            bias = torch.tensor(
                [(0.485 - 0.5) / 0.5, (0.456 - 0.5) / 0.5, (0.406 - 0.5) / 0.5]
            ).reshape(shape)
            image_tensor = image_tensor * scale + bias

        return self.net(image_tensor)

    def get_evaluator(self) -> BaseEvaluator:
        return ClassificationEvaluator()

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "image_tensor": TensorSpec(
                shape=(batch_size, 3, self.INPUT_DIM, self.INPUT_DIM),
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
            "class_logits": TensorSpec(
                io_type=IoType.TENSOR,
                labels_file="imagenet_labels.txt",
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> dict[str, list[np.ndarray]]:
        image = load_image(TEST_IMAGENET_IMAGE)
        tensor = self.INPUT_TRANSFORM(image).unsqueeze(0)
        return dict(image_tensor=[tensor.numpy()])

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ImagenetDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return OpenImagesV7Dataset

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("imagenet", output_dir, metadata)


class ImagenetClassifierWithModelBuilder(ImagenetClassifier):
    model_builder: Callable
    DEFAULT_WEIGHTS: str

    @classmethod
    def from_pretrained(
        cls,
        weights: str | None = None,
    ) -> Self:
        """
        Load the model with pretrained weights.

        Parameters
        ----------
        weights
            Pre-trained weights name string or URL.

        Returns
        -------
        model: Self
            The model loaded with pretrained weights.
        """
        net = cls.model_builder(weights=weights or cls.DEFAULT_WEIGHTS)
        return cls(net)
