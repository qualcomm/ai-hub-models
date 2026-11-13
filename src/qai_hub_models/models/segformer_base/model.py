# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from transformers import SegformerForSemanticSegmentation
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.ade20k import ADESegmentationDataset
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
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

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 2
DEFAULT_WEIGHTS = "nvidia/segformer-b0-finetuned-ade-512-512"
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "image_512.jpg"
)
NUM_CLASSES = 150


class SegformerBase(BaseModel):
    """Segformer Base segmentation"""

    @classmethod
    def from_pretrained(cls, ckpt: str = DEFAULT_WEIGHTS) -> Self:
        net = SegformerForSemanticSegmentation.from_pretrained(ckpt)
        return cls(net)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic segmentation an input `image`.

        Parameters
        ----------
        image_tensor
            A [1, 3, height, width] image.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        class_logits : torch.Tensor
            Raw logit probabilities as a tensor of shape
            [1, num_classes, modified_height, modified_width],
            where the modified height and width will be some factor smaller
            than the input image.
        """
        # self.model()... returns a tuple[Tensor] of length 1
        return self.model(normalize_image_torchvision(image_tensor), return_dict=False)[
            0
        ]

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
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
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "class_logits": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(NUM_CLASSES, resize_to_gt=True)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ADESegmentationDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return ADESegmentationDataset
