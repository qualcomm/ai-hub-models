# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.bsd import BSD300Dataset
from qai_hub_models.datasets.set14 import Set14Dataset
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.super_resolution.superres_evaluator import (
    SuperResolutionOutputEvaluator,
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
DEFAULT_SCALE_FACTOR = 4
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "super_resolution_input.jpg"
)


def validate_scale_factor(scale_factor: int) -> None:
    """Only these scales have pre-trained checkpoints available."""
    valid_scales = [2, 3, 4]
    assert scale_factor in valid_scales, "`scale_factor` must be in : " + ", ".join(
        [str(s) for s in valid_scales]
    )


class SuperResolutionModel(BaseModel):
    """Base Model for Super Resolution."""

    def __init__(
        self,
        model: torch.nn.Module,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.scale_factor = scale_factor

    def get_evaluator(self) -> BaseEvaluator:
        return SuperResolutionOutputEvaluator()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run Super Resolution on `image`, and produce an upscaled image.

        Parameters
        ----------
        image
            Pixel values pre-processed for model consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        upscaled_image : torch.Tensor
            Pixel values
            Range: float[0, 1]
            3-channel Color Space: RGB
        """
        return self.model(image)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
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
            "upscaled_image": TensorSpec(
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Set14Dataset, BSD300Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return BSD300Dataset
