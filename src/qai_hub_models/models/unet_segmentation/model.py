# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.segmentation.segmentation_evaluator import (
    SegmentationOutputEvaluator,
)
from qai_hub_models.models.unet_segmentation.dataset import (
    CarvanaDataset,
    UNetCalibrationDataset,
)
from qai_hub_models.models.unet_segmentation.external_repos.pytorch_unet.unet import (
    UNet as _UNet,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_torch,
)
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
MODEL_ASSET_VERSION = 1
# from https://github.com/milesial/Pytorch-UNet/releases/download/v3.0/unet_carvana_scale1.0_epoch2.pth
DEFAULT_WEIGHTS = "unet_carvana_scale1.0_epoch2.pth"
IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "unet_test_image.jpg"
)


class UNet(BaseModel):
    @classmethod
    def from_pretrained(cls, weights: str | None = DEFAULT_WEIGHTS) -> Self:
        net = _UNet(n_channels=3, n_classes=2, bilinear=False)
        if weights is not None:
            checkpoint_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, weights
            ).fetch()
            state_dict = load_torch(checkpoint_path)
            net.load_state_dict(state_dict)
        return cls(net)

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str | None = None
    ) -> str:
        # min_max avoids tf_enhanced clipping UNet's wide-range edge activations (~3pt mIOU on w8a8).
        options = other_options or ""
        if "--range_scheme" in options:
            return options
        return options + " --range_scheme min_max"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run UNet on `image`, and produce a segmentation mask over the image.

        Parameters
        ----------
        image
            A [1, 3, H, W] image. The smaller of H, W should be >= 16, the larger should be >=32.
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        mask : torch.Tensor
            Segmentation mask of shape [1, n_classes, H, W] where H, W are the same as the input image.
            n_classes is 2 for the default model.
            Each channel represents the raw logit predictions for a given class.
            Taking the softmax over all channels for a given pixel gives the
            probability distribution over classes for that pixel.
        """
        return self.model(image)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 640,
        width: int = 1280,
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
            "mask": TensorSpec(
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

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(num_classes=2, resize_to_gt=True)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CarvanaDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return UNetCalibrationDataset
