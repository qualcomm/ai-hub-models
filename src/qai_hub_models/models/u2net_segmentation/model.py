# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.duts.duts import DUTSDataset, preprocess_image_for_u2net
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.u2net_segmentation.evaluator import (
    MeanAbsoluteErrorEvaluator,
)
from qai_hub_models.models.u2net_segmentation.external_repos.u2net.model.u2net_refactor import (
    U2NET_full,
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

DEFAULT_WEIGHTS = "u2net.pth"

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "u2net_test_image.png"
)


# Note: _shared/segmentation App and SegmentationOutputEvaluator (mIoU) are not reused
# because U2-Net uses salient-object MAE metric (lower=better, range [0,1]) which measures
# per-pixel absolute error against binary GT masks — the standard benchmark for salient
# object detection. This is fundamentally different from mIoU used in shared segmentation.
class SegmentationU2Net(BaseModel):
    """
    U²-Net for salient object segmentation / background removal.

    Implements a two-level nested U-structure (RSU blocks) that produces
    high-quality foreground saliency maps at 320x320 resolution.

    Reference: https://arxiv.org/abs/2005.09007
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        weights: str | None = DEFAULT_WEIGHTS,
    ) -> Self:
        """Load U²-Net with pre-trained salient object detection weights."""
        net = U2NET_full()

        if weights is not None:
            checkpoint_path = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, weights
            ).fetch()
            state_dict = load_torch(checkpoint_path)
            net.load_state_dict(state_dict)

        return cls(net)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run U²-Net on `image`, produce a salient object segmentation mask.

        Parameters
        ----------
        image
            [1, 3, H, W] image tensor.
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            Color Space: RGB

        Returns
        -------
        torch.Tensor
            [1, 1, H, W] saliency map — values in [0, 1].
            1.0 = foreground (salient object), 0.0 = background.
        """
        # Divide-by-max normalization matching paper ToTensorLab(flag=0)
        max_val = image.amax(dim=(1, 2, 3), keepdim=True)
        image = torch.where(max_val > 1e-6, image / max_val, image)
        image = normalize_image_torchvision(image)
        outputs = self.model(image)
        return outputs[0]

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 320,
        width: int = 320,
    ) -> InputSpec:
        """
        Returns the input specification (name -> TensorSpec).
        Used to submit profiling/compile jobs on Qualcomm AI Hub.
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
                apply_runtime_channel_reordering=False,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if input_spec is None:
            input_spec = self.get_input_spec()
        h, w = input_spec["image"][0][2:]
        image = load_image(IMAGE_ADDRESS)
        return {"image": [preprocess_image_for_u2net(image, h, w).numpy()]}

    def get_evaluator(self) -> BaseEvaluator:
        return MeanAbsoluteErrorEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [DUTSDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return DUTSDataset
