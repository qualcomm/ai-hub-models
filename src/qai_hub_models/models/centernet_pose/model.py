# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.coco import CocoBodyDataset
from qai_hub_models.models.centernet_pose.evaluator import (
    CenternetPoseEvaluator,
)
from qai_hub_models.models.templates.centernet.external_repos.centernet.src.lib.models.decode import (
    multi_pose_decode,
)
from qai_hub_models.models.templates.centernet.model import CenterNet
from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
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

IMAGE = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "image.jpg")

# checkpoint download from https://drive.google.com/file/d/1mC2PAQT_RuHi_9ZMZgkt4rg7BSY2_Lkd
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "multi_pose_dla_3x.pth",
)


class CenterNetPose(CenterNet):
    """
    CenterNetPose Object Detection.

    Parameters
    ----------
    model
        Centernet pose model.
    multi_pose_decode
        Pose detection decoder function.
    """

    def __init__(
        self,
        model: nn.Module,
        multi_pose_decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
    ) -> None:
        super().__init__()
        self.model = model
        self.decode = multi_pose_decode

    @classmethod
    def from_pretrained(cls, ckpt_path: str = "default") -> Self:
        heads = {
            "hm": 1,
            "wh": 2,
            "hps": 34,
            "reg": 2,
            "hm_hp": 17,
            "hp_offset": 2,
        }
        if ckpt_path == "default":
            ckpt_path = str(DEFAULT_WEIGHTS.fetch())
        model = super()._load_pose_net(ckpt_path, heads)

        return cls(model, multi_pose_decode)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run CenterNetPose model and returns hm, wh, hps, reg, hm_hp, hm_offset.

        Parameters
        ----------
        image
            Preprocessed image with shape [B, C, H, W] as float32 with range [0, 1] in RGB format.
            B = batch size, C = 3, H = img height, W = img width.

        Returns
        -------
        hm : torch.Tensor
            Heatmap with shape [B, num_classes, H//4, W//4].
        wh : torch.Tensor
            Width/Height value with shape [B, 2, H//4, W//4].
        hps : torch.Tensor
            Keypoint offsets relative to the object center with shape [B, 2*num_joints, H//4, W//4].
        reg : torch.Tensor
            2D regression value with shape [B, 2, H//4, W//4].
        hm_hp : torch.Tensor
            Keypoint heatmap with shape [B, num_joints, H//4, W//4].
        hm_offset : torch.Tensor
            Heatmap offset with shape [B, 2, H//4, W//4].
            Note: num_joints = 17, num_classes = 1.
        """
        image = image[:, [2, 1, 0]]
        mean = torch.Tensor([0.408, 0.447, 0.470]).reshape(1, 3, 1, 1)
        std = torch.Tensor([0.289, 0.274, 0.278]).reshape(1, 3, 1, 1)
        image = (image - mean) / std

        hm, wh, hps, reg, hm_hp, hm_offset = self.model(image)[-1].values()
        hm = torch.sigmoid(hm)
        hm_hp = torch.sigmoid(hm_hp)
        return hm, wh, hps, reg, hm_hp, hm_offset

    def get_output_spec(self) -> OutputSpec:
        return {
            "hm": TensorSpec(),
            "wh": TensorSpec(),
            "hps": TensorSpec(),
            "reg": TensorSpec(),
            "hm_hp": TensorSpec(),
            "hm_offset": TensorSpec(),
        }

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 256,
        width: int = 256,
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

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(IMAGE.fetch())
        image_array = np.array(image)
        h, w = self.get_input_spec()["image"][0][2:]
        height, width = image_array.shape[0:2]
        c = np.array([width / 2, height / 2], dtype=np.float32)
        s = np.array([max(height, width), max(height, width)], dtype=np.float32)

        img = pre_process_with_affine(image_array, c, s, 0, (h, w))
        return {
            "image": [img.numpy()],
        }

    def get_evaluator(self) -> BaseEvaluator:
        return CenternetPoseEvaluator(
            decode=self.decode,
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoBodyDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoBodyDataset
