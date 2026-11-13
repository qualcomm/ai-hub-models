# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.foot_track_net.dataset import FootTrackDataset
from qai_hub_models.models.foot_track_net.evaluator import (
    FootTrackNetEvaluator,
)
from qai_hub_models.models.foot_track_net.layers import (
    CBAModule,
    DetectModule,
    HeadModule,
    Mbv3SmallFast,
    UpModule,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
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
from qai_hub_models.utils.labels import write_labels_file

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_WEIGHTS = "SA-e30_finetune50_conv_norm.pth"


class FootTrackNet(BaseModel):
    """FootTrackNet model"""

    def __init__(
        self,
        wide: int = 64,
        has_ext: bool = True,
        upmode: str = "UCBA",
        act: str = "relu",
        RGB: bool = True,
        strict: bool = False,
        n_lmk: int = 17,
    ) -> None:
        """
        FootTrackNet multi task human detector model for person, face detection plus head and feet landmark detection.
        Draw the given points on the frame.
        It takes RGB (uint8) as input directly.

        Parameters
        ----------
        wide
            The channel size of bandwith of the intermediate layers
        has_ext
            If add extension layer in the head module.
        upmode
            Upsampling mode.
        act
            Activation function.
        RGB
            If the input is a 3 channel RGB
        strict
            If load the model weights in a strict way
        n_lmk
            The number of landmarks for detection.
        """
        super().__init__()
        self.use_rgb = RGB
        self.strict = strict

        # internal normalization layer,  RGB->BGR, as a conv norm
        self.conv_layer = nn.Conv2d(3, 3, 1, stride=1, padding=0, bias=False)
        # norm layer
        self.bn2 = nn.BatchNorm2d(3)

        # define backbone
        self.bb = Mbv3SmallFast(act, RGB)

        # Get the number of branch node channels stride 4, 8, 16
        c0, c1, c2 = self.bb.uplayer_shape
        act = "relu" if act == "hswish" else act
        self.conv3 = CBAModule(
            self.bb.output_channels,
            wide,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            act=act,
        )  # s32
        self.connect0 = CBAModule(c0, wide, kernel_size=1, act=act)  # s4
        self.connect1 = CBAModule(c1, wide, kernel_size=1, act=act)  # s8
        self.connect2 = CBAModule(
            c2, wide, kernel_size=1, act=act
        )  # s16, conv, batchnorm activation.

        self.up0 = UpModule(
            wide, wide, kernel_size=2, stride=2, mode=upmode, act=act
        )  # s16 nearest
        self.up1 = UpModule(
            wide, wide, kernel_size=2, stride=2, mode=upmode, act=act
        )  # s8
        self.up2 = UpModule(
            wide, wide, kernel_size=2, stride=2, mode=upmode, act=act
        )  # s4
        self.detect = DetectModule(wide, act=act)

        self.heatmap1 = HeadModule(wide, 1, has_ext=has_ext)
        self.box1 = HeadModule(wide, 4, has_ext=has_ext)
        self.heatmap2 = HeadModule(wide, 1, has_ext=has_ext)
        self.box2 = HeadModule(wide, 4, has_ext=has_ext)
        self.heatmap3 = HeadModule(wide, 1, has_ext=has_ext)
        self.box3 = HeadModule(wide, 4, has_ext=has_ext)

        self.landmark = HeadModule(wide, 2 * n_lmk, has_ext=has_ext)
        self.landmark_vis = HeadModule(wide, n_lmk, has_ext=has_ext)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward computation of FootTrackNet.

        Parameters
        ----------
        x
            Input image. RGB, range [0 - 1], shape [N, C, H, W].

        Returns
        -------
        heatmap : torch.Tensor
            N,C,H,W the heatmap for the person/face detection.
        bbox : torch.Tensor
            N,C*4, H,W the bounding box coordinate as a map.
        landmark : torch.Tensor
            N,C*34,H,W the coordinates of landmarks as a map.
        landmark_visibility : torch.Tensor
            N,C*17,H,W the visibility of the landmark as a map.
        """
        x = x * 255.0
        x = self.conv_layer(x * 1.0)  # conv to float
        x = self.bn2(x)
        s4, s8, s16, s32 = self.bb(x)
        s32 = self.conv3(s32)
        s16 = self.up0(s32) + self.connect2(s16)
        s8 = self.up1(s16) + self.connect1(s8)
        s4 = self.up2(s8) + self.connect0(s4)
        x = self.detect(s4)

        # simplify with sigmoid
        center1 = self.heatmap1(x).sigmoid()
        center2 = self.heatmap2(x).sigmoid()
        center3 = self.heatmap3(x).sigmoid()

        box1 = self.box1(x)
        box2 = self.box2(x)
        box3 = self.box3(x)  # when demo, no hand
        landmark = self.landmark(x)  # 2 * 17
        landmark_vis = self.landmark_vis(x).sigmoid()
        return (
            torch.cat((center1, center2, center3), dim=1),
            torch.cat((box1, box2, box3), dim=1),
            landmark,
            landmark_vis,
        )  # simple one landmark

    def load_weights(self, base_file: str) -> None:
        """Load pretrained weights"""
        _, ext = os.path.splitext(base_file)
        if ext in {".pkl", ".pth"}:
            print("Loading pretrained weights into state dict...")

            pretrained_dict = torch.load(
                base_file, map_location=lambda storage, loc: storage, weights_only=False
            )
            model_dict = self.state_dict()

            if not self.strict:
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() if k in model_dict
                }
                if (
                    self.use_rgb and pretrained_dict["bb.conv1.weight"].shape[1] == 1
                ):  # single channel.
                    pretrained_dict["bb.conv1.weight"] = torch.tile(
                        pretrained_dict["bb.conv1.weight"], [1, 3, 1, 1]
                    )  # the input channel to 3.
            model_dict.update(pretrained_dict)

            self.load_state_dict(model_dict, strict=self.strict)
            print("Finished!")
        else:
            raise ValueError("Sorry only .pth and .pkl files supported.")

    @classmethod
    def from_pretrained(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, checkpoint_path: str | None = None
    ) -> Self:
        """
        Load model from pretrained weights.

        Parameters
        ----------
        checkpoint_path
            Checkpoint path of pretrained weights.

        Returns
        -------
        model : Self
            FootTrackNet model.
        """
        model = cls()
        checkpoint_to_load = (
            checkpoint_path
            if checkpoint_path is not None
            else CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()
        )

        model.load_weights(checkpoint_to_load)
        model.to(torch.device("cpu"))
        model.eval()
        return model

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench. Default resolution is 2048x1024
        so this expects an image where width is twice the height.
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
            "heatmap": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "bbox": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "landmark": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
            "landmark_visibility": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_evaluator(self) -> BaseEvaluator:
        return FootTrackNetEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [FootTrackDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return FootTrackDataset

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("foot_track_net", output_dir, metadata)
