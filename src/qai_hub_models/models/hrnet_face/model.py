# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models.models.hrnet_face.dataset import (
    CocoFaceCalibrationDataset,
    COFWDataset,
)
from qai_hub_models.models.hrnet_face.evaluator import HRNetFaceEvaluator
from qai_hub_models.models.hrnet_face.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.hrnet_face.external_repos.hrnet_facial_landmark_detection.lib.config import (
    config,
)
from qai_hub_models.models.hrnet_face.external_repos.hrnet_facial_landmark_detection.lib.models.hrnet import (
    get_face_alignment_net,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
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

# Model downloaded from https://onedrive.live.com/?authkey=%21AFIsEUQl8jgUaMk&id=735C9ADA5267A325%21116&cid=735C9ADA5267A325&parId=root&parQt=sharedby&o=OneUp
DEFAULT_WEIGHTS = "HR18-COFW"
MODEL_ASSET_VERSION = 1
DEFAULT_CONFIG = "face_alignment_cofw_hrnet_w18.yaml"

DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "HR18-COFW.pth"
)


class HRNetFace(BaseModel):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls, weights_name: str = DEFAULT_WEIGHTS, config_name: str = DEFAULT_CONFIG
    ) -> Self:
        weights = load_torch(weights_name)

        repo_dir = EXTERNAL_REPO_PATHS["hrnet_facial_landmark_detection"]
        config_file = Path(repo_dir) / "experiments" / "cofw" / config_name

        if not config_file.exists():
            raise ValueError(f"Config file not found: {config_file}")

        config.defrost()
        config.merge_from_file(str(config_file))
        config.freeze()

        net = get_face_alignment_net(config)
        net.load_state_dict(weights)
        return cls(net).eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict facial keypoints heatmaps from an input image.

        Parameters
        ----------
        image
            Input image as a torch.Tensor,
            shape [B, 3, H, W], Pixel values in [0, 1].

        Returns
        -------
        heatmaps : torch.Tensor
            Heatmaps of shape [B, 29, 64, 64], where 29 is the number of keypoints,
            containing probability distributions for keypoint locations.
        """
        return self.model(normalize_image_torchvision(image))

    def get_output_spec(self) -> OutputSpec:
        return {
            "heatmaps": TensorSpec(),
        }

    def get_evaluator(self) -> BaseEvaluator:
        return HRNetFaceEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [COFWDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoFaceCalibrationDataset

    def get_input_spec(
        self,
        height: int = 256,
        width: int = 256,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type) of the face landmark detector.
        This can be used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "image": TensorSpec(
                shape=(1, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }
