# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.coco.coco_person_keypoints import (
    CocoDetectorKeypointsDataset,
)
from qai_hub_models.datasets.mpii import MPIIDataset
from qai_hub_models.models.hrnet_pose.evaluator import HRNetPoseEvaluator
from qai_hub_models.models.hrnet_pose.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.hrnet_pose.external_repos.hrnet.lib.config import cfg
from qai_hub_models.models.hrnet_pose.external_repos.hrnet.lib.models.pose_hrnet import (
    PoseHighResolutionNet,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.pose.pose_evaluator import MPIIPoseEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
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
DEFAULT_VARIANT = "coco"

_HRNET_REPO_ROOT = EXTERNAL_REPO_PATHS["hrnet"]

# This model originally comes from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# but we'll use the weights from AIMET
# Weights and config stored in S3 are sourced from
# https://github.com/quic/aimet-model-zoo/blob/develop/aimet_zoo_torch/hrnet_posenet/models/model_cards/hrnet_posenet_w8a8.json
# Weights are found here
# https://github.com/quic/aimet-model-zoo/releases/download/phase_2_march_artifacts/hrnet_posenet_FP32_state_dict.pth
WEIGHTS = {
    "coco": "hrnet_posenet_FP32_state_dict.pth",
    "mpii": "pose_hrnet_w32_256x256.pth",
}
CONFIG_FILE = {
    "coco": str(
        _HRNET_REPO_ROOT
        / "experiments"
        / "coco"
        / "hrnet"
        / "w32_256x192_adam_lr1e-3.yaml"
    ),
    "mpii": str(
        _HRNET_REPO_ROOT
        / "experiments"
        / "mpii"
        / "hrnet"
        / "w32_256x256_adam_lr1e-3.yaml"
    ),
}
SAMPLE_INPUTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "sample_hrnet_inputs.npy"
)


class HRNetPose(BaseModel):
    def __init__(self, model: nn.Module, variant: str) -> None:
        super().__init__()
        self.model = model
        self.variant = variant

    @classmethod
    def from_pretrained(cls, variant: str = DEFAULT_VARIANT) -> Self:
        weights_file = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, WEIGHTS[variant]
        ).fetch()
        weights = torch.load(weights_file, map_location="cpu", weights_only=False)

        cfg.defrost()
        cfg.merge_from_file(CONFIG_FILE[variant])
        cfg.freeze()
        net = PoseHighResolutionNet(cfg)
        net.load_state_dict(weights)
        return cls(net, variant)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Image inputs are expected to be in RGB format in the range [0, 1]."""
        image = normalize_image_torchvision(image)
        return self.model(image)

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        return {"image": [load_numpy(SAMPLE_INPUTS)]}

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 192,
    ) -> InputSpec:
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
            "heatmaps": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_evaluator(self) -> BaseEvaluator:
        if self.variant == "mpii":
            return MPIIPoseEvaluator()
        return HRNetPoseEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoDetectorKeypointsDataset, MPIIDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDetectorKeypointsDataset
