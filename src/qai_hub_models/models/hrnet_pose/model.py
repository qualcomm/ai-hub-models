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
from qai_hub_models.models.hrnet_pose.dataset import CocoHRNetPoseDataset
from qai_hub_models.models.hrnet_pose.evaluator import (
    HRNetPoseEvaluator,
    HRNetPosePrecomputedBoxEvaluator,
)
from qai_hub_models.models.hrnet_pose.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.hrnet_pose.external_repos.hrnet.lib.config import cfg
from qai_hub_models.models.hrnet_pose.external_repos.hrnet.lib.models.pose_hrnet import (
    PoseHighResolutionNet,
)
from qai_hub_models.models.rtmdet.model import RTMDet
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.pose.pose_evaluator import MPIIPoseEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_numpy
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
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

# RTMDet-m is trained and configured at 640x640; the decode in RTMDet.forward
# assumes those stride grids (80/40/20), so this is not freely adjustable.
DEFAULT_DETECTOR_HEIGHT = 640
DEFAULT_DETECTOR_WIDTH = 640


class RTMDetPersonDetector(RTMDet):
    """
    RTMDet-m person detector, as a CollectionModel component.

    mmpose's MMPoseInferencer silently resolves every body/human/wholebody config to
    RTMDet-m @ 640x640 and runs it in eager PyTorch on the host. models/rtmdet
    already wraps that exact checkpoint (rtmdet_m_8xb32-300e_coco) as an exportable
    graph, so this subclass only narrows it to the person class and fixes the input
    resolution -- deliberately reusing that recipe rather than duplicating its
    ~80-line head decode here.

    Takes an RGB image tensor (float32, [0, 1]) and returns decoded
    (boxes, scores, class_idx). Filtering to the person class and picking the
    best box happen on the host -- see
    qai_hub_models.models.templates.pose.topdown.select_person_box.
    """

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = DEFAULT_DETECTOR_HEIGHT,
        detector_width: int = DEFAULT_DETECTOR_WIDTH,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, detector_height, detector_width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(color_format=ColorFormat.RGB),
                apply_runtime_channel_reordering=True,
            )
        }


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


class PoseEstimator(BaseModel):
    """HRNet top-down pose estimator, run on a person crop."""

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
        return HRNetPosePrecomputedBoxEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoDetectorKeypointsDataset, MPIIDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDetectorKeypointsDataset


class HRNetPose(WorkbenchModelCollection):
    """
    Two-component top-down pose pipeline:
      1. RTMDetPersonDetector — finds the person box in the scene image.
      2. PoseEstimator        — predicts keypoint heatmaps from the person crop.

    Both components are compiled and profiled, so the published figures cover the
    whole deployable pipeline. Previously the detector ran uncompiled in eager
    PyTorch inside the app layer and was excluded from the numbers entirely.

    Accuracy runs both components on full COCO scene images.
    """

    def __init__(
        self,
        person_detector: RTMDetPersonDetector,
        pose_estimator: PoseEstimator,
    ) -> None:
        super().__init__(
            {"person_detector": person_detector, "pose_estimator": pose_estimator}
        )
        self.person_detector = person_detector
        self.pose_estimator = pose_estimator
        self.variant = pose_estimator.variant

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = DEFAULT_DETECTOR_HEIGHT,
        detector_width: int = DEFAULT_DETECTOR_WIDTH,
        height: int = 256,
        width: int = 192,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            detector_height=detector_height,
            detector_width=detector_width,
            height=height,
            width=width,
        )

    @classmethod
    def from_pretrained(cls, variant: str = DEFAULT_VARIANT) -> Self:
        return cls(
            RTMDetPersonDetector.from_pretrained(),
            PoseEstimator.from_pretrained(variant),
        )

    def get_evaluator(self) -> BaseEvaluator:
        if self.variant == "mpii":
            return MPIIPoseEvaluator()
        return HRNetPoseEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [CocoHRNetPoseDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDetectorKeypointsDataset
