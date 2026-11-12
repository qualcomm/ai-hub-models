# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import pathlib

import face_detection as _fd_pkg
import torch
from face_detection.alignment import load_net
from typing_extensions import Self

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.models.sixd_repnet.external_repos.sixdrepnet.sixdrepnet.model import (
    SixDRepNet as UpstreamSixDRepNet,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.export.result import ComponentGroup
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Fixed input resolution for the pose estimator, matching the upstream training config.
# See https://github.com/thohemp/6DRepNet/blob/master/sixdrepnet/config.py
INPUT_IMAGE_DIM = 224

DEFAULT_WEIGHTS = "6DRepNet_300W_LP_AFLW2000"

# Originally from "https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth"
DEFAULT_WEIGHTS_FILE = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    f"{DEFAULT_WEIGHTS}.pth",
)

# Minimum RetinaFace detection confidence. Matches the threshold used in the
# upstream 6DRepNet demo: https://github.com/thohemp/6DRepNet/blob/master/demo.py
SCORE_THRESHOLD = 0.95

# BGR mean used by the upstream RetinaFace MobileNet, matching batch_detect() in
# face_detection/alignment.py: mean = [104, 117, 123]
_RETINA_BGR_MEAN = torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1)


class RetinaFaceDetector(BaseModel):
    """
    RetinaFace face detector component.

    Wraps the RetinaFace MobileNet model from the face_detection package as a
    BaseModel so it can participate in a CollectionModel pipeline.

    Accepts an RGB image tensor (float32, [0, 1]) and returns the raw network
    outputs: (bbox_regressions, classifications, landmark_regressions).
    Post-processing (anchor decoding, NMS, score filtering) is performed in
    the application layer (SixDRepNetApp).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.detector_model = model

    @classmethod
    def from_pretrained(
        cls,
        gpu_id: int = -1,
        network: str = "mobilenet",
    ) -> Self:
        device = torch.device("cpu") if gpu_id == -1 else torch.device("cuda", gpu_id)
        fd_dir = os.path.dirname(_fd_pkg.__file__ or "")
        model_path = os.path.join(
            fd_dir,
            "weights",
            "mobilenet0.25_Final.pth"
            if network == "mobilenet"
            else "Resnet50_Final.pth",
        )
        net = load_net(model_path, device, network)
        return cls(net)

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run RetinaFace on an RGB image tensor and return raw network outputs.

        Parameters
        ----------
        image
            RGB image tensor of shape (1, 3, H, W), float32, values in [0, 1].

        Returns
        -------
        bbox_regressions : torch.Tensor
            Shape (1, num_anchors, 4) — raw box regression deltas.
        classifications : torch.Tensor
            Shape (1, num_anchors, 2) — softmax class scores [background, face].
        landmark_regressions : torch.Tensor
            Shape (1, num_anchors, 10) — raw landmark regression deltas.
        """
        # Convert RGB [0,1] float → BGR [0,255] float, matching batch_detect() preprocessing.
        img_bgr = torch.flip(image, dims=[1]) * 255.0
        mean = _RETINA_BGR_MEAN.to(img_bgr.device, dtype=img_bgr.dtype)
        img_bgr = img_bgr - mean
        bbox_regressions, classifications, landmark_regressions = self.detector_model(
            img_bgr
        )
        return bbox_regressions, classifications, landmark_regressions

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = 640,
        detector_width: int = 640,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, detector_height, detector_width),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "bbox_regressions": TensorSpec(),
            "classifications": TensorSpec(),
            "landmark_regressions": TensorSpec(),
        }


class PoseEstimator(BaseModel):
    """
    6DRepNet head pose estimation model.

    Uses a RepVGG-B1g2 backbone with a 6D rotation representation head.
    Accepts a face-crop image and returns a 3x3 rotation matrix.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls, weights_name: str = DEFAULT_WEIGHTS) -> Self:
        model = UpstreamSixDRepNet(
            backbone_name="RepVGG-B1g2",
            backbone_file="",
            deploy=True,
            pretrained=False,
        )

        weights_file = weights_name
        if weights_name == DEFAULT_WEIGHTS:
            weights_file = DEFAULT_WEIGHTS_FILE.fetch()

        weights_path = pathlib.Path(weights_file).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Estimate head pose from a face crop.

        Parameters
        ----------
        image
            Face crop tensor of shape (batch, 3, 224, 224), float32, RGB, values in [0, 1].
            ImageNet normalization is applied internally.

        Returns
        -------
        torch.Tensor
            Rotation matrices of shape (batch, 3, 3).
        """
        return self.model(normalize_image_torchvision(image))

    def get_input_spec(
        self,
        batch_size: int = 1,
        pose_height: int = INPUT_IMAGE_DIM,
        pose_width: int = INPUT_IMAGE_DIM,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, pose_height, pose_width),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "rotation_matrix": TensorSpec(),
        }


class SixDRepNet(WorkbenchModelCollection):
    """
    Two-component head pose estimation pipeline:
      1. RetinaFaceDetector — detects face bounding boxes in the scene image.
      2. PoseEstimator      — estimates head pose (rotation matrix) from each face crop.
    """

    def __init__(
        self,
        face_detector: RetinaFaceDetector,
        pose_estimator: PoseEstimator,
    ) -> None:
        super().__init__(
            {"face_detector": face_detector, "pose_estimator": pose_estimator}
        )
        self.face_detector = face_detector
        self.pose_estimator = pose_estimator

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = 640,
        detector_width: int = 640,
        pose_height: int = INPUT_IMAGE_DIM,
        pose_width: int = INPUT_IMAGE_DIM,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            detector_height=detector_height,
            detector_width=detector_width,
            pose_height=pose_height,
            pose_width=pose_width,
        )

    @classmethod
    def from_pretrained(
        cls,
        gpu_id: int = -1,
        pose_weights: str = DEFAULT_WEIGHTS,
    ) -> Self:
        return cls(
            RetinaFaceDetector.from_pretrained(gpu_id=gpu_id),
            PoseEstimator.from_pretrained(pose_weights),
        )
