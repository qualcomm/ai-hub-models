# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models.mobile_facenet.dataset import LFWDataset
from qai_hub_models.models.mobile_facenet.evaluator import (
    MobileFaceNetEvaluator,
)
from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.mobilefacenet import (
    MobileFaceNet as SourceMobileFaceNet,
)
from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.retinaface.detector import (
    detector,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.retinaface.model import (
    DEFAULT_DETECTOR_HEIGHT,
    DEFAULT_DETECTOR_WIDTH,
    RetinaFaceDetectorBase,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_torch,
)
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
MODEL_ASSET_VERSION = 1

# Weights taken from https://github.com/foamliu/MobileFaceNet/blob/master/weights/mobilefacenet.pt
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "mobilefacenet.pt"
)
DEFAULT_THRESHOLD = 74.18

# Minimum RetinaFace detection confidence. Matches the confidence_threshold
# default of the upstream RetinafaceDetector.detect_faces().
SCORE_THRESHOLD = 0.9


class RetinaFaceDetector(RetinaFaceDetectorBase):
    """
    RetinaFace face detector component.

    Weights come from the vendored MobileFaceNet repo rather than the
    ``face_detection`` pip package, so only ``from_pretrained`` differs from the
    shared base.
    """

    @classmethod
    def from_pretrained(cls) -> Self:
        model = detector.model
        model.eval()
        return cls(model)


class FaceRecognizer(BaseModel):
    """MobileFaceNet embedding backbone, run on an aligned face crop."""

    @classmethod
    def from_pretrained(
        cls, weights_name: str | CachedWebModelAsset = DEFAULT_WEIGHTS
    ) -> Self:
        checkpoint = load_torch(weights_name)
        net = SourceMobileFaceNet()
        net.load_state_dict(checkpoint)
        return cls(net)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for face verification.

        Each image is run through the backbone twice (original + horizontal flip)
        for test-time augmentation. The two embedding vectors are summed per image
        then interleaved so the evaluator can split them with stride-2 indexing.

        Parameters
        ----------
        img1
            RGB face tensor of shape ``[B, 3, H, W]`` with values in ``[0, 1]``.
        img2
            Second RGB face tensor of shape ``[B, 3, H, W]`` with values in ``[0, 1]``.

        Returns
        -------
        embeddings : torch.Tensor
            Shape ``[B*2, 128]``, interleaved: even rows (0, 2, …) are TTA embeddings
            for img1; odd rows (1, 3, …) are TTA embeddings for img2.
        """
        img1_norm = normalize_image_torchvision(img1)
        img2_norm = normalize_image_torchvision(img2)
        img1_flip = torch.flip(img1_norm, dims=[-1])
        img2_flip = torch.flip(img2_norm, dims=[-1])
        B = img1_norm.size(0)
        imgs = torch.cat([img1_norm, img1_flip, img2_norm, img2_flip], dim=0)
        out = self.model(imgs)  # (4B, 128)
        e1 = out[:B] + out[B : 2 * B]  # TTA sum for img1
        e2 = out[2 * B : 3 * B] + out[3 * B :]  # TTA sum for img2
        return torch.stack([e1, e2], dim=1).reshape(B * 2, -1)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 112,
        width: int = 112,
    ) -> InputSpec:
        shape = (batch_size, 3, height, width)

        return {
            "img1": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
            "img2": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "embeddings": TensorSpec(),
        }

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [LFWDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return LFWDataset

    def get_evaluator(self) -> BaseEvaluator:
        return MobileFaceNetEvaluator(threshold=DEFAULT_THRESHOLD)


class MobileFaceNet(WorkbenchModelCollection):
    """
    Two-component face verification pipeline:
      1. RetinaFaceDetector — detects face boxes and 5-point landmarks in the scene image.
      2. FaceRecognizer     — embeds each aligned face crop.

    Both components are compiled and profiled, so the published figure covers the
    whole pipeline. LFW evaluation aligns from the dataset's own landmarks (the
    standard verification protocol), so the detector is exported but not part of
    the accuracy path.
    """

    def __init__(
        self,
        face_detector: RetinaFaceDetector,
        face_recognizer: FaceRecognizer,
    ) -> None:
        super().__init__(
            {"face_detector": face_detector, "face_recognizer": face_recognizer}
        )
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = DEFAULT_DETECTOR_HEIGHT,
        detector_width: int = DEFAULT_DETECTOR_WIDTH,
        height: int = 112,
        width: int = 112,
    ) -> ComponentGroup[InputSpec]:
        return super().get_input_spec(
            batch_size=batch_size,
            detector_height=detector_height,
            detector_width=detector_width,
            height=height,
            width=width,
        )

    @classmethod
    def from_pretrained(
        cls,
        weights_name: str | CachedWebModelAsset = DEFAULT_WEIGHTS,
    ) -> Self:
        return cls(
            RetinaFaceDetector.from_pretrained(),
            FaceRecognizer.from_pretrained(weights_name),
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [LFWDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return LFWDataset

    def get_evaluator(self) -> BaseEvaluator:
        return MobileFaceNetEvaluator(threshold=DEFAULT_THRESHOLD)
