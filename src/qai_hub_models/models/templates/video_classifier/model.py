# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import torch

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.kinetics400 import (
    Kinetics400Dataset,
    preprocess_video_kinetics_400,
    read_video_per_second,
)
from qai_hub_models.models.templates.video_classifier.video_classification_evaluator import (
    VideoClassificationEvaluator,
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

MODEL_ID = "video_classifier"
MODEL_ASSET_VERSION = 1

INPUT_VIDEO_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "surfing_cutback.mp4"
)


class SimpleAvgPool(torch.nn.Module):
    """
    Replacement for Global Average Pool that's numerically equivalent to the one used
    in the torchvision models. It operates in rank 3 instead of rank 5 which makes it
    more NPU friendly.
    """

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        return tensor.reshape(shape[0], shape[1], -1).mean(dim=2, keepdim=False)


class KineticsClassifier(BaseModel):
    """Base class for all Kinetics Classifier models within QAI Hub Models."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        # TODO: rename input_mean/input_std back to mean/std when
        # #https://github.com/pytorch/pytorch/issues/168211 is fixed
        self.input_mean = torch.Tensor([0.43216, 0.394666, 0.37645]).reshape(
            1, 3, 1, 1, 1
        )
        self.input_std = torch.Tensor([0.22803, 0.22145, 0.216989]).reshape(
            1, 3, 1, 1, 1
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for an input `video`.

        Parameters
        ----------
        video
            A [B, C, Number of frames, H, W] video.
            Assumes video has been resized and normalized to range [0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        class_probs : torch.Tensor
            A [B, 400] tensor of per-class softmax probabilities for the
            video belonging to the corresponding Kinetics class.
        """
        video = (video - self.input_mean) / self.input_std
        logits = self.model(video)
        return torch.softmax(logits, dim=1)

    def get_input_spec(
        self,
        num_frames: int = 16,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.

        Parameters
        ----------
        num_frames
            Number of frames in the video input.

        Returns
        -------
        input_spec : InputSpec
            Input specification mapping input names to (shape, type) tuples.
        """
        return {
            "video": TensorSpec(
                shape=(1, 3, num_frames, 112, 112),
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
        input_tensor = read_video_per_second(str(INPUT_VIDEO_PATH.fetch()))
        input_tensor = preprocess_video_kinetics_400(input_tensor).unsqueeze(0)
        if input_spec:
            num_frames = input_spec["video"][0][2]
            input_tensor = input_tensor[:, :, :num_frames]
        return {"video": [input_tensor.numpy()]}

    def get_output_spec(self) -> OutputSpec:
        return {
            "class_probs": TensorSpec(
                io_type=IoType.TENSOR,
                softmax_applied=True,
                labels_file="kinetics400_labels.txt",
            ),
        }

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("kinetics400", output_dir, metadata)

    def get_evaluator(self) -> BaseEvaluator:
        return VideoClassificationEvaluator(num_classes=400)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Kinetics400Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Kinetics400Dataset
