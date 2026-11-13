# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.kinetics400 import (
    Kinetics400Dataset,
    preprocess_video_224,
    read_video_per_second,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.video_classifier.model import (
    INPUT_VIDEO_PATH,
    KineticsClassifier,
)
from qai_hub_models.models.video_mae.external_repos.videomae.modeling_finetune import (
    vit_base_patch16_224,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# MCG-NJU/VideoMAE ViT-B, 1600 epoch finetune on Kinetics-400, 81.5% top-1.
# https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "checkpoint.pth"
)


class VideoMAE(KineticsClassifier):
    @classmethod
    def from_pretrained(
        cls,
        weights: Any = None,
    ) -> Self:
        ckpt_path = str(DEFAULT_WEIGHTS.fetch()) if weights is None else str(weights)
        checkpoint = load_torch(ckpt_path)
        state_dict = checkpoint.get("module", checkpoint)

        net = vit_base_patch16_224(
            pretrained=False,
            num_classes=400,
            all_frames=16,
            tubelet_size=2,
            use_mean_pooling=True,
        )
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys loading VideoMAE: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys loading VideoMAE: {missing}")
        return cls(net)

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
        video = normalize_image_torchvision(
            video, image_tensor_has_batch=True, is_video=True
        )
        logits = self.model(video)
        return torch.softmax(logits, dim=1)

    def get_input_spec(
        self,
        num_frames: int = 16,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type)). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "video": TensorSpec(
                shape=(1, 3, num_frames, 224, 224),
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
        input_tensor = preprocess_video_224(input_tensor).unsqueeze(0)
        if input_spec:
            num_frames = input_spec["video"][0][2]
            input_tensor = input_tensor[:, :, :num_frames]
        return {"video": [input_tensor.numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Kinetics400Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Kinetics400Dataset
