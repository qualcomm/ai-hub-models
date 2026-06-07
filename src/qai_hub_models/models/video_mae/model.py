# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.kinetics400 import Kinetics400Dataset
from qai_hub_models.models._shared.video_classifier.model import (
    INPUT_VIDEO_PATH,
    KineticsClassifier,
)
from qai_hub_models.models._shared.video_classifier.utils import (
    DEFAULT_NUM_VIEWS,
    preprocess_video_224,
    read_video_per_second,
)
from qai_hub_models.models.video_mae.external_repos.videomae.modeling_finetune import (
    vit_base_patch16_224,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec, IoType, TensorSpec

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
        Predict class logits for a multi-view video.

        Parameters
        ----------
        video
            Shape ``[B, V*3, T, H, W]`` where ``V`` is the number of views.

        Returns
        -------
        class_logits : torch.Tensor
            Shape ``[B, 400]`` — summed softmax probabilities across all views.
        """
        # [B, V*3, T, H, W] -> [B*V, 3, T, H, W] with explicit sizes so the
        # torchscript tracer keeps the output rank for ONNX export.
        B, VC, T, H, W = video.shape
        V = VC // 3
        flat = video.view(B * V, 3, T, H, W)
        flat = normalize_image_torchvision(
            flat, image_tensor_has_batch=True, is_video=True
        )
        # Cast input to model's parameter dtype to avoid dtype mismatch
        param_dtype = next(self.model.parameters()).dtype
        flat = flat.to(dtype=param_dtype)
        logits = self.model(flat)
        probs = torch.softmax(logits, dim=1)
        return probs.view(B, V, -1).sum(dim=1)

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_frames: int = 16,
        num_views: int = DEFAULT_NUM_VIEWS,
    ) -> InputSpec:
        return {
            "video": TensorSpec(
                shape=(batch_size, num_views * 3, num_frames, 224, 224),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        input_tensor = read_video_per_second(str(INPUT_VIDEO_PATH.fetch()))
        input_tensor = preprocess_video_224(input_tensor, short_side_size=320)
        num_views = DEFAULT_NUM_VIEWS
        if input_spec:
            num_frames = input_spec["video"][0][2]
            num_views = input_spec["video"][0][1] // 3
            input_tensor = input_tensor[:, :num_frames]
        C, T, H, W = input_tensor.shape
        return {
            "video": [
                input_tensor.unsqueeze(0)
                .expand(num_views, -1, -1, -1, -1)
                .reshape(num_views * C, T, H, W)
                .unsqueeze(0)
                .numpy()
            ]
        }

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Kinetics400Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return Kinetics400Dataset
