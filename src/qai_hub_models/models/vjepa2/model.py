# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
import transformers.models.vjepa2.modeling_vjepa2 as _vjepa2_mod
from torch import nn
from transformers import VJEPA2Config, VJEPA2Model
from typing_extensions import Self

from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.vjepa2.model_patch import _patch_vjepa2_rope
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    InputSpec,
    IoType,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Facebook V-JEPA 2 ViT-L, 256px, 64-frame-per-clip pretrain. Loaded via HF transformers.
# We reduce frames_per_clip 64 -> 8 (tokens = (8/2)*(256/16)^2 = 1024) for NPU feasibility.
DEFAULT_WEIGHTS = "facebook/vjepa2-vitl-fpc64-256"

FRAMES = 8
TUBELET = 2
IMG_SIZE = 256
PATCH = 16
HIDDEN = 1024


class _VJEPA2EncoderConv2d(nn.Module):
    """
    Wraps the HF VJEPA2 encoder for NPU-friendly export.

    The stock VJEPA2 patch-embed is a Conv3d over a 5D ``(B, C, T, H, W)`` tensor.
    QNN HTP mishandles the 5D Conv3d -> Reshape boundary, so we replace it with an
    equivalent Conv2d over pre-folded tubelets, keeping every tensor <= 4D:

      video (B, 3, T, H, W)
        -> fold tubelet into batch: (B*T/TUBELET, 3*TUBELET, H, W)
        -> Conv2d(3*TUBELET -> HIDDEN, kernel=PATCH, stride=PATCH)
        -> tokens (B, num_tokens, HIDDEN)
        -> transformer layers + final layernorm
        -> last_hidden_state (B, num_tokens, HIDDEN)

    This is numerically identical to the original Conv3d encoder (verified cos == 1.0).
    """

    def __init__(self, hf_encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = hf_encoder
        enc: Any = hf_encoder
        conv3d = enc.embeddings.patch_embeddings.proj
        out_c, in_c, t, ph, pw = conv3d.weight.shape
        self.conv2d = nn.Conv2d(in_c * t, out_c, kernel_size=(ph, pw), stride=(ph, pw))
        with torch.no_grad():
            self.conv2d.weight.copy_(conv3d.weight.reshape(out_c, in_c * t, ph, pw))
            self.conv2d.bias.copy_(conv3d.bias)  # type: ignore[union-attr]
        self.tubelet = t
        self.hidden = out_c

    def _patch_embed(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B, C, T, H, W)
        b, c, t, h, w = video.shape
        n_t = t // self.tubelet
        # (B, C, n_t, tubelet, H, W) -> (B, n_t, C, tubelet, H, W) -> (B*n_t, C*tubelet, H, W)
        x = video.reshape(b, c, n_t, self.tubelet, h, w)
        x = x.permute(0, 2, 1, 3, 4, 5)
        x = x.reshape(b * n_t, c * self.tubelet, h, w)
        y = self.conv2d(x)  # (B*n_t, hidden, H/PATCH, W/PATCH)
        hh, ww = y.shape[-2], y.shape[-1]
        # channel-last collapse (layout-safe for QNN): (B*n_t, hh, ww, hidden)
        y = y.permute(0, 2, 3, 1)
        return y.reshape(b, n_t * hh * ww, self.hidden)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        hidden = self._patch_embed(video)
        enc: Any = self.encoder
        for layer in enc.layer:
            hidden = layer(hidden)[0]
        return enc.layernorm(hidden)


class VJEPA2(BaseModel):
    """
    Facebook V-JEPA 2 ViT-L video encoder (self-supervised video representation backbone).

    Outputs per-token embeddings (``last_hidden_state``) for a short video clip; used as a
    frozen backbone for downstream video understanding tasks.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__(net)

    @classmethod
    def from_pretrained(cls, weights: Any = None) -> Self:
        _patch_vjepa2_rope(_vjepa2_mod)
        weights = weights or DEFAULT_WEIGHTS
        config = VJEPA2Config.from_pretrained(weights)
        config.frames_per_clip = FRAMES
        hf_model = VJEPA2Model.from_pretrained(
            weights, config=config, ignore_mismatched_sizes=True, dtype=torch.float32
        )
        net = _VJEPA2EncoderConv2d(hf_model.encoder)
        return cls(net)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-token embeddings from a normalized video clip.

        Parameters
        ----------
        video
            Shape ``[B, 3, T, H, W]`` (channels-first, T frames), pixel values in [0, 1],
            RGB. Normalized to ImageNet stats inside the network.

        Returns
        -------
        last_hidden_state : torch.Tensor
            Shape ``[B, num_tokens, 1024]`` per-token embeddings.
        """
        video = normalize_image_torchvision(
            video, image_tensor_has_batch=True, is_video=True
        )
        return self.model(video)

    def get_input_spec(
        self,
        batch_size: int = 1,
        num_frames: int = FRAMES,
    ) -> InputSpec:
        return {
            "video": TensorSpec(
                shape=(batch_size, 3, num_frames, IMG_SIZE, IMG_SIZE),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_output_spec(
        self,
        batch_size: int = 1,
        num_frames: int = FRAMES,
    ) -> dict[str, TensorSpec]:
        num_tokens = (num_frames // TUBELET) * (IMG_SIZE // PATCH) ** 2
        return {
            "last_hidden_state": TensorSpec(
                shape=(batch_size, num_tokens, HIDDEN),
                dtype="float32",
                io_type=IoType.TENSOR,
            ),
        }

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        from qai_hub_models.models.vjepa2.dataset import Kinetics400VJEPA2Dataset

        return Kinetics400VJEPA2Dataset
