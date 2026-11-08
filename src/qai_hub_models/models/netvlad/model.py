# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torch import nn
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.models.netvlad.dataset import TinyTokyoTimeMachineDataset
from qai_hub_models.models.netvlad.evaluator import PlaceRecognitionEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Number of VLAD cluster centroids; matches the pytorch-NetVlad default.
NUM_CLUSTERS = 64
# VGG-16 conv5_3 output channels.
ENCODER_DIM = 512
# Default canonical input shape (H, W).
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640

DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "vgg16_netvlad_checkpoint.zip"
)


class _NetVLADLayer(nn.Module):
    """
    NetVLAD aggregation layer.

    Implements the VLAD pooling described in Arandjelovic et al. (CVPR 2016).
    Two formulations are supported, controlled by ``use_4d_vlad``:

    * ``False`` (default, float): bmm — no 4-D intermediate, faster on NPU.
    * ``True`` (quantization): original 4-D residual — bmm causes large-value
      cancellation under quantization; 4-D form subtracts before weighting.
    """

    def __init__(
        self,
        num_clusters: int,
        dim: int,
        normalize_input: bool = True,
        use_4d_vlad: bool = False,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.use_4d_vlad = use_4d_vlad
        # 1x1 conv maps descriptor dim to soft cluster assignments.
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            Dense feature map from backbone, shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            L2-normalized VLAD descriptor, shape (N, num_clusters * C).
        """
        N, C, _H, _W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)

        # soft_assign: (N, K, H*W)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        # x_flat: (N, C, H*W)
        x_flat = x.view(N, C, -1)

        if self.use_4d_vlad:
            # Original 4-D residual form — better quantization numerics.
            x_expand = x_flat.unsqueeze(1)
            centroids_expand = self.centroids.unsqueeze(0).unsqueeze(-1)
            residual = x_expand - centroids_expand
            vlad = (residual * soft_assign.unsqueeze(2)).sum(dim=-1)  # (N, K, C)
        else:
            # bmm form — no 4-D intermediate, faster on NPU at float.
            vlad = torch.bmm(soft_assign, x_flat.transpose(1, 2))  # (N, K, C)
            vlad = vlad - soft_assign.sum(
                dim=-1, keepdim=True
            ) * self.centroids.unsqueeze(0)

        # Intra-normalization, flatten, final L2 normalize.
        vlad = F.normalize(vlad, p=2, dim=2)
        return F.normalize(vlad.view(N, -1), p=2, dim=1)


class NetVLAD(BaseModel):
    """
    NetVLAD place-recognition descriptor model.

    VGG-16 backbone (truncated after conv5_3) with a NetVLAD aggregation
    layer. Produces a 32,768-d (64 clusters x 512 channels) L2-normalized
    descriptor suitable for nearest-neighbour place retrieval.

    Pretrained weights are downloaded from the asset store by default.
    """

    def __init__(self, encoder: nn.Sequential, pool: _NetVLADLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = pool

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: CachedWebModelAsset | str | None = DEFAULT_WEIGHTS,
        num_clusters: int = NUM_CLUSTERS,
        precision: Precision = Precision.float,
    ) -> Self:
        """
        Instantiate NetVLAD.

        Parameters
        ----------
        checkpoint
            Weights source. Defaults to the asset-store checkpoint (S3).
            Pass a local ``.pth.tar`` or ``.zip`` path to use a local file instead.
            Pass None for random weights (structural tests and export only).
        num_clusters
            Number of VLAD cluster centroids.
        precision
            Target precision; selects the VLAD formulation used at export.

        Returns
        -------
        Self
            Instantiated NetVLAD model.
        """
        vgg = tv_models.vgg16(weights=None)
        # Keep features[0:29] = all conv/pool through conv5_3, drop final MaxPool.
        layers = list(vgg.features.children())[:-1]
        encoder = nn.Sequential(*layers)

        pool = _NetVLADLayer(
            num_clusters=num_clusters,
            dim=ENCODER_DIM,
            use_4d_vlad=(precision != Precision.float),
        )

        model = cls(encoder=encoder, pool=pool)

        if checkpoint is not None and isinstance(checkpoint, CachedWebModelAsset):
            checkpoint = str(checkpoint.fetch())
        if checkpoint is not None:
            if checkpoint.endswith(".zip"):
                with zipfile.ZipFile(checkpoint) as z:
                    # Find the .pth.tar inside the zip (path-agnostic).
                    pth_name = next(n for n in z.namelist() if n.endswith(".pth.tar"))
                    buf = io.BytesIO(z.read(pth_name))
                    state = torch.load(buf, map_location="cpu", weights_only=False)
            else:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            # pytorch-NetVlad checkpoints carry a top-level 'state_dict' key.
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict)

        return model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image
            RGB image tensor, shape (N, 3, H, W), float32, range [0, 1].

        Returns
        -------
        torch.Tensor
            L2-normalized VLAD descriptor, shape (N, num_clusters * encoder_dim).
        """
        # ImageNet normalization — model was trained with torchvision transforms.
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        x = (image - mean) / std

        x = self.encoder(x)
        return self.pool(x)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"descriptor": TensorSpec()}

    def get_evaluator(self) -> BaseEvaluator:
        return PlaceRecognitionEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [TinyTokyoTimeMachineDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset] | None:
        return TinyTokyoTimeMachineDataset

    def get_hub_litemp_percentage(self, precision: Precision) -> float | None:
        return 5.0
