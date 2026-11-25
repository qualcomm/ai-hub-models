# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from typing_extensions import Self

from qai_hub_models.models.igev_plusplus.constants import DEFAULT_HEIGHT, DEFAULT_WIDTH
from qai_hub_models.models.igev_plusplus.dataset import SceneFlowFT3DDataset
from qai_hub_models.models.igev_plusplus.evaluator import DisparityEvaluator
from qai_hub_models.models.igev_plusplus.external_repos.igev_plusplus.core_rt.rt_igev_stereo import (
    IGEVStereo,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
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
MODEL_ASSET_VERSION = 1

# Pretrained on SceneFlow synthetic dataset (real-time variant)
# Source: https://github.com/gangweiX/IGEV-plusplus
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "igev_rt_sceneflow.pth"
)


def _build_args() -> argparse.Namespace:
    """Build the default argparse Namespace required to instantiate IGEVStereo (RT variant)."""
    return argparse.Namespace(
        hidden_dim=96,
        corr_levels=2,
        corr_radius=4,
        n_downsample=2,
        n_gru_layers=3,
        max_disp=192,
        mixed_precision=False,
        precision_dtype="float32",
    )


class IGEVPlusPlus(BaseModel):
    """IGEV++ stereo matching model for disparity estimation.

    IGEV++ (Iterative Multi-range Geometry Encoding Volumes) builds multi-range
    geometry encoding volumes to handle both small and large disparities, and
    refines disparity iteratively using a ConvGRU-based update block.

    Source: https://github.com/gangweiX/IGEV-plusplus
    """

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> Self:
        """Load IGEVPlusPlus from a pretrained checkpoint.

        Parameters
        ----------
        weights_path
            Path to a local ``.pth`` checkpoint file. If ``None``, the default
            SceneFlow pretrained weights are downloaded from the asset store.

        Returns
        -------
        Self
            Model instance ready for inference.
        """
        ckpt_path = weights_path or str(DEFAULT_WEIGHTS.fetch())
        args = _build_args()
        net = IGEVStereo(args)
        state_dict = torch.load(
            ckpt_path, map_location=torch.device("cpu"), weights_only=True
        )
        # Checkpoint was saved with torch.nn.DataParallel — strip "module." prefix
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

        net.load_state_dict(state_dict, strict=True)
        return cls(net)

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
    ) -> torch.Tensor:
        """Run IGEV++ on a rectified stereo pair and return a disparity map.

        Parameters
        ----------
        left_image
            Float32 RGB [0, 1] tensor of shape ``[B, 3, H, W]``.
            H and W must be divisible by 32.
        right_image
            Float32 RGB [0, 1] tensor of shape ``[B, 3, H, W]``.
            H and W must be divisible by 32.

        Returns
        -------
        torch.Tensor
            Disparity map of shape ``[B, 1, H, W]`` in pixels (float32).
        """
        # IGEVStereo expects pixel values in [0, 255] and normalizes internally
        # via: image = (2 * (image / 255.0) - 1.0)
        left_image = left_image * 255.0
        right_image = right_image * 255.0
        return self.model(left_image, right_image, iters=8, test_mode=True)

    def get_evaluator(self) -> BaseEvaluator:
        return DisparityEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type]:
        return [SceneFlowFT3DDataset]

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> InputSpec:
        """Return the input spec for IGEV++.

        Parameters
        ----------
        batch_size
            Number of stereo pairs per batch.
        height
            Image height in pixels. Must be divisible by 32.
        width
            Image width in pixels. Must be divisible by 32.

        Returns
        -------
        InputSpec
            Dict with ``left_image`` and ``right_image`` tensor specs.
        """
        assert height % 32 == 0, f"Height ({height}) must be divisible by 32."
        assert width % 32 == 0, f"Width ({width}) must be divisible by 32."
        assert width >= 192, f"Width ({width}) must be >= 192 for max_disp=192."
        shape = (batch_size, 3, height, width)
        rgb_metadata = ImageMetadata(color_format=ColorFormat.RGB)
        return {
            "left_image": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=rgb_metadata,
                apply_runtime_channel_reordering=True,
            ),
            "right_image": TensorSpec(
                shape=shape,
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=rgb_metadata,
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        """Return the output spec for IGEV++.

        Returns
        -------
        OutputSpec
            Dict with ``disparity`` tensor spec of shape ``[B, 1, H, W]``.
        """
        return {
            "disparity": TensorSpec(),
        }
