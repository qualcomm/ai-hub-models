# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from qai_hub_models_cli.proto_helpers._common import safe_load_yaml
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.datasets.pandaset import PandaSetLidarDataset
from qai_hub_models.datasets.semantic_kitti import SemanticKittiDataset
from qai_hub_models.models.rangenet_plus_plus.external_repos.lidar_bonnetal.train.tasks.semantic.modules.segmentator import (
    Segmentator,
)
from qai_hub_models.models.templates.semantic_kitti.semantic_kitti_evaluator import (
    SemanticKittiEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

NUM_CLASSES = 20
INPUT_HEIGHT = 64
INPUT_WIDTH = 2048
INPUT_CHANNELS = 5

SAMPLE_POINT_CLOUD_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "000000.bin"
)

OUTPUT_MASK_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "rangenet_mask.npy"
)

# Source: https://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz
DARKNET53_MODEL_ASSET = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "darknet53.tar.gz"
)


class RangeNetPlusPlus(BaseModel):
    """RangeNet++ LiDAR semantic segmentation model."""

    def __init__(self, model: Segmentator) -> None:
        super().__init__()
        self.model: Segmentator = model
        self.learning_map: dict[int, int] = {}
        self.learning_ignore: dict[int, bool] = {}
        self.knn_params: dict = {}

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path | None = None,
    ) -> Self:
        """
        Load pretrained RangeNet++.

        Parameters
        ----------
        model_dir
            Path to the darknet53 model folder. Downloaded automatically if None.

        Returns
        -------
        Self
            Loaded model instance.
        """
        if model_dir is None:
            model_dir = DARKNET53_MODEL_ASSET.fetch(extract=True)

        model_dir = Path(model_dir)
        with open(model_dir / "arch_cfg.yaml") as f:
            arch_cfg = safe_load_yaml(f)
        with open(model_dir / "data_cfg.yaml") as f:
            data_cfg = safe_load_yaml(f)

        model = Segmentator(arch_cfg, NUM_CLASSES, path=str(model_dir))
        model.cpu()
        model.eval()
        instance = cls(model)
        instance.learning_map = {
            int(k): int(v) for k, v in data_cfg["learning_map"].items()
        }
        instance.learning_ignore = {
            int(k): bool(v) for k, v in data_cfg["learning_ignore"].items()
        }
        instance.knn_params = arch_cfg["post"]["KNN"]["params"]
        return instance

    def forward(self, range_image: torch.Tensor) -> torch.Tensor:
        """
        Predict per-class logits for a LiDAR range image.

        Parameters
        ----------
        range_image
            float32 tensor of shape [1, 5, H, W] with channels ordered
            [depth, x, y, z, intensity]. Values must be pre-normalised
            per-channel using the SemanticKITTI mean/std statistics
            (see ``project_points_to_range_image`` in app.py). The
            expected input range after normalisation is approximately
            [-3, 3] for each channel.

        Returns
        -------
        torch.Tensor
            float32 logits of shape [1, NUM_CLASSES, H, W].
            Apply argmax over dim=1 to get the predicted class index per pixel.
        """
        # Bypass Segmentator.forward() which applies F.softmax before returning.
        # Raw logits are required so that HTP does not fail on a quantized Softmax
        # at the output boundary. argmax(softmax(x)) == argmax(x) so accuracy is unchanged.
        y, skips = self.model.backbone(range_image)
        y = self.model.decoder(y, skips)
        return self.model.head(y)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = INPUT_HEIGHT,
        width: int = INPUT_WIDTH,
    ) -> InputSpec:
        return {
            "range_image": TensorSpec(
                shape=(batch_size, INPUT_CHANNELS, height, width),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "logits": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        from qai_hub_models.models.rangenet_plus_plus.app import (
            project_points_to_range_image,
        )

        points = np.fromfile(
            str(SAMPLE_POINT_CLOUD_ADDRESS.fetch()), dtype=np.float32
        ).reshape(-1, 4)
        arr, _, _ = project_points_to_range_image(points)
        return {"range_image": [arr]}

    def get_evaluator(self) -> BaseEvaluator:
        if not self.learning_map:
            raise RuntimeError(
                "learning_map is empty — use RangeNetPlusPlus.from_pretrained() "
                "to create an instance with evaluation support."
            )
        return SemanticKittiEvaluator(
            NUM_CLASSES,
            self.learning_map,
            self.learning_ignore,
            knn_params=self.knn_params,
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [SemanticKittiDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return PandaSetLidarDataset
