# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
from qai_hub_models_cli.proto_helpers._common import safe_load_yaml
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.pandaset import PandaSetLidarDataset
from qai_hub_models.datasets.semantic_kitti import SemanticKittiDataset
from qai_hub_models.models.salsanext.external_repos.salsanext.train.common.laserscan import (
    SemLaserScan,
)
from qai_hub_models.models.salsanext.external_repos.salsanext.train.tasks.semantic.modules.SalsaNext import (
    SalsaNext as SalsaNextModel,
)
from qai_hub_models.models.templates.semantic_kitti.semantic_kitti_evaluator import (
    SemanticKittiEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 3
DEFAULT_WEIGHTS = "pretrained/SalsaNext.pt"
INPUT_LIDAR_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "000000.bin"
).fetch()
# Load configuration files

ARCH_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pretrained/arch_cfg.yaml"
).fetch()

DATA_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pretrained/data_cfg.yaml"
).fetch()


class SalsaNext(BaseModel):
    """Exportable Salsanext segmentation end-to-end."""

    @classmethod
    def from_pretrained(cls, weights_path: str | None = None) -> Self:
        """Load salsanext from a weightfile created by the source salsanext repository."""
        # Load PyTorch model from disk
        salsanext_model = _load_salsanext_source_model_from_weights(weights_path)
        return cls(salsanext_model)

    def forward(self, lidar: torch.Tensor) -> tuple[torch.Tensor]:
        return self.model(lidar)

    def load_lidar_bin(self, lidar_bin_path: str) -> tuple[torch.Tensor, Any]:
        with open(ARCH_ADDRESS) as f:
            arch = safe_load_yaml(f)
        with open(DATA_ADDRESS) as f:
            data = safe_load_yaml(f)
        color_map = data["color_map"]
        sensor = arch["dataset"]["sensor"]
        img_H = sensor["img_prop"]["height"]
        img_W = sensor["img_prop"]["width"]
        img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        fov_up = sensor["fov_up"]
        fov_down = sensor["fov_down"]

        scan = SemLaserScan(
            color_map, project=True, H=img_H, W=img_W, fov_up=fov_up, fov_down=fov_down
        )
        scan.open_scan(lidar_bin_path)

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)

        proj = torch.cat(
            [
                proj_range.unsqueeze(0),
                proj_xyz.permute(2, 0, 1),
                proj_remission.unsqueeze(0),
            ]
        )
        proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
        proj *= proj_mask.float()
        return proj.unsqueeze(0), scan

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        lidar_input, _ = self.load_lidar_bin(str(INPUT_LIDAR_ADDRESS))
        return {"lidar": [lidar_input.numpy()]}

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 64,
        width: int = 2048,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
        # the model input has fixed channels i.e 5
        channel = 5
        return {
            "lidar": TensorSpec(
                shape=(batch_size, channel, height, width),
                dtype="float32",
                io_type=IoType.TENSOR,
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "predict": TensorSpec(),
        }

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [SemanticKittiDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return PandaSetLidarDataset

    def get_evaluator(self) -> BaseEvaluator:
        with open(DATA_ADDRESS) as f:
            data = safe_load_yaml(f)
        n_classes = len(data["learning_map_inv"])
        return SemanticKittiEvaluator(
            n_classes, data["learning_map"], data["learning_ignore"]
        )


def _load_salsanext_source_model_from_weights(
    weights_path_salsanext: str | None = None,
) -> torch.nn.Module:
    # Load SalsaNext model from the source repository using the given weights.
    # download the weights file
    if not weights_path_salsanext:
        weights_path_salsanext = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        ).fetch()

    model: torch.nn.Module = SalsaNextModel(20)
    model = torch.nn.DataParallel(model)
    # load pretrained model
    checkpoint = torch.load(
        str(weights_path_salsanext), map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to("cpu").eval()
    return model
