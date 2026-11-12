# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.kitti_lidar_place_recognition import (
    KittiLidarPlaceRecognitionDataset,
)
from qai_hub_models.datasets.nclt_lidar_place_recognition import (
    NCLTLidarPlaceRecognitionDataset,
)
from qai_hub_models.datasets.nclt_lidar_place_recognition.nclt_lidar_place_recognition import (
    nclt_scan_to_range_image,
    read_velodyne_sync_bin,
)
from qai_hub_models.models.overlap_transformer.evaluator import (
    LidarPlaceRecognitionEvaluator,
)
from qai_hub_models.models.overlap_transformer.external_repos.overlap_transformer.modules.overlap_transformer import (
    featureExtracter,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

INPUT_HEIGHT = 64
INPUT_WIDTH = 900

# Upstream's only checkpoint distro is a Google Drive file (no HF/GitHub
# mirror); mirrored in private S3 since Drive throttles automated downloads.
DEFAULT_WEIGHTS_GDRIVE = CachedWebModelAsset.from_google_drive(
    "1FNrx9pcDa9NF7z8CFtuTWyauNkeSEFW4",
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "pretrained_overlap_transformer.pth.tar",
    private_s3_key=f"qai-hub-models/models/{MODEL_ID}/v{MODEL_ASSET_VERSION}/pretrained_overlap_transformer.pth.tar",
)

# Same NCLT scan used by the demo (see demo.py's QUERY_SCAN); reused here so
# scorecard inference/accuracy checks run on a real range image, not random data.
SAMPLE_INPUT_SCAN = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "query.bin"
)


class OverlapTransformer(BaseModel):
    """LiDAR place recognition: range image in, 256-d retrieval descriptor out."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model=model)

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None) -> Self:
        """Load OverlapTransformer, optionally from a local checkpoint file.

        Defaults to upstream's pretrained KITTI checkpoint
        (``pretrained_overlap_transformer.pth.tar``), auto-fetched from its
        Google Drive distribution. Pass a local path as `checkpoint` to use
        a different checkpoint instead.
        """
        net = featureExtracter(
            height=INPUT_HEIGHT, width=INPUT_WIDTH, channels=1, use_transformer=True
        )
        checkpoint = checkpoint or str(DEFAULT_WEIGHTS_GDRIVE.fetch())
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state["state_dict"])
        return cls(net)

    def forward(self, range_image: torch.Tensor) -> torch.Tensor:
        """
        Run OverlapTransformer and return a LiDAR place-recognition descriptor.

        Parameters
        ----------
        range_image
            Single-channel spherical range image, shape [N, 1, 64, 900],
            raw range values in meters (0-80), unnormalized.

        Returns
        -------
        descriptor : torch.Tensor
            L2-normalized global descriptor, shape [N, 256].
        """
        return self.model(range_image)

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "range_image": TensorSpec(
                shape=(batch_size, 1, INPUT_HEIGHT, INPUT_WIDTH), dtype="float32"
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"descriptor": TensorSpec()}

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        if input_spec is None:
            input_spec = self.get_input_spec()
        _, _, height, width = input_spec["range_image"][0]
        scan = read_velodyne_sync_bin(SAMPLE_INPUT_SCAN.fetch())
        range_image = nclt_scan_to_range_image(scan, height, width)
        tensor = torch.from_numpy(range_image).unsqueeze(0).unsqueeze(0)
        return {"range_image": [tensor.numpy()]}

    def get_evaluator(self) -> LidarPlaceRecognitionEvaluator:
        return LidarPlaceRecognitionEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [KittiLidarPlaceRecognitionDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return NCLTLidarPlaceRecognitionDataset
