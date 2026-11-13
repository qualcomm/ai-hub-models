# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

KITTI_LPR_FOLDER_NAME = "kitti_lidar_place_recognition"
KITTI_LPR_VERSION = 1
SEQUENCE = "00"

KITTI_LPR_INSTALLATION_STEPS = [
    "Open http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip, provide your email, and click the request download link button",
    "Download data_odometry_velodyne.zip from the link sent to your email",
    "Open http://www.cvlibs.net/download.php?file=data_odometry_poses.zip, provide your email, and click the request download link button",
    "Download data_odometry_poses.zip from the link sent to your email",
    "Note: only sequences/00 from the extracted archive is used by this dataset.",
]
KITTI_LPR_CONFIGURE_FILES = [
    "data_odometry_velodyne.zip",
    "data_odometry_poses.zip",
]

# Staged in the private S3 bucket (internal use only); external users must
# `configure()` from their own cvlibs.net download instead.
KITTI_LPR_VELODYNE_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{KITTI_LPR_FOLDER_NAME}/v{KITTI_LPR_VERSION}/partial_data_odometry_velodyne_seq00.zip",
    KITTI_LPR_FOLDER_NAME,
    KITTI_LPR_VERSION,
    "data_odometry_velodyne.zip",
    installation_steps=KITTI_LPR_INSTALLATION_STEPS,
    configure_files=KITTI_LPR_CONFIGURE_FILES,
)
KITTI_LPR_POSES_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{KITTI_LPR_FOLDER_NAME}/v{KITTI_LPR_VERSION}/data_odometry_poses.zip",
    KITTI_LPR_FOLDER_NAME,
    KITTI_LPR_VERSION,
    "data_odometry_poses.zip",
    installation_steps=KITTI_LPR_INSTALLATION_STEPS,
    configure_files=KITTI_LPR_CONFIGURE_FILES,
)


def lidar_scan_to_range_image(
    points: np.ndarray,
    height: int,
    width: int,
    fov_up_deg: float = 3.0,
    fov_down_deg: float = -25.0,
    max_range: float = 80.0,
) -> np.ndarray:
    """Project a LiDAR point cloud (N, 3+) into a single-channel spherical range image.

    Standard spherical range-image projection used across LiDAR place-recognition
    and segmentation literature (OverlapNet, RangeNet++): each point is binned by
    (yaw, pitch) into a (height, width) grid, and each cell holds the Euclidean
    range of the nearest point that lands there. max_range=80m matches upstream
    OverlapTransformer's own tools/utils/utils.py::range_projection default (and
    this repo's segmentation-task models); verified by descriptor parity against
    upstream's actual train/eval path (range_projection(max_range=80) -> PNG ->
    cv2.IMREAD_GRAYSCALE) on the 3 demo scans -- cos_sim 0.9995-1.0 at max_range=80
    vs 0.42-0.64 at the previous (incorrect) max_range=50.
    """
    fov_up = fov_up_deg / 180.0 * np.pi
    fov_down = fov_down_deg / 180.0 * np.pi
    fov = abs(fov_up) + abs(fov_down)

    xyz = points[:, :3]
    depth = np.linalg.norm(xyz, axis=1)
    valid = (depth > 1e-6) & (depth < max_range)
    xyz, depth = xyz[valid], depth[valid]

    yaw = -np.arctan2(xyz[:, 1], xyz[:, 0])
    pitch = np.arcsin(np.clip(xyz[:, 2] / depth, -1.0, 1.0))

    proj_x = np.clip(np.floor(0.5 * (yaw / np.pi + 1.0) * width), 0, width - 1).astype(
        np.int64
    )
    proj_y = np.clip(
        np.floor((1.0 - (pitch + abs(fov_down)) / fov) * height), 0, height - 1
    ).astype(np.int64)

    # Nearest point wins when multiple points land in the same cell.
    order = np.argsort(-depth)
    range_image = np.zeros((height, width), dtype=np.float32)
    range_image[proj_y[order], proj_x[order]] = depth[order]
    return range_image


class LidarPlaceRecognitionDataset(BaseDataset):
    """Shared base for LiDAR place-recognition retrieval datasets.

    Subclasses populate self.scan_paths, self.positions, self.frame_ids, self.height
    and self.width in __init__, and implement _load_range_image() to turn a scan path
    into a single-channel range image. Keeping __getitem__ here (rather than
    duplicated per-dataset) ensures the evaluator's frame-gap logic always sees the
    same (range_image, (position, frame_index)) shape.
    """

    scan_paths: list[Path]
    positions: np.ndarray
    frame_ids: np.ndarray
    height: int
    width: int

    def __len__(self) -> int:
        return len(self.scan_paths)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        range_image = self._load_range_image(self.scan_paths[index])
        range_image_tensor = torch.from_numpy(range_image).unsqueeze(0)
        position = torch.from_numpy(self.positions[index]).float()
        frame_index = torch.tensor(self.frame_ids[index], dtype=torch.long)
        return range_image_tensor, (position, frame_index)

    @abstractmethod
    def _load_range_image(self, scan_path: Path) -> np.ndarray: ...

    @staticmethod
    def default_samples_per_job() -> int:
        # True revisits are sparse even in the dense VAL split, so 100
        # evenly-spaced samples yields too few valid queries for a stable Recall@1.
        return 1000


class KittiLidarPlaceRecognitionDataset(LidarPlaceRecognitionDataset):
    """KITTI odometry sequence 00, framed as a LiDAR place-recognition retrieval task.

    KITTI odometry's license is non-commercial, so this dataset is eval-only (VAL) --
    quantization calibration uses NCLTLidarPlaceRecognitionDataset instead.
    """

    def __init__(
        self,
        input_velodyne_zip: str | None = None,
        input_poses_zip: str | None = None,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        if split != DatasetSplit.VAL:
            raise ValueError(
                f"{type(self).__name__} is eval-only (non-commercial license) and "
                "only supports DatasetSplit.VAL."
            )
        self.input_velodyne_zip = input_velodyne_zip
        self.input_poses_zip = input_poses_zip
        self.sequence_dir = (
            KITTI_LPR_VELODYNE_ASSET.extracted_path / "sequences" / SEQUENCE
        )
        self.poses_path = (
            KITTI_LPR_POSES_ASSET.extracted_path / "poses" / f"{SEQUENCE}.txt"
        )
        BaseDataset.__init__(
            self, KITTI_LPR_VELODYNE_ASSET.extracted_path.parent, split, input_spec
        )

        # Matches OverlapTransformer.get_input_spec()'s defaults; duplicated here
        # since importing from model.py would be a circular import.
        input_spec = input_spec or {"range_image": TensorSpec(shape=(1, 1, 64, 900))}
        _, _, self.height, self.width = input_spec["range_image"][0]

        self.scan_paths = sorted((self.sequence_dir / "velodyne").glob("*.bin"))
        self.positions = self._load_positions(self.poses_path)
        self.frame_ids = np.arange(len(self.scan_paths))

    @staticmethod
    def _load_positions(poses_path: Path) -> np.ndarray:
        # Column 3 of each flattened 3x4 camera-frame pose matrix is the translation;
        # pairwise distance is preserved under the fixed lidar<->camera transform.
        poses = np.loadtxt(poses_path).reshape(-1, 3, 4)
        return poses[:, :, 3]

    def _load_range_image(self, scan_path: Path) -> np.ndarray:
        scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
        return lidar_scan_to_range_image(scan, self.height, self.width)

    def _validate_data(self) -> bool:
        velodyne_dir = self.sequence_dir / "velodyne"
        if not (velodyne_dir.is_dir() and self.poses_path.is_file()):
            return False
        scan_paths = sorted(velodyne_dir.glob("*.bin"))
        if not scan_paths:
            return False
        try:
            num_poses = np.loadtxt(self.poses_path).reshape(-1, 3, 4).shape[0]
        except ValueError:
            # Malformed/truncated poses file from a partial extraction.
            return False
        return len(scan_paths) == num_poses

    def _download_data(self) -> None:
        KITTI_LPR_VELODYNE_ASSET.fetch(extract=True, local_path=self.input_velodyne_zip)
        KITTI_LPR_POSES_ASSET.fetch(extract=True, local_path=self.input_poses_zip)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 2:
            raise ValueError(
                f"{cls.__name__}.configure expects 2 file(s), got {len(files)}."
            )
        cls(input_velodyne_zip=str(files[0]), input_poses_zip=str(files[1]))

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.cvlibs.net/datasets/kitti/eval_odometry.php",
            split_description=(
                "all of sequence 00 -- every scan serves as both query and "
                "database entry, per the standard LiDAR place-recognition protocol"
            ),
        )
