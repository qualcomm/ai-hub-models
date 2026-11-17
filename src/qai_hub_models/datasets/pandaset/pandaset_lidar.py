# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from qai_hub_models.datasets.pandaset.pandaset import PANDASET_ID, PandaSetDatasetBase
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import (
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec

PANDASET_VERSION = 1

PANDASET_ASSET = CachedWebDatasetAsset.from_asset_store(
    PANDASET_ID,
    PANDASET_VERSION,
    "pandaset-dataset.zip",
)

# Pandar64 sensor: 64-beam 360° mechanical spinning LiDAR
PANDAR64_FOV_UP = 15.0  # degrees
PANDAR64_FOV_DOWN = -25.0  # degrees
PANDAR64_H = 64
PANDAR64_W = 2048

# Per-channel normalisation stats [depth, x, y, z, intensity]
# Computed from PandaSet Pandar64 sensor-frame point clouds.
# Channel order: [depth, x, y, z, intensity]
SENSOR_IMG_MEANS = np.array([27.70, -4.01, 2.17, 1.17, 15.67], dtype=np.float32)
SENSOR_IMG_STDS = np.array([23.98, 22.63, 28.37, 1.79, 29.55], dtype=np.float32)


def _world_to_sensor(points_world: np.ndarray, pose: dict) -> np.ndarray:
    """Transform (N,3) points from world coordinates to sensor frame."""
    tx = pose["position"]["x"]
    ty = pose["position"]["y"]
    tz = pose["position"]["z"]
    qw = pose["heading"]["w"]
    qx = pose["heading"]["x"]
    qy = pose["heading"]["y"]
    qz = pose["heading"]["z"]

    R = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )

    t = np.array([tx, ty, tz], dtype=np.float64)
    return (R.T @ (points_world.astype(np.float64) - t).T).T.astype(np.float32)


def _project_to_range_image(
    points: np.ndarray,
    intensity: np.ndarray,
    H: int = PANDAR64_H,
    W: int = PANDAR64_W,
    fov_up: float = PANDAR64_FOV_UP,
    fov_down: float = PANDAR64_FOV_DOWN,
) -> np.ndarray:
    """Project sensor-frame point cloud to a normalised (1,5,H,W) range image."""
    fov_up_r = fov_up / 180.0 * np.pi
    fov_down_r = fov_down / 180.0 * np.pi
    fov = abs(fov_down_r) + abs(fov_up_r)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    depth = np.sqrt(x**2 + y**2 + z**2)
    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / np.clip(depth, 1e-5, None))

    u = (0.5 * (yaw / np.pi + 1.0) * W).astype(int).clip(0, W - 1)
    v = ((1.0 - (pitch + abs(fov_down_r)) / fov) * H).astype(int).clip(0, H - 1)

    img = np.zeros((5, H, W), dtype=np.float32)
    img[0, v, u] = depth
    img[1, v, u] = x
    img[2, v, u] = y
    img[3, v, u] = z
    img[4, v, u] = intensity
    img_norm: np.ndarray = (
        (img - SENSOR_IMG_MEANS[:, None, None]) / SENSOR_IMG_STDS[:, None, None]
    ).astype(np.float32)
    return img_norm[np.newaxis]  # (1, 5, H, W)


class PandaSetLidarDataset(PandaSetDatasetBase):
    """Calibration dataset using a mini subset of PandaSet Pandar64 LiDAR scans.

    This is not the full PandaSet dataset — it contains 14 sequences x 50 frames
    = 700 frames total, selected for model calibration purposes.

    License: CC0 Public Domain.
    Source: https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset

    Returns projected range images of shape [5, H, W] — same format and
    normalisation as SemanticKittiDataset, suitable for calibrating RangeNet++.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(PANDASET_ASSET, split=split)

    def _build_frame_list(self) -> None:
        self._frames: list[tuple[Path, dict]] = self._build_lidar_frame_list()

    def _build_lidar_frame_list(self) -> list[tuple[Path, dict]]:
        """Collect all (pkl_path, pose_dict) pairs across all sequences."""
        frames: list[tuple[Path, dict]] = []
        if not self.data_root.exists():
            return frames
        for seq_dir in sorted(self.data_root.iterdir()):
            lidar_dir = seq_dir / "lidar"
            poses_file = lidar_dir / "poses.json"
            if not lidar_dir.exists() or not poses_file.exists():
                continue
            with open(poses_file) as f:
                poses = json.load(f)
            for frame_idx, pkl_file in enumerate(sorted(lidar_dir.glob("*.pkl"))):
                if frame_idx < len(poses):
                    frames.append((pkl_file, poses[frame_idx]))
        return frames

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a projected range image for calibration.

        Parameters
        ----------
        index
            Sample index.

        Returns
        -------
        range_image : torch.Tensor
            float32 tensor of shape [5, H, W] — normalised range image with
            channels [depth, x, y, z, intensity].
        label : int
            Dummy label (always 0 — calibration does not require ground truth).
        """
        pkl_path, pose = self._frames[index]

        with open(pkl_path, "rb") as f:
            df = pickle.load(f)  # nosec: file extracted from hash-verified PANDASET_ASSET

        # Use only Pandar64 (sensor_id == 0); exclude PandarGT (sensor_id == 1)
        df = df[df["d"] == 0]

        points_world = df[["x", "y", "z"]].values.astype(np.float32)
        intensity = df["i"].values.astype(np.float32)

        points_sensor = _world_to_sensor(points_world, pose)
        range_image = _project_to_range_image(points_sensor, intensity)

        return torch.from_numpy(range_image[0]), 0  # [5, H, W]

    @classmethod
    def dataset_name(cls) -> str:
        return "pandaset"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.kaggle.com/datasets/usharengaraju/pandaset-dataset",
            split_description="Mini subset: 14 sequences x 50 frames (700 total), Pandar64 LiDAR only (CC0 Public Domain)",
        )
