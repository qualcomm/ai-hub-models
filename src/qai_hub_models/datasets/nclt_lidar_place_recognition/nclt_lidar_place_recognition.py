# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np

from qai_hub_models.datasets.kitti_lidar_place_recognition.kitti_lidar_place_recognition import (
    LidarPlaceRecognitionDataset,
    lidar_scan_to_range_image,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

# NCLT (https://robots.engin.umich.edu/nclt/) is distributed under the Open
# Database License (ODbL); this derivative contains information from it under
# that license. Cite Carlevaris-Bianco et al., "University of Michigan North
# Campus Long-Term Vision and LIDAR Dataset", IJRR 2016.
NCLT_LPR_FOLDER_NAME = "nclt_lidar_place_recognition"
NCLT_LPR_VERSION = 1
NCLT_LPR_SESSION = "2013-01-10"

# Datasheet fov_up=10.67 wastes ~1/6 of the row grid on empty sky (measured
# returns are <5deg for 99% of points this session); 5.0 keeps a small margin.
NCLT_LPR_FOV_UP_DEG = 5.0
NCLT_LPR_FOV_DOWN_DEG = -30.67

# Groundtruth is a continuous ~100Hz trace, so an in-range scan's nearest sample
# is normally within a few ms; anything farther means the scan falls outside the
# groundtruth trace and _nearest_gt_indices() just clamped to the endpoint.
NCLT_LPR_POSITION_TOLERANCE_US = 50_000  # 50ms

# Upstream source (umich's own S3 bucket), e.g.:
#   https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/2013-01-10_vel.tar.gz
#   https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_2013-01-10.csv
# Mirrored in our private S3 below so CI/runtime fetches don't depend on it,
# with this URL kept as the fallback for external users without Qualcomm AWS credentials.
NCLT_LPR_VELODYNE_URL = f"https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/velodyne_data/{NCLT_LPR_SESSION}_vel.tar.gz"
NCLT_LPR_GROUNDTRUTH_URL = f"https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_{NCLT_LPR_SESSION}.csv"

NCLT_LPR_VELODYNE_ASSET = CachedWebDatasetAsset(
    NCLT_LPR_VELODYNE_URL,
    NCLT_LPR_FOLDER_NAME,
    NCLT_LPR_VERSION,
    "vel.tar.gz",
    private_s3_key=f"qai-hub-models/datasets/{NCLT_LPR_FOLDER_NAME}/v{NCLT_LPR_VERSION}/{NCLT_LPR_SESSION}_vel.tar.gz",
)
NCLT_LPR_GROUNDTRUTH_ASSET = CachedWebDatasetAsset(
    NCLT_LPR_GROUNDTRUTH_URL,
    NCLT_LPR_FOLDER_NAME,
    NCLT_LPR_VERSION,
    "groundtruth.csv",
    private_s3_key=f"qai-hub-models/datasets/{NCLT_LPR_FOLDER_NAME}/v{NCLT_LPR_VERSION}/groundtruth_{NCLT_LPR_SESSION}.csv",
)

# Per-scan record layout from upstream's read_vel_sync.py: x,y,z as
# little-endian uint16, then intensity and laser id as uint8 each.
_VELODYNE_SYNC_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("z", "<u2"), ("i", "<u1"), ("l", "<u1")]
)
_VELODYNE_SYNC_SCALE = 0.005
_VELODYNE_SYNC_OFFSET = -100.0


def read_velodyne_sync_bin(path: str | Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=_VELODYNE_SYNC_DTYPE)
    xyz = (
        np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float32)
        * _VELODYNE_SYNC_SCALE
        + _VELODYNE_SYNC_OFFSET
    )
    intensity = raw["i"].astype(np.float32).reshape(-1, 1)
    laser_id = raw["l"].astype(np.float32).reshape(-1, 1)
    return np.concatenate([xyz, intensity, laser_id], axis=1)


def _load_groundtruth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0], data[:, 1:4]


def _nearest_gt_indices(query_utimes: np.ndarray, gt_utimes: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(gt_utimes, query_utimes), 1, len(gt_utimes) - 1)
    left, right = idx - 1, idx
    pick_left = np.abs(query_utimes - gt_utimes[left]) <= np.abs(
        query_utimes - gt_utimes[right]
    )
    return np.where(pick_left, left, right)


def nearest_positions(
    query_utimes: np.ndarray, gt_utimes: np.ndarray, gt_xyz: np.ndarray
) -> np.ndarray:
    return gt_xyz[_nearest_gt_indices(query_utimes, gt_utimes)]


def fill_empty_rows(range_image: np.ndarray) -> np.ndarray:
    """Nearest-neighbor-fill elevation rows with no projected points.

    Even with NCLT_LPR_FOV_UP_DEG tuned to the scene's actual return range,
    NCLT's 32 physical beams don't densely cover every one of this model's 64
    elevation rows the way KITTI's 64-beam scans do. Replicating each empty
    row from its nearest real beam row approximates that density instead of
    leaving systematic gaps.
    """
    valid_rows = np.flatnonzero(np.any(range_image != 0, axis=1))
    if len(valid_rows) in (0, range_image.shape[0]):
        return range_image

    all_rows = np.arange(range_image.shape[0])
    idx = np.clip(np.searchsorted(valid_rows, all_rows), 1, len(valid_rows) - 1)
    left, right = idx - 1, idx
    pick_left = np.abs(all_rows - valid_rows[left]) <= np.abs(
        all_rows - valid_rows[right]
    )
    nearest_rows = np.where(pick_left, valid_rows[left], valid_rows[right])
    return range_image[nearest_rows]


def nclt_scan_to_range_image(scan: np.ndarray, height: int, width: int) -> np.ndarray:
    """Project an NCLT point cloud and fill its empty rows -- the one path both the
    dataset (eval/calibration) and OverlapTransformerApp (demo/test) must share so
    they see identical range images.
    """
    range_image = lidar_scan_to_range_image(
        scan,
        height,
        width,
        fov_up_deg=NCLT_LPR_FOV_UP_DEG,
        fov_down_deg=NCLT_LPR_FOV_DOWN_DEG,
    )
    return fill_empty_rows(range_image)


class NCLTLidarPlaceRecognitionDataset(LidarPlaceRecognitionDataset):
    """NCLT session 2013-01-10, framed as a LiDAR place-recognition retrieval task.

    The source data has no held-out split, so every CALIBRATION_FRAME_STRIDE-th frame
    is carved out as TRAIN (calibration only) and the disjoint remainder is VAL. Each
    VAL sample's frame index is the original sequence position, not the local index,
    so the evaluator's time-gap revisit logic stays correct.
    """

    # Reserve 1 in every N frames for calibration (TRAIN), leaving the dense
    # remainder in VAL so loop-closure revisits stay detectable there.
    CALIBRATION_FRAME_STRIDE = 5

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        if split not in (DatasetSplit.TRAIN, DatasetSplit.VAL):
            raise ValueError(
                f"{type(self).__name__} does not support DatasetSplit.TEST"
            )
        self.sequence_dir = NCLT_LPR_VELODYNE_ASSET.extracted_path
        self.groundtruth_path = NCLT_LPR_GROUNDTRUTH_ASSET.local_cache_path
        BaseDataset.__init__(self, self.sequence_dir.parent, split, input_spec)

        # Matches OverlapTransformer.get_input_spec()'s defaults; duplicated here
        # since importing from model.py would be a circular import.
        input_spec = input_spec or {"range_image": TensorSpec(shape=(1, 1, 64, 900))}
        _, _, self.height, self.width = input_spec["range_image"][0]

        all_scan_paths = sorted((self.sequence_dir / "velodyne_sync").glob("*.bin"))
        all_utimes = np.array([int(p.stem) for p in all_scan_paths], dtype=np.float64)
        gt_utimes, gt_xyz = _load_groundtruth(self.groundtruth_path)
        nearest_gt_idx = _nearest_gt_indices(all_utimes, gt_utimes)
        all_positions = gt_xyz[nearest_gt_idx]

        in_range = (
            np.abs(all_utimes - gt_utimes[nearest_gt_idx])
            <= NCLT_LPR_POSITION_TOLERANCE_US
        )
        num_out_of_range = int((~in_range).sum())
        if num_out_of_range:
            print(
                f"Dropping {num_out_of_range} scan(s) whose timestamp falls outside "
                "the groundtruth trace (would otherwise be mislabeled with the "
                "nearest endpoint position)."
            )

        is_train_frame = (
            np.arange(len(all_scan_paths)) % self.CALIBRATION_FRAME_STRIDE == 0
        )
        frame_mask = is_train_frame if split == DatasetSplit.TRAIN else ~is_train_frame
        frame_mask &= in_range
        self.frame_ids = np.nonzero(frame_mask)[0]
        self.scan_paths = [all_scan_paths[i] for i in self.frame_ids]
        self.positions = all_positions[self.frame_ids]

    def _load_range_image(self, scan_path: Path) -> np.ndarray:
        scan = read_velodyne_sync_bin(scan_path)
        return nclt_scan_to_range_image(scan, self.height, self.width)

    def _validate_data(self) -> bool:
        velodyne_dir = self.sequence_dir / "velodyne_sync"
        if not (velodyne_dir.is_dir() and self.groundtruth_path.is_file()):
            return False
        scan_paths = sorted(velodyne_dir.glob("*.bin"))
        if not scan_paths:
            return False
        try:
            gt_utimes, _ = _load_groundtruth(self.groundtruth_path)
        except ValueError:
            # Malformed/truncated groundtruth file from a partial extraction.
            return False
        if len(gt_utimes) == 0:
            return False
        # groundtruth is a continuous ~100Hz trace (no 1-row-per-scan), so check
        # it covers the scans' time range instead of a count match.
        scan_utimes = np.array([int(p.stem) for p in scan_paths], dtype=np.float64)
        # Scans can extend slightly beyond the groundtruth trace; nearest_positions()
        # clamps these to the nearest endpoint, so allow a small tolerance here.
        tolerance_us = 5_000_000  # 5s
        return bool(
            gt_utimes.min() - tolerance_us <= scan_utimes.min()
            and gt_utimes.max() + tolerance_us >= scan_utimes.max()
        )

    def _download_data(self) -> None:
        NCLT_LPR_VELODYNE_ASSET.fetch(extract=True)
        NCLT_LPR_GROUNDTRUTH_ASSET.fetch()

    @staticmethod
    def default_num_calibration_samples() -> int:
        # TRAIN is every 5th of the session's 5120 scans (1024 total); the base
        # class's default of 100 would silently drop most of it.
        return 1024

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://robots.engin.umich.edu/nclt/",
            split_description=(
                f"session {NCLT_LPR_SESSION} minus every "
                f"{NCLTLidarPlaceRecognitionDataset.CALIBRATION_FRAME_STRIDE}th "
                "frame (held out for calibration) -- every remaining scan serves "
                "as both query and database entry, per the standard LiDAR "
                "place-recognition protocol"
            ),
        )
