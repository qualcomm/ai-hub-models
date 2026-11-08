# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import pickle as cPickle
import shutil
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.models.agpose.model_patches import ag_pose_repo_paths
from qai_hub_models.utils.depth_completion import fill_missing
from qai_hub_models.utils.image_processing import numpy_image_to_torch
from qai_hub_models.utils.image_processing_3d import depth_to_full_pointcloud
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

FPS_NPOINTS = [512, 256, 128, 64]


DATASET_ID = "nocs_real275"
DATASET_ASSET_VERSION = 1

NOCS_REAL275_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/nocs_real275/v{DATASET_ASSET_VERSION}/nocs_real275_v1.tar.gz",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "nocs_real275_v1.tar.gz",
    installation_steps=[
        "Visit https://github.com/hughw19/NOCS_CVPR2019 and follow the dataset download instructions.",
        "Download the REAL275 dataset (real/test split) from the NOCS project page.",
        "Download segmentation_results/REAL275/*.pkl from https://github.com/Leeiieeo/AG-Pose (data preparation section).",
        "Run `qai-hub-models configure-dataset --class qai_hub_models.models.agpose.dataset.NocsReal275Dataset --files /path/to/nocs_real275_v1.tar.gz`",
    ],
    configure_files=["nocs_real275_v1.tar.gz"],
)

NUM_POINTS: int = 1024

# RGB crop size expected by the DINOv2-S (ViT-S/14) backbone
IMAGE_SIZE: int = 224

# NOCS camera intrinsics for REAL dataset
# Source: https://github.com/hughw19/NOCS_CVPR2019/blob/master/dataset.py
REAL_FX: float = 591.0125
REAL_FY: float = 590.16775
REAL_CX: float = 322.525
REAL_CY: float = 244.11084

DEPTH_NORM_SCALE: float = 1000.0
_MIN_DEPTH_PIXELS: int = 16

CATEGORY_NAMES: list[str] = ["bottle", "bowl", "camera", "can", "laptop", "mug"]
NOCS_CATEGORY_MAP: dict[int, tuple[int, str]] = {
    i + 1: (i, name) for i, name in enumerate(CATEGORY_NAMES)
}


def compute_fps_indices(pts: torch.Tensor, npoints: list = FPS_NPOINTS) -> list:
    """
    Pre-compute FPS indices outside the model — never traced into ONNX graph.
    Each SA layer runs FPS on the downsampled output of the previous layer.

    Parameters
    ----------
    pts
        Point cloud tensor of shape (B, N, 3).
    npoints
        List of point counts for each SA layer.

    Returns
    -------
    list
        List of FPS index tensors for each SA layer.
    """
    with ag_pose_repo_paths():
        import pointnet2_utils as pn2

    fps_indices = []
    xyz = pts
    for npoint in npoints:
        idx = pn2.farthest_point_sample(xyz, npoint)
        fps_indices.append(idx)
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        xyz = pn2.gather_operation(xyz_flipped, idx).transpose(1, 2).contiguous()
    return fps_indices


def load_depth(depth_path: str) -> np.ndarray:
    """
    Load a depth image from disk and return as uint16 array.

    Handles both 16-bit single-channel and 3-channel encoded depth images.
    For 3-channel images, depth is decoded as: channel1 * 256 + channel2.
    Pixel value 32001 is treated as invalid (set to 0).

    Parameters
    ----------
    depth_path
        Path to depth image file.

    Returns
    -------
    np.ndarray
        Depth image as uint16 array of shape (H, W).
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to load depth: {depth_path}")
    if len(depth.shape) == 3:
        depth16 = depth[:, :, 1] * 256 + depth[:, :, 2]
        depth16 = np.where(depth16 == 32001, 0, depth16)
        return depth16.astype(np.uint16)
    return depth


def get_bbox(bbox: np.ndarray) -> tuple[int, int, int, int]:
    """
    Compute a square bounding box centred on the instance bbox.

    Expands the tighter axis to the nearest multiple of 40, caps at 440,
    and clamps to image boundaries (480 x 640).


    Parameters
    ----------
    bbox
        Array [y1, x1, y2, x2] from NOCS annotation.

    Returns
    -------
    tuple[int, int, int, int]
        (rmin, rmax, cmin, cmax) pixel crop coordinates.
    """
    y1, x1, y2, x2 = bbox
    img_width, img_length = 480, 640
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def has_sufficient_depth_pixels(
    inst_mask: np.ndarray,
    depth_filled: np.ndarray,
    min_pixels: int = _MIN_DEPTH_PIXELS,
) -> bool:
    """
    Check whether an instance has enough valid depth pixels for pose estimation.

    Parameters
    ----------
    inst_mask
        Boolean mask of shape (H, W) indicating instance pixels.
    depth_filled
        Depth image of shape (H, W) with holes filled.
    min_pixels
        Minimum number of valid pixels required.

    Returns
    -------
    bool
        True if the number of valid depth pixels exceeds min_pixels.
    """
    valid = inst_mask & (depth_filled > 0)
    return int(valid.sum()) > min_pixels


def preprocess_rgb(rgb_crop: np.ndarray) -> torch.Tensor:
    """
    Resize an RGB crop and convert it to a raw [0, 1] tensor.

    Parameters
    ----------
    rgb_crop
        RGB crop, shape (H, W, 3), dtype uint8.

    Returns
    -------
    torch.Tensor
        Raw [0, 1] tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE), float32.
        Normalization is folded into DINOv2 patch-embed weights at load time
        (model_patches.py) — returns raw [0, 1] matching value_range=(0,1).
    """
    resized = cv2.resize(
        rgb_crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    return numpy_image_to_torch(resized).squeeze(0)


def sample_choose_and_pts(
    pts_full: np.ndarray,
    valid_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed: int = 0,
    num_points: int = NUM_POINTS,
    image_size: int = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample point cloud and pixel-index mapping from a depth back-projection.

    Shared by NocsReal275Dataset.__getitem__ and AGPoseCollectionApp.preprocess_depth
    so that eval and demo preprocessing cannot drift.

    Parameters
    ----------
    pts_full
        Full-frame back-projected point cloud, shape (H, W, 3), float32, metres.
    valid_mask
        Full-frame boolean mask (instance & depth > 0), shape (H, W).
    bbox
        Crop coordinates (rmin, rmax, cmin, cmax).
    seed
        RNG seed — pass ``index`` from the dataset for deterministic eval,
        pass ``0`` for the demo / app path.
    num_points
        Number of points to sample (default NUM_POINTS = 1024).
    image_size
        Target image side length for choose index mapping (default IMAGE_SIZE = 224).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        pts_arr      : Sampled point cloud, shape (num_points, 3), float32.
        choose_mapped: Index mapping from pts to flattened resized image,
                       shape (num_points,), int32, in [0, image_size**2).
    """
    rmin, rmax, cmin, cmax = bbox
    crop_h = rmax - rmin
    crop_w = cmax - cmin

    crop_valid = valid_mask[rmin:rmax, cmin:cmax]
    choose_raw = crop_valid.flatten().nonzero()[0]

    rng = np.random.default_rng(seed=seed)
    if len(choose_raw) == 0:
        choose_raw = np.arange(crop_h * crop_w)

    if len(choose_raw) <= num_points:
        choose_sampled = rng.choice(choose_raw, num_points, replace=True)
    else:
        choose_sampled = rng.choice(choose_raw, num_points, replace=False)

    pts_crop = pts_full[rmin:rmax, cmin:cmax, :].reshape(-1, 3)
    pts_arr = pts_crop[choose_sampled, :]

    # get_bbox always returns a square crop (crop_h == crop_w),
    # so one ratio covers both row and column scaling.
    assert crop_h == crop_w, (
        f"Expected square crop from get_bbox but got crop_h={crop_h}, crop_w={crop_w}."
    )
    ratio = image_size / crop_w
    col_idx = choose_sampled % crop_w
    row_idx = choose_sampled // crop_w
    choose_mapped = (
        np.floor(row_idx * ratio) * image_size + np.floor(col_idx * ratio)
    ).astype(np.int32)
    choose_mapped = np.clip(choose_mapped, 0, image_size * image_size - 1)

    return pts_arr, choose_mapped


class NocsReal275Dataset(BaseDataset):
    """
    NOCS REAL275 dataset wrapper for AG-Pose evaluation.

    Each sample contains one object instance from one scene frame.
    Calibration and evaluation both use the REAL275 test split; samples are
    not held out between the two paths.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        data_dir: str | Path | None = None,
    ) -> None:
        if split != DatasetSplit.TEST:
            raise ValueError(
                f"NocsReal275Dataset only supports DatasetSplit.TEST — "
                f"REAL275 has no train or validation split. Got: {split}"
            )
        if data_dir is not None:
            asset_root = Path(data_dir).parent
            self._real_data_root = Path(data_dir)
        else:
            asset_root = NOCS_REAL275_ASSET.fetch(extract=True)
            self._real_data_root = asset_root / "real"
        self._pkl_dir = asset_root / "segmentation_results" / "REAL275"
        split_str = "test"
        dataset_path = self._real_data_root / split_str

        super().__init__(dataset_path=dataset_path, split=split)

        self._samples: list[dict[str, Any]] = []
        self._build_sample_index()

    def _validate_data(self) -> bool:
        if not self.dataset_path.exists():
            return False
        if not list(self.dataset_path.glob("scene_*")):
            return False
        return (
            self._pkl_dir.exists()
            and len(list(self._pkl_dir.glob("results_*.pkl"))) > 0
        )

    @classmethod
    def dataset_name(cls) -> str:
        return "nocs_real275"

    def _download_data(self) -> None:
        NOCS_REAL275_ASSET.fetch(extract=True)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        """Configure the dataset from a downloaded archive.

        Parameters
        ----------
        files
            List containing the path to nocs_real275_v1.tar.gz.
        """
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        shutil.copy(str(files[0]), NOCS_REAL275_ASSET.local_cache_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 50

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/hughw19/NOCS_CVPR2019",
            split_description="REAL275 test split",
        )

    def _build_sample_index(self) -> None:
        """
        Iterate PKL segmentation result files — exactly like TestDataset in nocs_dataset.py.
        Reference: nocs_dataset.py::TestDataset.__init__:
            result_pkl_list = glob.glob(.../segmentation_results/REAL275/results_*.pkl)
        """
        pkl_files = sorted(self._pkl_dir.glob("results_*.pkl"))

        # Cache fill_missing results per frame — a single scene frame can
        # contain multiple instances, and fill_missing is expensive (heavy
        # cv2 pipeline with per-column Python loops). Computing it once per
        # frame and reusing across instances avoids O(instances) redundant
        # calls while keeping the sample index identical to the uncached path.
        depth_filled_cache: dict[str, np.ndarray] = {}

        if not pkl_files:
            raise RuntimeError(f"No PKL files found at {self._pkl_dir}")

        skipped = 0
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = cPickle.load(f)

            img_path = data["image_path"].removeprefix("data/")
            parts = img_path.split("/")
            scene_name = parts[-2]
            frame_idx = int(parts[-1])

            color_path = (
                self._real_data_root
                / "test"
                / scene_name
                / f"{frame_idx:04d}_color.png"
            )
            depth_path = (
                self._real_data_root
                / "test"
                / scene_name
                / f"{frame_idx:04d}_depth.png"
            )

            if not (color_path.exists() and depth_path.exists()):
                skipped += 1
                continue

            depth_raw = load_depth(str(depth_path))
            depth_key = str(depth_path)
            if depth_key not in depth_filled_cache:
                depth_filled_cache[depth_key] = fill_missing(
                    depth_raw.astype(np.float32), DEPTH_NORM_SCALE, 1
                )
            depth_filled = depth_filled_cache[depth_key]

            pred_masks = data["pred_masks"]
            pred_cat_ids = data["pred_class_ids"]

            for inst_idx in range(len(pred_cat_ids)):
                cat_id_1idx = int(pred_cat_ids[inst_idx])
                if cat_id_1idx not in NOCS_CATEGORY_MAP:
                    continue

                inst_mask = pred_masks[:, :, inst_idx].astype(bool)
                if not inst_mask.any():
                    continue

                if not has_sufficient_depth_pixels(
                    inst_mask, depth_filled, _MIN_DEPTH_PIXELS
                ):
                    continue

                cat_id_0idx, cat_name = NOCS_CATEGORY_MAP[cat_id_1idx]

                self._samples.append(
                    {
                        "scene_name": scene_name,
                        "frame_idx": frame_idx,
                        "inst_idx": inst_idx,
                        "cat_id_0idx": cat_id_0idx,
                        "cat_id_1idx": cat_id_1idx,
                        "cat_name": cat_name,
                        "color_path": color_path,
                        "depth_path": depth_path,
                        "pkl_path": pkl_path,
                        "pred_bbox": data["pred_bboxes"][inst_idx],
                        "pred_mask": inst_mask,
                    }
                )

        if skipped > 0:
            warnings.warn(
                f"[NocsReal275Dataset] Skipped {skipped} frames with missing "
                "color or depth files. The dataset may be partially extracted.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        tuple[str, int, int, int, int],
    ]:
        """
        Return one object instance sample and its ground-truth metadata.

        Parameters
        ----------
        index
            Index into the sample list built by _build_sample_index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            8-tuple: (rgb, pts, choose, category_label, fps1, fps2, fps3, fps4).
            category_label is 0-indexed category ID (0-5).
        tuple[str, int, int, int, int]
            Ground truth: (scene_name, frame_idx, inst_idx, cat_id_0idx, cat_id_1idx).
        """
        s = self._samples[index]

        color_bgr = cv2.imread(str(s["color_path"]), cv2.IMREAD_COLOR)
        depth_raw = load_depth(str(s["depth_path"]))

        if color_bgr is None:
            raise RuntimeError(f"Failed to load color: {s['color_path']}")
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_filled = fill_missing(depth_raw.astype(np.float32), DEPTH_NORM_SCALE, 1)

        inst_mask_raw = s["pred_mask"]
        rmin, rmax, cmin, cmax = get_bbox(s["pred_bbox"])

        valid_mask = inst_mask_raw & (depth_filled > 0)

        rgb_crop = color_rgb[rmin:rmax, cmin:cmax]
        rgb_tensor = preprocess_rgb(rgb_crop)

        pts_full = depth_to_full_pointcloud(
            depth_filled,
            fx=REAL_FX,
            fy=REAL_FY,
            cx=REAL_CX,
            cy=REAL_CY,
            depth_scale=DEPTH_NORM_SCALE,
        )
        pts_arr, choose_mapped = sample_choose_and_pts(
            pts_full, valid_mask, (rmin, rmax, cmin, cmax), seed=index
        )

        pts_tensor = torch.from_numpy(pts_arr).float()
        choose_tensor = torch.from_numpy(choose_mapped).int()
        category_label = torch.tensor(s["cat_id_0idx"], dtype=torch.int32)

        fps_indices = compute_fps_indices(pts_tensor.unsqueeze(0))
        fps1 = fps_indices[0].squeeze(0)
        fps2 = fps_indices[1].squeeze(0)
        fps3 = fps_indices[2].squeeze(0)
        fps4 = fps_indices[3].squeeze(0)

        inputs = (
            category_label,
            choose_tensor,
            fps1,
            fps2,
            fps3,
            fps4,
            pts_tensor,
            rgb_tensor,
        )
        ground_truth = (
            s["scene_name"],
            s["frame_idx"],
            s["inst_idx"],
            s["cat_id_0idx"],
            s["cat_id_1idx"],
        )
        return inputs, ground_truth

    @staticmethod
    def collate_fn(batch: list) -> tuple:
        """
        Custom collate: ground_truth contains strings (scene_name)
        which cannot be stacked as tensors.
        category_label stacks to (B,); Net.py reshapes with .reshape(-1).
        """
        inputs_list, gt_list = zip(*batch, strict=False)

        cat = torch.stack([x[0] for x in inputs_list])
        choose = torch.stack([x[1] for x in inputs_list])
        fps1 = torch.stack([x[2] for x in inputs_list])
        fps2 = torch.stack([x[3] for x in inputs_list])
        fps3 = torch.stack([x[4] for x in inputs_list])
        fps4 = torch.stack([x[5] for x in inputs_list])
        pts = torch.stack([x[6] for x in inputs_list])
        rgb = torch.stack([x[7] for x in inputs_list])
        return (cat, choose, fps1, fps2, fps3, fps4, pts, rgb), list(
            zip(*gt_list, strict=False)
        )
