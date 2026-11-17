# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.pandaset.pandaset import PANDASET_ID, PandaSetDatasetBase
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    transform_to_matrix,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import (
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

PANDASET_BEV_VERSION = 2

PANDASET_BEV_ASSET = CachedWebDatasetAsset.from_asset_store(
    PANDASET_ID,
    PANDASET_BEV_VERSION,
    "pandaset-multicamera.zip",
)

CAMERA_NAMES = (
    "front_left_camera",
    "front_camera",
    "front_right_camera",
    # PandaSet side cameras are the nearest equivalents to rear-left/right views.
    "left_camera",
    "back_camera",
    "right_camera",
)

NUM_CAMERAS = len(CAMERA_NAMES)

DEFAULT_IMAGE_SPEC = TensorSpec(shape=(1, NUM_CAMERAS * 3, 256, 704))


def _pose_to_matrix(
    pose: dict[str, dict[str, float]], *, inv: bool = False, flat: bool = False
) -> np.ndarray:
    position = pose["position"]
    heading = pose["heading"]
    return transform_to_matrix(
        [position[axis] for axis in ("x", "y", "z")],
        [heading[axis] for axis in ("w", "x", "y", "z")],
        inv=inv,
        flat=flat,
    )


class PandaSetBevDataset(PandaSetDatasetBase):
    """PandaSet six-camera inputs for BEV model calibration."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        image_spec = (input_spec or {}).get("image", DEFAULT_IMAGE_SPEC)
        self.input_height = image_spec[0][-2]
        self.input_width = image_spec[0][-1]
        self._frames: list[tuple[Path, int, str]] = []
        self._metadata: dict[Path, dict[str, Any]] = {}
        self._is_complete = True
        super().__init__(PANDASET_BEV_ASSET, split=split, input_spec=input_spec)

    def _build_frame_list(self) -> None:
        self._frames, self._metadata, self._is_complete = self._build_bev_frame_list()

    def _validate_data(self) -> bool:
        return super()._validate_data() and self._is_complete

    def _build_bev_frame_list(
        self,
    ) -> tuple[list[tuple[Path, int, str]], dict[Path, dict[str, Any]], bool]:
        frames: list[tuple[Path, int, str]] = []
        metadata: dict[Path, dict[str, Any]] = {}
        is_complete = True
        if not self.data_root.exists():
            return frames, metadata, is_complete

        for sequence_dir in sorted(
            path for path in self.data_root.iterdir() if path.is_dir()
        ):
            camera_root = sequence_dir / "camera"
            lidar_poses_path = sequence_dir / "lidar" / "poses.json"
            try:
                camera_metadata = {
                    camera_name: {
                        "intrinsics": json.loads(
                            (camera_root / camera_name / "intrinsics.json").read_text()
                        ),
                        "poses": json.loads(
                            (camera_root / camera_name / "poses.json").read_text()
                        ),
                    }
                    for camera_name in CAMERA_NAMES
                }
                lidar_poses = json.loads(lidar_poses_path.read_text())
            except FileNotFoundError as e:
                print(f"Skipping sequence {sequence_dir.name}: missing {e.filename}")
                is_complete = False
                continue

            metadata[sequence_dir] = {
                "cameras": camera_metadata,
                "lidar_poses": lidar_poses,
            }
            image_paths = sorted((camera_root / "front_camera").glob("*.jpg"))
            if len(image_paths) != len(lidar_poses):
                is_complete = False
            for image_path in image_paths:
                frame_index = int(image_path.stem)
                if frame_index >= len(lidar_poses) or any(
                    frame_index >= len(camera_metadata[name]["poses"])
                    or not (camera_root / name / image_path.name).exists()
                    for name in CAMERA_NAMES
                ):
                    is_complete = False
                    continue
                frames.append((sequence_dir, frame_index, image_path.name))
        return frames, metadata, is_complete

    def _get_camera_inputs(
        self,
        frame: tuple[Path, int, str],
        metadata: dict[str, Any],
        input_height: int,
        input_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load the six camera views of one frame.

        Parameters
        ----------
        frame
            Sequence directory, frame index, and image filename.
        metadata
            Camera intrinsics, camera poses, and LiDAR poses for the sequence.
        input_height
            Output image height after cropping.
        input_width
            Output image width after resizing.

        Returns
        -------
        images : torch.Tensor
            Shape [6, 3, H, W], float32, RGB in range [0, 1], resized to the
            model width and top-cropped to the model height.
        intrinsics : torch.Tensor
            Shape [6, 3, 3], float32. Camera intrinsics adjusted for resize and crop.
        camera_to_reference : torch.Tensor
            Shape [6, 4, 4], float32. Transforms camera-frame points into the
            yaw-only LiDAR frame of the same frame.
        """
        sequence_dir, frame_index, image_name = frame
        world_to_reference = _pose_to_matrix(
            metadata["lidar_poses"][frame_index], inv=True, flat=True
        )

        images: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        camera_to_reference: list[torch.Tensor] = []
        for camera_name in CAMERA_NAMES:
            camera_dir = sequence_dir / "camera" / camera_name
            image_path = camera_dir / image_name
            image = Image.open(image_path).convert("RGB")
            original_width, original_height = image.size
            scale = input_width / original_width
            resized_height = int(original_height * scale)
            top_crop = resized_height - input_height
            if top_crop < 0:
                raise ValueError(
                    f"PandaSet image {image_path} is too short for "
                    f"{input_height}x{input_width} preprocessing."
                )
            image = image.resize((input_width, resized_height)).crop(
                (0, top_crop, input_width, resized_height)
            )
            images.append(
                torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
                / 255.0
            )

            camera_metadata = metadata["cameras"][camera_name]
            calibration = camera_metadata["intrinsics"]
            intrinsic = np.array(
                [
                    [calibration["fx"], 0.0, calibration["cx"]],
                    [0.0, calibration["fy"], calibration["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            intrinsic[0] *= scale
            intrinsic[1] *= scale
            intrinsic[1, 2] -= top_crop
            intrinsics.append(torch.from_numpy(intrinsic))

            camera_to_world = _pose_to_matrix(camera_metadata["poses"][frame_index])
            camera_to_reference.append(
                torch.from_numpy(
                    (world_to_reference @ camera_to_world).astype(np.float32)
                )
            )

        return (
            torch.stack(images),
            torch.stack(intrinsics),
            torch.stack(camera_to_reference),
        )

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/georghess/pandaset",
            split_description="PandaSet six-camera frames used for model calibration (CC-BY-4.0)",
        )

    def __getitem__(
        self, index: int
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        int,
    ]:
        """
        Get the six-camera BEV inputs of one frame.

        Parameters
        ----------
        index
            Index of the frame.

        Returns
        -------
        input_data : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            S = 6 cameras, H = image height, W = image width.
            images
                Shape [S*3, H, W], float32, RGB in range [0, 1].
            sensor2keyegos
                Shape [S, 4, 4], float32. Camera to yaw-only LiDAR frame.
            inv_intrins
                Shape [S, 3, 3], float32. Inverse of the adjusted camera intrinsics.
            inv_post_rots
                Shape [S, 3, 3], float32. Identity, as resize/crop is folded into the intrinsics.
            post_trans
                Shape [S, 1, 3], float32. Zeros, as resize/crop is folded into the intrinsics.
        label : int
            Dummy label (always 0 - calibration does not need ground truth).
        """
        frame = self._frames[index]
        images, intrinsics, sensor2keyegos = self._get_camera_inputs(
            frame, self._metadata[frame[0]], self.input_height, self.input_width
        )
        return (
            images.reshape(-1, self.input_height, self.input_width),
            sensor2keyegos,
            torch.inverse(intrinsics),
            torch.eye(3).expand(NUM_CAMERAS, -1, -1).clone(),
            torch.zeros(NUM_CAMERAS, 1, 3),
        ), 0

    @classmethod
    def dataset_name(cls) -> str:
        return "pandaset_bev"
