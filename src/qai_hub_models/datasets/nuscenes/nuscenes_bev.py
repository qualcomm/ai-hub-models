# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.nuscenes.nuscenes import NuscenesDatasetBase
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    transform_to_matrix,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

NUSCENE_ID = "nuscenes"
NUSCENE_FILE = "v1.0-mini"
NUSCENE_VERSION = 2
NUM_CLASSES = 12  # Based on CLASSES = STATIC + DIVIDER + DYNAMIC
NUSCENE_LABEL = CachedWebDatasetAsset.from_asset_store(
    NUSCENE_ID,
    NUSCENE_VERSION,
    "cvt_labels_nuscenes.tar.gz",
)


class NuscenesBevDatasetBase(NuscenesDatasetBase):
    """
    Shared machinery for nuScenes Bird's-Eye-View (BEV) segmentation datasets
    using 6 camera inputs. ``_get_bev_item`` loads the common camera tensors
    and BEV labels; subclasses implement ``__getitem__`` to reshape these into
    the per-model input layout.
    """

    def __init__(
        self,
        dataset_file: str | None = None,
        input_spec: InputSpec | None = None,
        top_crop: int | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
    ) -> None:
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 6, 3, 224, 480))}
        super().__init__(
            source_dataset_file=dataset_file,
            split=split,
            input_spec=input_spec,
        )
        # Remove top part of image (sky + car hood) - not useful for road detection
        # Default 46 pixels works well for BEV tasks
        self.top_crop = top_crop if top_crop is not None else 46
        self.input_height = input_spec["image"][0][3]
        self.input_width = input_spec["image"][0][4]
        self.bev_labels_dir = NUSCENE_LABEL.extracted_path
        if not os.path.exists(self.bev_labels_dir):
            NUSCENE_LABEL.fetch(extract=True)

        self._load_bev_labels()

    def _load_bev_labels(self) -> None:
        """Load BEV and Visibility label paths from JSON files in the BEV and Visibility labels directory."""
        self.bev_labels = {}
        self.visibility_labels = {}
        labels_path = Path(self.bev_labels_dir)
        for scene_file in labels_path.glob("scene-*.json"):
            with open(scene_file) as f:
                scene_data = json.load(f)
                for sample in scene_data:
                    bev_path = sample.get("bev", None)
                    if bev_path is not None:
                        bev_image_path = os.path.join(
                            self.bev_labels_dir, sample["scene"], bev_path
                        )
                        self.bev_labels[sample["token"]] = (
                            bev_image_path if os.path.exists(bev_image_path) else None
                        )
                    visibility_path = sample.get("visibility", None)
                    if visibility_path is not None:
                        visibility_image_path = os.path.join(
                            self.bev_labels_dir, sample["scene"], visibility_path
                        )
                        self.visibility_labels[sample["token"]] = (
                            visibility_image_path
                            if os.path.exists(visibility_image_path)
                            else None
                        )

    def _get_bev_item(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Get item from infos according to the given index.
        Returns a tuple of input tensors and ground truth data.

        Parameters
        ----------
        idx
            Index of the sample to retrieve.

        Returns
        -------
        input_data : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            images
                Shape [S, 3, H, W], float32.
                Preprocessed RGB images from S=6 cameras, normalized to [0, 1], size 224x480.
            intrinsics
                Shape [S, 3, 3] as float32.
                Camera intrinsic matrices, adjusted for resize and crop.
            extrinsics
                Shape [S, 4, 4] as float32.
                Matrices transforming camera sensor to global frame, aligned with LiDAR.
        gt_data : tuple[torch.Tensor, torch.Tensor]
            gt_bev
                torch.Tensor of shape [H_bev, W_bev, 12] as float32
                Binary BEV segmentation map with 12 classes.
            visibility
                torch.Tensor of shape [H_bev, W_bev] as uint8
                in range [1-255], higher value = more visible
        """
        info = self.data_infos[idx]
        token = info.token
        cam_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        # Load BEV ground truth
        bev_path = self.bev_labels.get(token)
        assert bev_path is not None, f"BEV label not found for token: {token}"
        bev_image = Image.open(bev_path)

        visibility_path = self.visibility_labels.get(token)
        assert visibility_path is not None, (
            f"visibility_path not found for token: {token}"
        )
        visibility_image = Image.open(visibility_path)
        visibility = torch.tensor(np.array(visibility_image, dtype=np.uint8))

        # The CVT labels are bit-packed: each pixel stores 12 class flags in 12 bits
        # Unpack using bitwise shift + mask to get one-hot per class
        shift = np.arange(NUM_CLASSES, dtype=np.int32)[None, None]
        x = np.asarray(bev_image)[..., None]
        x = (x >> shift) & 1
        gt_bev = torch.tensor(x.astype(np.float32))

        # Get camera images and transformations
        images: list[torch.Tensor] = []
        intrinsics: list[torch.Tensor] = []
        extrinsics: list[torch.Tensor] = []

        for cam_name in cam_names:
            if cam_name in info.cams:
                cam_info = info.cams[cam_name]

                # Load and preprocess image
                img_path = os.path.join(self.data_path, cam_info["data_path"])
                image = Image.open(img_path).convert("RGB")

                # Resize then crop
                h_resize = self.input_height + self.top_crop
                w_resize = self.input_width
                original_w, original_h = image.size
                image_resized = image.resize(
                    (w_resize, h_resize), Image.Resampling.BILINEAR
                )
                image_cropped = image_resized.crop(
                    (0, self.top_crop, w_resize, h_resize)
                )

                # Convert to tensor
                image_tensor = torch.from_numpy(np.array(image_cropped)).float() / 255.0
                image_tensor = image_tensor.permute(2, 0, 1)
                images.append(image_tensor)

                # Process intrinsics
                intrinsic = np.array(cam_info["cam_intrinsic"], dtype=np.float32)
                scale_x = w_resize / original_w
                scale_y = h_resize / original_h
                intrinsic[0, 0] *= scale_x  # fx
                intrinsic[0, 2] *= scale_x  # cx
                intrinsic[1, 1] *= scale_y  # fy
                intrinsic[1, 2] *= scale_y  # cy
                intrinsic[1, 2] -= self.top_crop  # cy adjustment for crop
                intrinsics.append(torch.tensor(intrinsic))

                # Process extrinsics using transform_to_matrix
                sensor2ego = transform_to_matrix(
                    translation=cast(list[float], cam_info["sensor2ego_translation"]),
                    rotation=cast(list[float], cam_info["sensor2ego_rotation"]),
                    inv=True,
                )
                ego2global = transform_to_matrix(
                    translation=cast(list[float], cam_info["ego2global_translation"]),
                    rotation=cast(list[float], cam_info["ego2global_rotation"]),
                    inv=True,
                )
                world_from_egolidarflat = transform_to_matrix(
                    translation=cast(list[float], info.ego2global_translation),
                    rotation=cast(list[float], info.ego2global_rotation),
                    flat=True,
                )
                extrinsic = sensor2ego @ ego2global @ world_from_egolidarflat
                extrinsics.append(torch.tensor(extrinsic, dtype=torch.float32))

        # Stack lists into tensors
        images_tensor = torch.stack(images)
        intrinsics_tensor = torch.stack(intrinsics)
        extrinsics_tensor = torch.stack(extrinsics)

        return (
            images_tensor,
            intrinsics_tensor,
            extrinsics_tensor,
        ), (gt_bev, visibility)


class NuscenesBevDataset(NuscenesBevDatasetBase):
    """nuScenes BEV segmentation dataset returning 6-camera inputs."""

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        return self._get_bev_item(idx)


class NuscenesBevGKTDataset(NuscenesBevDatasetBase):
    """Wrapper around nuScenes BEV dataset for GKT model."""

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        (images_tensor, intrinsics_tensor, extrinsics_tensor), (gt_bev, visibility) = (
            self._get_bev_item(idx)
        )
        return (
            images_tensor,
            intrinsics_tensor,
            extrinsics_tensor,
            torch.inverse(intrinsics_tensor),
            torch.inverse(extrinsics_tensor),
        ), (gt_bev, visibility)

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "nuscenes_bev_gkt"


class NuscenesBevCVTDataset(NuscenesBevDatasetBase):
    """Wrapper around nuScenes BEV dataset for CVT model."""

    def __getitem__(
        self, idx: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        (images_tensor, intrinsics_tensor, extrinsics_tensor), (gt_bev, visibility) = (
            self._get_bev_item(idx)
        )
        return (
            images_tensor,
            torch.inverse(intrinsics_tensor),
            torch.inverse(extrinsics_tensor),
        ), (gt_bev, visibility)

    @classmethod
    def dataset_name(cls) -> str:
        """
        Name for the dataset,
            which by default is set to the filename where the class is defined.
        """
        return "nuscenes_bev_cvt"
