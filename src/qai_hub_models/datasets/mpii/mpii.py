# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
import os
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io import loadmat

from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

MPII_FOLDER_NAME = "mpii"
MPII_VERSION = 2
MPII_ASSET = CachedWebDatasetAsset.from_asset_store(
    MPII_FOLDER_NAME,
    MPII_VERSION,
    "mpii_human_pose_v1_train_val.tar.gz",
)

GT = CachedWebDatasetAsset.from_asset_store(
    MPII_FOLDER_NAME,
    MPII_VERSION,
    "gt_valid.mat",
)
ANNO = {"train": "train.json", "val": "valid.json"}


class MPIIDatasetBase(BaseDataset):
    """
    Shared machinery for MPII Human Pose datasets (download, image indexing).
    Subclasses implement ``__getitem__`` to return the ground truth shape they
    need (pose keypoints, calibration-only images, etc.).

    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download

    MPII keypoints:

        0: 'right_ankle'
        1: 'right_knee',
        2: 'right_hip',
        3: 'left_hip',
        4: 'left_knee',
        5: 'left_ankle',
        6: 'pelvis',
        7: 'thorax',
        8: 'upper_neck',
        9: 'head_top',
        10: 'right_wrist',
        11: 'right_elbow',
        12: 'right_shoulder',
        13: 'left_shoulder',
        14: 'left_elbow',
        15: 'left_wrist'
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        num_samples: int = -1,
    ) -> None:
        BaseDataset.__init__(self, MPII_ASSET.extracted_path.parent, split)
        assert self.split_str in ["train", "val"]
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 256, 192))}
        self.num_joints = 16
        gt_mat = GT.fetch()
        anno_path = CachedWebDatasetAsset.from_asset_store(
            MPII_FOLDER_NAME,
            MPII_VERSION,
            ANNO[self.split_str],
        ).fetch()

        self.gt_dict = loadmat(gt_mat)

        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]

        with open(anno_path) as anno_file:
            anno = json.load(anno_file)

        self.images_dict_list = []
        for a in anno:
            image_name = a["image"]

            c = np.array(a["center"], dtype=float)
            s = np.array([a["scale"], a["scale"]], dtype=float) * 200.0

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            self.images_dict_list.append(
                {
                    "image_path": os.path.join(MPII_ASSET.extracted_path, image_name),
                    "center": c,
                    "scale": s,
                }
            )
            if len(self.images_dict_list) == num_samples:
                break

    def _get_keypoint_item(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, NDArray, NDArray],
    ]:
        """
        Parameters
        ----------
        index
            The index to retrieve.

        Returns
        -------
        input_image : torch.Tensor
            NCHW input image of range [0-1] and RGB channel layout.
            Height and width will confirm to self.input_spec.

        gt_data : tuple[torch.Tensor, torch.Tensor, torch.Tensor, NDArray, NDArray]
            gt_keypoints
                shape (self.num_joints, 2):
                coordinates of ground truth keypoint
            headboxes
                shape (2, 2):
                [[x1,y1]     top-left corner
                 [x2,y2]]    bottom-right corner
            joint_missing
                shape (self.num_joints,):
                value 0: joint is visible
                value 1: joint is missing
            center
                shape (2,):
                center used for image transform
            scale
                shape (2,):
                scale used for image transform
        """
        image_dict = self.images_dict_list[index]
        image_path = cast(str, image_dict["image_path"])
        data_numpy = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        assert data_numpy is not None, f"Failed to load image: {image_path}"

        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        center = cast(NDArray, image_dict["center"])
        scale = cast(NDArray, image_dict["scale"])
        rotate = 0

        # transforms image
        image = pre_process_with_affine(
            data_numpy, center, scale, rotate, (self.input_width, self.input_height)
        ).squeeze(0)

        joint_missing = self.gt_dict["jnt_missing"][..., index]
        gt_keypoints = self.gt_dict["pos_gt_src"][..., index]
        headboxes = self.gt_dict["headboxes_src"][..., index]

        return image, (gt_keypoints, headboxes, joint_missing, center, scale)

    def __len__(self) -> int:
        return len(self.images_dict_list)

    def _download_data(self) -> None:
        MPII_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset",
            split_description="validation split",
        )


class MPIIDataset(MPIIDatasetBase):
    """MPII Human Pose dataset returning keypoint ground truth."""

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, NDArray, NDArray],
    ]:
        return self._get_keypoint_item(index)
