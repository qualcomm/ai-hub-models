# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np
import torch

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, extract_zip_file

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_semantic_kitti_files,
    )
except ImportError:
    download_semantic_kitti_files = None  # type: ignore[assignment]

SEMANTIC_KITTI_VERSION = 1
SEMANTIC_KITTI_ID = "semantic_kitti"
SEMANTIC_KITTI_LIDARS_DIR_NAME = "data_odometry_velodyne"
SEMANTIC_KITTI_GT_DIR_NAME = "data_odometry_labels"

# Pick a single sequence for train and validation to save disk space
# (full dataset is 22 sequences; 80GB)
VAL_SEQUENCE = "01"
TRAIN_SEQUENCE = "04"


class SemanticKittiDataset(BaseDataset):
    def __init__(
        self,
        input_lidars_zip: str | None = None,
        input_gt_zip: str | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_height: int = 64,
        input_width: int = 2048,
        max_points: int = 150000,  # max number of points present in dataset
    ) -> None:
        self.input_lidars_zip = input_lidars_zip
        self.input_gt_zip = input_gt_zip
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            SEMANTIC_KITTI_ID, SEMANTIC_KITTI_VERSION, "data"
        )
        self.sequences_path = os.path.join(self.data_path, "dataset/sequences")
        self.lidars_path = self.data_path / SEMANTIC_KITTI_LIDARS_DIR_NAME
        self.gt_path = self.data_path / SEMANTIC_KITTI_GT_DIR_NAME
        BaseDataset.__init__(self, self.data_path, split=split)

        self.proj_H = input_height
        self.proj_W = input_width
        self.max_points = max_points

        self.proj_fov_up = 3.0
        self.proj_fov_down = -25.0

        sequence = VAL_SEQUENCE if self.split == DatasetSplit.VAL else TRAIN_SEQUENCE

        self.sensor_img_means = torch.tensor(
            [12.12, 10.88, 0.23, -1.04, 0.21]
        )  # range,x,y,z,signal
        self.sensor_img_stds = torch.tensor(
            [12.32, 11.47, 6.91, 0.86, 0.16]
        )  # range,x,y,z,signal

        # get paths for each
        scan_path = os.path.join(self.sequences_path, sequence, "velodyne")
        label_path = os.path.join(self.sequences_path, sequence, "labels")

        # get files
        self.scan_files = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(scan_path))
            for f in fn
        ]
        self.label_files = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(label_path))
            for f in fn
        ]

        # check all scans have labels
        assert len(self.scan_files) == len(self.label_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()

    def _validate_data(self) -> bool:
        paths = [
            os.path.join(self.sequences_path, TRAIN_SEQUENCE, "velodyne"),
            os.path.join(self.sequences_path, TRAIN_SEQUENCE, "labels"),
            os.path.join(self.sequences_path, VAL_SEQUENCE, "velodyne"),
            os.path.join(self.sequences_path, VAL_SEQUENCE, "labels"),
        ]

        return all(os.path.exists(path) for path in paths)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Returns a tuple of input lidar proj tensor and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        proj : torch.Tensor
            Input lidar projection tensor.
        label_data : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            proj_x
                x coordinates of lidar points with shape [max_points,].
            proj_y
                y coordinates of lidar points with shape [max_points,].
            unproj_labels
                Semantic labels with shape [max_points,].
        """
        scan_file = self.scan_files[index]
        label_file = self.label_files[index]

        label = np.fromfile(label_file, dtype=np.int32)
        label = label.reshape(-1)
        sem_label = label & 0xFFFF  # semantic label in lower half

        scan = np.fromfile(scan_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        # put in attribute
        points = scan[:, 0:3]  # get xyz
        remissions = scan[:, 3]  # get remission
        (
            proj_x_numpy,
            proj_y_numpy,
            proj_range_numpy,
            proj_xyz_numpy,
            proj_remission_numpy,
            proj_mask_numpy,
        ) = self.do_range_projection(points, remissions)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = points.shape[0]
        unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
        unproj_labels[:unproj_n_points] = torch.from_numpy(sem_label)

        # get points and labels
        proj_range = torch.from_numpy(proj_range_numpy).clone()
        proj_xyz = torch.from_numpy(proj_xyz_numpy).clone()
        proj_remission = torch.from_numpy(proj_remission_numpy).clone()
        proj_mask = torch.from_numpy(proj_mask_numpy)

        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(proj_x_numpy)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(proj_y_numpy)
        proj = torch.cat(
            [
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
                proj_remission.unsqueeze(0).clone(),
            ]
        )
        proj -= self.sensor_img_means.reshape(-1, 1, 1)
        proj /= self.sensor_img_stds.reshape(-1, 1, 1)
        proj *= proj_mask.float()

        return proj, (
            proj_x,
            proj_y,
            unproj_labels,
        )

    def __len__(self) -> int:
        return len(self.scan_files)

    def _download_data(
        self, lidars_zip: str | None = None, gt_zip: str | None = None
    ) -> None:
        # Use passed args if provided, otherwise use instance attributes
        if lidars_zip is None:
            lidars_zip = self.input_lidars_zip
        if gt_zip is None:
            gt_zip = self.input_gt_zip

        # If no files provided/set, try auto-download
        if lidars_zip is None and download_semantic_kitti_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                lidars_zip = os.path.join(
                    tmpdir, f"{SEMANTIC_KITTI_LIDARS_DIR_NAME}.zip"
                )
                gt_zip = os.path.join(tmpdir, f"{SEMANTIC_KITTI_GT_DIR_NAME}.zip")
                download_semantic_kitti_files(
                    lidars_zip, gt_zip, SEMANTIC_KITTI_VERSION
                )
                self._download_data(lidars_zip, gt_zip)
            return

        if (
            lidars_zip is None
            or not lidars_zip.endswith(SEMANTIC_KITTI_LIDARS_DIR_NAME + ".zip")
            or gt_zip is None
            or not gt_zip.endswith(SEMANTIC_KITTI_GT_DIR_NAME + ".zip")
        ):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=[
                    "Open http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip, provide your Email address, and click the request download link button",
                    "Download the data_odometry_velodyne.zip file by the link sent to your email.",
                    "Download the data_odometry_labels.zip file at https://semantic-kitti.org/assets/data_odometry_labels.zip",
                    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset semantic_kitti --files /path/to/data_odometry_velodyne.zip /path/to/data_odometry_labels.zip`",
                ],
            )

        os.makedirs(self.lidars_path.parent, exist_ok=True)
        extract_zip_file(lidars_zip, self.lidars_path.parent)
        extract_zip_file(gt_zip, self.gt_path.parent)

        # Remove extraneous directories to save space
        for sequence in os.listdir(self.sequences_path):
            if sequence in [VAL_SEQUENCE, TRAIN_SEQUENCE]:
                continue
            subdir = os.path.join(self.sequences_path, sequence)
            if os.path.isdir(subdir):
                shutil.rmtree(subdir)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 50

    def do_range_projection(
        self, points: np.ndarray, remissions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)

        Parameters
        ----------
        points
            Predicted point cloud points
        remissions
            Predicted remissinos

        Returns
        -------
        img_proj_x : np.ndarray
            projections in image coords in x axis in range[0,W-1]
        img_proj_y : np.ndarray
            projections in image coords in y axis in range[0,H-1]
        proj_range : np.ndarray
            projected range image - [H,W] range (-1 is no data)
        proj_xyz : np.ndarray
            projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
        proj_remission : np.ndarray
            projected remission - [H,W] intensity (-1 is no data)
        proj_mask : np.ndarray
            projected index (for each pixel, what I am in the pointcloud)
            [H,W] index (-1 is no data)
        """
        # laser parameters
        fov_up = self.proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # get depth of all points
        depth = np.linalg.norm(points, 2, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= self.proj_W  # in [0.0, W]
        proj_y *= self.proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(self.proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        img_proj_x = np.copy(proj_x)  # store a copy in orig order

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(self.proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        img_proj_y = np.copy(proj_y)  # stope a copy in original order

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = points[order]
        remission = remissions[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_range = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        proj_xyz[proj_y, proj_x] = points

        proj_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_remission[proj_y, proj_x] = remission

        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices
        proj_mask = (proj_idx > 0).astype(np.int32)

        return img_proj_x, img_proj_y, proj_range, proj_xyz, proj_remission, proj_mask

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://semantic-kitti.org/",
            split_description="sequence #01 of 22",
        )
