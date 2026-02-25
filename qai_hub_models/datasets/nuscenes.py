# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import tarfile
import tempfile
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
    UnfetchableDatasetError,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG, load_json

try:
    from qai_hub_models.utils._internal.download_private_datasets import (
        download_nuscenes_files,
    )
except ImportError:
    download_nuscenes_files = None  # type: ignore[assignment]
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    get_post_rot_and_tran,
)
from qai_hub_models.utils.input_spec import InputSpec

NUSCENE_ID = "nuscenes"
NUSCENE_FILE = "v1.0-mini"
NUSCENE_VERSION = 1


@dataclass
class NuScenesSampleInfo:
    """
    Container for essential metadata of a NuScenes dataset sample.

    Attributes
    ----------
    token
        Unique identifier for the sample
    cams
        Camera data organized by each sensor name,
        containing image paths and calibration parameters
    lidar2ego_translation
        Translation vector [x, y, z] in meters (shape: (3,))
        from LiDAR sensor frame to ego vehicle frame
    lidar2ego_rotation
        3x3 rotation matrix (shape: (3, 3)) from LiDAR to ego frame
    ego2global_translation
        Translation vector [x, y, z] in meters (shape: (3,))
        from ego vehicle frame to global coordinate system
    ego2global_rotation
        3x3 rotation matrix (shape: (3, 3)) from ego to global frame
    """

    token: str
    cams: dict
    lidar2ego_translation: np.ndarray
    lidar2ego_rotation: np.ndarray
    ego2global_translation: np.ndarray
    ego2global_rotation: np.ndarray


class NuscenesDataset(BaseDataset):
    """Wrapper class around Nuscenes Dataset"""

    def __init__(
        self,
        source_dataset_file: str | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.data_path = ASSET_CONFIG.get_local_store_dataset_path(
            NUSCENE_ID, NUSCENE_VERSION, "data"
        )
        self.source_dataset_file = source_dataset_file
        BaseDataset.__init__(self, str(self.data_path), split)

        # WARNING: This must be included after the Base __init__ to allow UnfetchableDatasetError
        # to be thrown before the ImportError, if the dataset is not downloaded to disk.
        try:
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils import splits
        except ImportError as e:
            raise ImportError(
                "nuscenes-devkit must be installed to create the nuscenes dataset."
            ) from e

        sample_json_path = os.path.join(self.data_path, "v1.0-mini", "sample.json")
        samples = load_json(sample_json_path)

        self.token_to_id = {sample["token"]: idx for idx, sample in enumerate(samples)}
        self.id_to_token = {idx: sample["token"] for idx, sample in enumerate(samples)}

        self.nusc = NuScenes(
            version="v1.0-mini", dataroot=self.data_path, verbose=False
        )

        if split == DatasetSplit.TRAIN:
            self.data_infos = self._fill_infos(splits.mini_train)
        else:
            self.data_infos = self._fill_infos(splits.mini_val)
        input_spec = input_spec or {"image": ((1, 3, 256, 704), "")}
        self.input_height = input_spec["image"][0][2]
        self.input_width = input_spec["image"][0][3]

    def __getitem__(
        self, index: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[int, torch.Tensor, torch.Tensor],
    ]:
        """
        Get item from infos according to the given index.
        Returns a tuple of input tensors and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        input_data : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            S = number of cameras, C = 3, H = img height, W = img width
            imgs
                torch.Tensor of shape [S*C, H, W] as float32
                Preprocessed image with range[0-1] in RGB format.
            sensor2keyegos
                torch.Tensor of shape [S, 4, 4] as float32
                transformation matix to convert from camera sensor
                to ego-vehicle at front camera coordinate frame
            inv_intrins
                torch.Tensor of shape [S, 3, 3] as float32
                Inverse of Camera intrinsic matrix
                used to project 2D image coordinates to 3D points
            inv_post_rots
                torch.tensor with shape [N, 3, 3] as float32
                inverse post rotation matrix in camera coordinate system
            post_trans
                torch.tensor with shape [N, 1, 3] as float32
                post translation tensor in camera coordinate system

        gt_data : tuple[int, torch.Tensor, torch.Tensor]
            samplet_token
                Unique identifier for the sample.
            trans
                ego2global Translation with the shape of [3,].
            rots
                ego2global Rotation with the shape of [4,].
        """
        from pyquaternion import Quaternion

        info = self.data_infos[index]

        imgs_list = []
        sensor2egos_list = []
        ego2globals_list = []
        intrins_list = []
        post_rots_list = []
        post_trans_list = []
        cam_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]

        for cam_name in cam_names:
            cam_data = info.cams[cam_name]
            filename = cam_data["data_path"]

            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data["cam_intrinsic"])

            # sweep sensor to sweep ego
            w, x, y, z = info.cams[cam_name]["sensor2ego_rotation"]
            sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
            sensor2ego_tran = torch.Tensor(
                info.cams[cam_name]["sensor2ego_translation"]
            )
            sensor2ego = sensor2ego_rot.new_zeros((4, 4))
            sensor2ego[3, 3] = 1
            sensor2ego[:3, :3] = sensor2ego_rot
            sensor2ego[:3, -1] = sensor2ego_tran

            # sweep ego to global
            w, x, y, z = info.cams[cam_name]["ego2global_rotation"]
            ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
            ego2global_tran = torch.Tensor(
                info.cams[cam_name]["ego2global_translation"]
            )
            ego2global = ego2global_rot.new_zeros((4, 4))
            ego2global[3, 3] = 1
            ego2global[:3, :3] = ego2global_rot
            ego2global[:3, -1] = ego2global_tran

            W, H = img.size
            resize = float(self.input_width) / float(W)
            resize_dims = (int(W * resize), int(H * resize))
            crop_h = resize_dims[1] - self.input_height
            crop_w = int(max(0, resize_dims[0] - self.input_width) / 2)
            crop = (
                crop_w,
                crop_h,
                crop_w + self.input_width,
                crop_h + self.input_height,
            )

            img = img.resize(resize_dims).crop(crop)
            post_rot, post_tran = get_post_rot_and_tran(
                resize=resize, crop=crop, rotate=0
            )

            _, img_tensor = app_to_net_image_inputs(img)
            imgs_list.append(img_tensor)
            intrins_list.append(intrin)
            sensor2egos_list.append(sensor2ego)
            ego2globals_list.append(ego2global)
            post_rots_list.append(post_rot)
            post_trans_list.append(post_tran)

        imgs = torch.concat(imgs_list, dim=1).squeeze(0)

        sensor2egos = torch.stack(sensor2egos_list)
        ego2globals = torch.stack(ego2globals_list)
        intrins = torch.stack(intrins_list)

        inv_post_rots = torch.inverse(torch.stack(post_rots_list))
        post_trans = torch.stack(post_trans_list).reshape(-1, 1, 3)

        # bug in source repo of bevdet
        # model is trained on camera front left as key ego,
        # it should be camera front as mentioned in paper,
        # using camera front left to maintain the accuracy
        global2keyego = torch.inverse(ego2globals[0])

        # transformation matix to convert from camera sensor
        # to ego-vehicle at front camera coordinate frame
        sensor2keyegos = global2keyego @ ego2globals @ sensor2egos

        # used to project 2D image coordinates to 3D points
        inv_intrins = torch.inverse(torch.tensor(intrins))
        return (imgs, sensor2keyegos, inv_intrins, inv_post_rots, post_trans), (
            self.token_to_id[info.token],
            torch.tensor(info.cams["CAM_FRONT_LEFT"]["ego2global_translation"]),
            torch.tensor(info.cams["CAM_FRONT_LEFT"]["ego2global_rotation"]),
        )

    def _fill_infos(self, scenes: list[str]) -> list[NuScenesSampleInfo]:
        """
        Generate the train/val infos from the raw data.

        Parameters
        ----------
        scenes
            list of scenes names(train or val scenes).

        Returns
        -------
        sample_infos : list[NuScenesSampleInfo]
            Information of training set or validation set
            that will be saved to the info file.
        """
        from pyquaternion import Quaternion

        nusc_infos = []

        all_scene = {s["token"]: s["name"] for s in self.nusc.scene}

        for sample in self.nusc.sample:
            if all_scene[sample["scene_token"]] not in scenes:
                continue
            sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            cs_record = self.nusc.get(
                "calibrated_sensor", sd_rec["calibrated_sensor_token"]
            )
            pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

            info = NuScenesSampleInfo(
                token=sample["token"],
                cams={},
                lidar2ego_translation=cs_record["translation"],
                lidar2ego_rotation=cs_record["rotation"],
                ego2global_translation=pose_record["translation"],
                ego2global_rotation=pose_record["rotation"],
            )

            l2e_r = info.lidar2ego_rotation
            l2e_t = info.lidar2ego_translation
            e2g_r = info.ego2global_rotation
            e2g_t = info.ego2global_translation
            l2e_r_mat = Quaternion(l2e_r).rotation_matrix
            e2g_r_mat = Quaternion(e2g_r).rotation_matrix

            # obtain 6 image's information per frame
            camera_types = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]
            for cam in camera_types:
                cam_token = sample["data"][cam]
                _, _, cam_intrinsic = self.nusc.get_sample_data(cam_token)
                cam_info = self.obtain_sensor2top(
                    cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                )
                cam_info.update(cam_intrinsic=cam_intrinsic)
                info.cams.update({cam: cam_info})

            nusc_infos.append(info)
        return nusc_infos

    def obtain_sensor2top(
        self,
        sensor_token: str,
        l2e_t: np.ndarray,
        l2e_r_mat: np.ndarray,
        e2g_t: np.ndarray,
        e2g_r_mat: np.ndarray,
        sensor_type: str = "lidar",
    ) -> dict:
        """
        Calculates the transformation matrices and sweep information to transform
        points from a given sensor's coordinate system to the top LiDAR's
        coordinate system: sensor -> ego vehicle -> global -> ego vehicle -> top_lidar.

        Transformation logic adapted from:
        https://github.com/HuangJunJie2017/BEVDet/blob/26144be7c11c2972a8930d6ddd6471b8ea900d13/tools/data_converter/nuscenes_converter.py#L276

        Parameters
        ----------
        sensor_token
            Sample data token for the specific camera sensor.
        l2e_t
            LiDAR-to-ego translation vector (shape: [1, 3])
        l2e_r_mat
            LiDAR-to-ego rotation matrix (shape: [3, 3])
        e2g_t
            Ego-to-global translation vector (shape: [1, 3])
        e2g_r_mat
            Ego-to-global rotation matrix (shape: [3, 3])
        sensor_type
            Sensor type to calibrate.

        Returns
        -------
        sweep_info : dict
            Transformed sweep information containing calibrated point data.
        """
        from pyquaternion import Quaternion

        sd_rec = self.nusc.get("sample_data", sensor_token)
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        data_path = str(self.nusc.get_sample_data_path(sd_rec["token"]))
        if os.getcwd() in data_path:
            data_path = data_path.split(f"{os.getcwd()}/")[-1]
        sweep = {
            "data_path": data_path,
            "type": sensor_type,
            "sample_data_token": sd_rec["token"],
            "sensor2ego_translation": cs_record["translation"],
            "sensor2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
        }
        l2e_r_s = sweep["sensor2ego_rotation"]
        l2e_t_s = sweep["sensor2ego_translation"]
        e2g_r_s = sweep["ego2global_rotation"]
        e2g_t_s = sweep["ego2global_translation"]

        # obtain the RT from sensor to Top LiDAR
        # sweep->ego->global->ego'->lidar
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        T -= (
            e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            + l2e_t @ np.linalg.inv(l2e_r_mat).T
        )
        sweep["sensor2lidar_rotation"] = R.T
        sweep["sensor2lidar_translation"] = T
        return sweep

    def __len__(self) -> int:
        return len(self.data_infos)

    def _download_data(self, tgz_path: str | None = None) -> None:
        # Use passed arg if provided, otherwise use instance attribute
        if tgz_path is None:
            tgz_path = self.source_dataset_file

        # If no file provided/set, try auto-download
        if tgz_path is None and download_nuscenes_files is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                tgz_path = os.path.join(tmpdir, f"{NUSCENE_FILE}.tgz")
                download_nuscenes_files(tgz_path, NUSCENE_VERSION)
                self._download_data(tgz_path)
            return

        if tgz_path is None or not tgz_path.endswith(NUSCENE_FILE + ".tgz"):
            raise UnfetchableDatasetError(
                dataset_name=self.dataset_name(),
                installation_steps=[
                    "Create an account and login in https://www.nuscenes.org/nuscenes#download",
                    "Download the v1.0-mini.tgz file from the website.",
                    "Run `python -m qai_hub_models.datasets.configure_dataset --dataset nuscenes --files /path/to/v1.0-mini.tgz`",
                ],
            )

        with tarfile.open(tgz_path) as f:
            f.extractall(self.data_path)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 50
