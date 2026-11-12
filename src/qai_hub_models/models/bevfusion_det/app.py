# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    compute_corners,
    draw_3d_bbox,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.optimization import optimized_cumsum

OBJECT_CLASSES = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}


class BEVFusionApp:
    def __init__(
        self,
        encoder1: Callable[[torch.Tensor], torch.Tensor],
        encoder2: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        encoder3: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        decoder: Callable[[torch.Tensor], torch.Tensor],
        num_classes: list[int] | Any,
        task_heads: list[Any] | Any,
        get_bboxes: Callable[..., Any] | Any,
        model_input_shape: tuple[int, int] = (256, 704),
        score_threshold: float = 0.4,
        nms_threshold: float = 4.0,
        nms_post_max_size: int = 83,
    ) -> None:
        """
        Initialize BEVFusionApp.

        Parameters
        ----------
        encoder1
            Model encoder component.
            For input specs, see `BEVFusionEncoder1.get_input_spec` in model.py.
        encoder2
            Model encoder component.
            For input specs, see `BEVFusionEncoder2.get_input_spec` in model.py.
        encoder3
            Model encoder component.
            For input specs, see `BEVFusionEncoder3.get_input_spec` in model.py.
        decoder
            Model decoder component.
            For input spec, see ``BEVFusionDecoder.get_input_spec`` in model.py.
        num_classes
            List of per-task class counts from CenterHead (``heads.num_classes``).
            Used to determine how to split the decoder output tensor per task.
        task_heads
            List of per-task head modules from CenterHead (``heads.task_heads``).
            Each entry exposes a ``heads`` dict whose values carry the output
            channel count for each regression key (reg, height, dim, rot, vel).
        get_bboxes
            Bound method ``heads.get_bboxes`` from CenterHead.
            Called with the unpacked prediction dicts to decode raw outputs into
            bounding boxes, scores, and labels.
        model_input_shape
            (height, width) of the model input.
        score_threshold
            Score threshold, default is 0.4.
        nms_threshold
            NMS threshold, default is 4.0.
        nms_post_max_size
            Max boxes retained after NMS, default is 500.
        """
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.decoder = decoder
        self.num_classes = num_classes
        self.task_heads = task_heads
        self.get_bboxes = get_bboxes
        self.model_input_shape = model_input_shape
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_post_max_size = nms_post_max_size

    def prepare_camera_inputs(
        self,
        cam_paths: dict[str, str],
        inputs: dict,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Load intrinsics, and sensor-to-key-ego transformations.

        Parameters
        ----------
        cam_paths
            Dictionary mapping camera names to image file paths.
        inputs
            JSON dictionary containing intrinsics and transformation data.

        Returns
        -------
        intrins_list : list[np.ndarray]
            List of intrinsic matrices (3x3) for each camera.
        sensor2keyegos_list : list[np.ndarray]
            List of sensor-to-key-ego transformation matrices (4x4) for each camera.
        """
        intrins_list, sensor2keyegos_list = [], []

        l2e_r = inputs["lidar2ego_rotation"]
        l2e_t = np.array(inputs["lidar2ego_translation"])
        e2g_r = inputs["ego2global_rotation"]
        e2g_t = np.array(inputs["ego2global_translation"])
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        for cam_name in cam_paths:
            intrin = np.array(inputs[cam_name]["intrins"], dtype=np.float32)
            intrins_list.append(intrin)

            l2e_r_s = inputs[cam_name]["sensor2ego_rotation"]
            l2e_t_s = inputs[cam_name]["sensor2ego_translation"]
            e2g_r_s = inputs[cam_name]["ego2global_rotation"]
            e2g_t_s = inputs[cam_name]["ego2global_translation"]

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
            inputs["sensor2lidar_rotation"] = R.T  # points @ R.T + T
            inputs["sensor2lidar_translation"] = T
            sensor2keyegos = np.eye(4).astype(np.float32)
            sensor2keyegos[:3, :3] = inputs["sensor2lidar_rotation"]
            sensor2keyegos[:3, 3] = inputs["sensor2lidar_translation"]
            sensor2keyegos_list.append(sensor2keyegos)

        return intrins_list, sensor2keyegos_list

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image.Image]:
        return self.predict_3d_boxes_from_images(*args, **kwargs)

    def predict_3d_boxes_from_images(
        self,
        images_list: list[Image.Image],
        cam_paths: dict[str, str],
        inputs_json: dict,
        raw_output: bool = False,
    ) -> np.ndarray | list[Image.Image]:
        """
        Run the BEVFusion model and predict 3D bounding boxes.

        Parameters
        ----------
        images_list
            List of PIL Images in RGB format.
        cam_paths
            Dictionary mapping camera names to image file paths.
        inputs_json
            JSON dictionary containing intrinsics and transformation data.
        raw_output
            If True, returns raw 3D bounding box corners.

        Returns
        -------
        corners_or_output_images : np.ndarray | list[Image.Image]
            If raw_output is True:
                Corners of 3D bounding boxes with shape (N, 8, 3).
            If raw_output is False:
                List of PIL Images with 3D bounding boxes drawn.
        """
        intrins_list, sensor2keyegos_list = self.prepare_camera_inputs(
            cam_paths, inputs_json
        )

        sensor2keyegos = torch.tensor(sensor2keyegos_list).unsqueeze(0)
        inv_intrins = torch.inverse(torch.tensor(intrins_list))
        imgs, inv_post_rots, post_trans = self.preprocess_images(images_list)

        assert imgs.shape[0] == 1, "Model supports only single batch"
        x = self.encoder1(torch.tensor(imgs))

        x, lengths, geom_feats = self.encoder2(
            x.unsqueeze(0),
            inv_intrins.unsqueeze(0),
            sensor2keyegos,
            inv_post_rots,
            post_trans,
        )
        segment_indices = torch.cumsum(lengths, dim=0).long()

        x_reshaped = x.unsqueeze(0).reshape(96, 118, 176, 80).float()
        x = optimized_cumsum(x_reshaped).reshape(x.shape)
        x = x.reshape(-1, 80)
        segment = x[segment_indices, :]
        diff = segment[1:] - segment[:-1]
        segment = torch.cat([segment[:1], diff], dim=0)

        x = self.encoder3(segment.unsqueeze(0), geom_feats.unsqueeze(0))
        pred_tensor = self.decoder(x)

        # unpack predictions into dict
        pred_dicts = []
        start = 0
        head_order = ["reg", "height", "dim", "rot", "vel", "heatmap"]
        for i, nc in enumerate(self.num_classes):
            task_head = self.task_heads[i]
            reg_heads = getattr(task_head, "heads", None)
            channels = []
            for key in head_order:
                if key == "heatmap":
                    channels.append(nc)
                else:
                    if reg_heads is None:
                        raise ValueError(f"reg_heads is None for task_head {i}")
                    channels.append(reg_heads[key][0])

            total = sum(channels)
            split_slice = pred_tensor[:, start : start + total]
            split_tensors = torch.split(split_slice, channels, dim=1)
            start += total

            pred_dict = dict(zip(head_order, split_tensors, strict=False))
            pred_dicts.append([pred_dict])

        bboxes, scores, labels = self.get_bboxes(pred_dicts)[0]
        corners = compute_corners(bboxes)

        # Filter based on confidence score
        indices = torch.tensor(scores) >= self.score_threshold
        bboxes, corners, scores, labels = (
            bboxes[indices],
            corners[indices],
            scores[indices],
            labels[indices],
        )

        bboxes_np = bboxes.numpy()
        corners_np = corners.numpy()

        corners_np[..., 2] -= bboxes_np[:, None, 5] / 2

        if raw_output:
            return corners_np

        # Filter based on confidence score

        output_images = []
        corners_keyego = corners_np.reshape(-1, 3)
        corners_keyego = np.concatenate(
            [corners_keyego, np.ones([corners_keyego.shape[0], 1])], axis=1
        )
        # lidar2image differs for each camera
        for k, pil_img in enumerate(images_list):
            image = np.array(pil_img)
            corners_sensor = corners_keyego @ np.linalg.inv(sensor2keyegos_list[k]).T
            image_coor = (corners_sensor[:, :3] @ intrins_list[k].T).reshape(-1, 8, 3)
            valid_indices = np.all(image_coor[..., 2] > 0, axis=1)
            image_coor = image_coor[..., :2] / image_coor[..., 2:3]
            image = draw_3d_bbox(
                image, image_coor[valid_indices], labels[valid_indices], OBJECT_CLASSES
            )
            output_images.append(Image.fromarray(image))

        return output_images

    def preprocess_images(
        self, images: list[Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess a list of camera images for BEVFusion model input.

        Resizes and crops input images to match the expected input dimensions of BEVFusionEncoder1,
        computes post-rotation and post-translation matrices, and concatenates images into a single tensor.

        Parameters
        ----------
        images
            List of PIL images from multiple cameras.

        Returns
        -------
        image_tensor : torch.Tensor
            Concatenated image tensor with shape [B, S*C, H, W],
            where B=1 (batch size), S=number of cameras, C=3 (RGB channels), H=height, W=width.
        inv_post_rots : torch.Tensor
            Inverse post-rotation matrices with shape [1, S, 3, 3].
        post_trans : torch.Tensor
            Post-translation vectors with shape [1, S, 1, 3].
        """
        images_tensor_list = []
        post_rot_list = []
        post_tran_list = []
        for img in images:
            W, H = img.size
            fH, fW = self.model_input_shape
            resize = 0.48
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            img = img.resize(resize_dims).crop(crop)

            rotation, translation = self.get_post_rot_and_tran(
                resize=resize,
                crop=crop,
                rotate=0,
            )
            _, img_tensor = app_to_net_image_inputs(img)

            images_tensor_list.append(img_tensor)
            post_rot_list.append(rotation)
            post_tran_list.append(translation)

        post_rots = torch.stack(post_rot_list)
        inv_post_rots = torch.inverse(post_rots).unsqueeze(0)
        post_trans = torch.stack(post_tran_list).reshape(1, -1, 1, 3)
        image_tensor = torch.concat(images_tensor_list, 1)
        return image_tensor, inv_post_rots, post_trans

    def img_transform(
        self,
        img: Image.Image,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        resize: float,
        crop: tuple[int, int, int, int],
        rotate: int,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
        """
        Apply post-homography transformation to image and associated transformation matrices.

        Parameters
        ----------
        img
            Input image to be transformed.
        rotation
            Rotation matrix of shape (3, 3) to be adjusted.
        translation
            Translation vector of shape (3,) to be adjusted.
        resize
            Scaling factor for resizing the image and transformations.
        crop
            Crop coordinates (left, upper, right, lower).
        rotate
            Rotation angle in degrees.

        Returns
        -------
        transformed_image : Image.Image
            Transformed PIL image.
        adjusted_rotation : torch.Tensor
            Adjusted rotation matrix of shape (3, 3).
        adjusted_translation : torch.Tensor
            Adjusted translation vector of shape (3,).
        """
        # Apply scaling to rotation matrix based on resize factor
        rotation *= resize

        translation -= torch.Tensor(crop[:2])
        theta: float = rotate / 180 * np.pi
        # Create rotation matrix A for the given angle
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        # Calculate center offset vector b based on crop dimensions
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        # Adjust b by applying negative rotation and adding back the offset
        b = A.matmul(-b) + b

        rotation = A.matmul(rotation)
        # Apply rotation and offset to the translation vector
        translation = A.matmul(translation) + b

        return img, rotation, translation

    def get_post_rot_and_tran(
        self, resize: float, crop: tuple[int, int, int, int], rotate: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get post rotation and post translation

        Parameters
        ----------
        resize
            Scaling factor for resizing the image and transformations.
        crop
            Tuple of (left, upper, right, lower) crop coordinates.
        rotate
            Rotation angle in degrees.

        Returns
        -------
        post_rot : torch.Tensor
            Post rotation matrix with shape [3, 3] in camera coordinate system.
        post_tran : torch.Tensor
            Post translation tensor with shape [3,] in camera coordinate system.
        """
        post_rot = torch.eye(3)
        post_tran = torch.zeros(3)

        # post-homography transformation
        post_rot[:2, :2] *= resize
        post_tran[:2] -= torch.Tensor(crop[:2])

        rotate_angle = torch.tensor(rotate / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        A = torch.Tensor([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot[:2, :2] = A.matmul(post_rot[:2, :2])
        post_tran[:2] = A.matmul(post_tran[:2]) + b

        return post_rot, post_tran
