# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Generator
from typing import cast

import cv2
import numpy as np
import torch

from qai_hub_models.models.agpose.dataset import (
    DEPTH_NORM_SCALE,
    REAL_CX,
    REAL_CY,
    REAL_FX,
    REAL_FY,
    compute_fps_indices,
    preprocess_rgb,
    sample_choose_and_pts,
)
from qai_hub_models.models.agpose.model import (
    AGPoseBackbone,
    AGPoseCollection,
    AGPoseRotationHead,
)
from qai_hub_models.models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.base_app import CollectionModelEvalGenerator
from qai_hub_models.utils.bounding_box_processing_3d import draw_3d_bbox, get_3d_bbox
from qai_hub_models.utils.depth_completion import fill_missing
from qai_hub_models.utils.image_processing_3d import (
    build_camera_projection_matrix,
    depth_to_full_pointcloud,
    project_to_image,
)
from qai_hub_models.utils.inference import AsyncOnDeviceModel, AsyncOnDeviceResult


class AGPoseCollectionApp:
    """
    End-to-end application for AGPoseCollection (WorkbenchModelCollection).

    Handles preprocessing (RGB resize, depth hole-filling, point cloud
    back-projection, choose index computation), inference via backbone
    and rotation head components, and result formatting (rotation, translation,
    size) and 3D bbox visualization.

    Component order (matches from_components):
        models[0] = AGPoseBackbone     (float)
        models[1] = AGPoseRotationHead (float)
    """

    def __init__(
        self,
        backbone: AGPoseBackbone | AsyncOnDeviceModel,
        rotation_head: AGPoseRotationHead | AsyncOnDeviceModel,
        intrinsics: dict | None = None,
    ) -> None:
        self.backbone = backbone
        self.rotation_head = rotation_head
        self.intrinsics = (
            intrinsics
            if intrinsics is not None
            else {"fx": REAL_FX, "fy": REAL_FY, "cx": REAL_CX, "cy": REAL_CY}
        )

    @classmethod
    def from_pretrained(cls) -> AGPoseCollectionApp:
        collection = AGPoseCollection.from_pretrained()
        return cls(collection.backbone, collection.rotation_head)

    @classmethod
    def from_components(
        cls,
        models: list[ExecutableModelProtocol] | list[AsyncOnDeviceModel],
    ) -> AGPoseCollectionApp:
        """
        Construct from ordered list of compiled/local components.
        models[0] = backbone, models[1] = rotation_head.
        """
        return cls(
            models[0],  # type: ignore[arg-type]
            models[1],  # type: ignore[arg-type]
        )

    @property
    def uses_ondevice_model(self) -> bool:
        return isinstance(self.backbone, AsyncOnDeviceModel) or isinstance(
            self.rotation_head, AsyncOnDeviceModel
        )

    def preprocess_depth(
        self,
        depth: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Back-project depth to point cloud and compute choose indices.

        Parameters
        ----------
        depth
            Raw depth image in sensor units, shape (H, W), dtype uint16.
        bbox
            Bounding box (rmin, rmax, cmin, cmax) in pixel coordinates.
        mask
            Binary instance mask, shape (H, W), dtype bool.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            pts_tensor: Point cloud, shape (1, 1024, 3), dtype float32, in metres.
            choose_tensor: Index mapping from pts to flattened resized image, shape (1, 1024), dtype int32.
        """
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        depth = fill_missing(depth.astype(np.float32), DEPTH_NORM_SCALE, 1)
        pts_full = depth_to_full_pointcloud(
            depth, fx=fx, fy=fy, cx=cx, cy=cy, depth_scale=DEPTH_NORM_SCALE
        )

        full_valid = mask & (depth > 0)
        pts_arr, choose_mapped = sample_choose_and_pts(
            pts_full, full_valid, bbox, seed=0
        )

        pts_tensor = torch.from_numpy(pts_arr).float().unsqueeze(0)
        choose_tensor = torch.from_numpy(choose_mapped).int().unsqueeze(0)
        return pts_tensor, choose_tensor

    @staticmethod
    def draw_pose(
        image: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        size: np.ndarray,
        intrinsics: dict,
    ) -> np.ndarray:
        """
        Render 3D bounding box and axes onto the image.

        Parameters
        ----------
        image
            RGB image, shape (H, W, 3), dtype uint8.
        rotation
            Rotation matrix, shape (3, 3), dtype float32.
        translation
            Translation vector in metres, shape (3,), dtype float32.
        size
            Object size in metres (w, h, d), shape (3,), dtype float32.
        intrinsics
            Camera intrinsics dict with keys fx, fy, cx, cy.

        Returns
        -------
        np.ndarray
            Image with 3D bbox and axes drawn, shape (H, W, 3), dtype uint8.
        """
        K = build_camera_projection_matrix(
            intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
        )

        corners_cam = (rotation @ get_3d_bbox(size)).T + translation
        pts_2d = project_to_image(corners_cam, K).astype(int)

        # Remap our corner ordering to draw_3d_bbox expected ordering.
        # get_3d_bbox corners: 0(+,+,+) 1(+,+,-) 2(-,+,+) 3(-,+,-)
        #                       4(+,-,+) 5(+,-,-) 6(-,-,+) 7(-,-,-)
        # draw_3d_bbox top face: 0-1-2-3, bottom: 4-5-6-7, verticals: 0-4..3-7
        # Remap: our [0,1,5,4,2,3,7,6] → draw_3d_bbox [0,1,2,3,4,5,6,7]
        remap = [0, 1, 5, 4, 2, 3, 7, 6]
        pts_2d_remapped = pts_2d[remap]
        vis = draw_3d_bbox(
            image,
            pts_2d_remapped[None].astype(int),
            np.array([0]),
            {0: (242, 121, 112)},
            thickness=2,
        )

        axes = np.array(
            [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], dtype=np.float32
        )
        axes_cam = (rotation @ axes.T).T + translation
        axes_2d = project_to_image(axes_cam, K).astype(int)
        cv2.arrowedLine(vis, tuple(axes_2d[0]), tuple(axes_2d[1]), (255, 0, 0), 2)
        cv2.arrowedLine(vis, tuple(axes_2d[0]), tuple(axes_2d[2]), (0, 255, 0), 2)
        cv2.arrowedLine(vis, tuple(axes_2d[0]), tuple(axes_2d[3]), (0, 0, 255), 2)

        return vis

    def run_model_for_eval(
        self,
        model_input: Generator[AsyncOnDeviceResult] | tuple[torch.Tensor, ...],
        model_batch_size: int,
    ) -> CollectionModelEvalGenerator:
        inputs: list[torch.Tensor] = list(model_input)  # type: ignore[arg-type]
        # Dataset __getitem__ returns tensors in backbone forward() order:
        # (category_label, choose, fps1, fps2, fps3, fps4, pts, rgb)
        backbone_output = self.backbone(*inputs)

        yield backbone_output

        if self.uses_ondevice_model:
            pre_pose_feat, c = backbone_output.wait()
            # Pass split tensors — AsyncOnDeviceModel batches them into 1 Hub job.
            # model_batch_size=1 (hardcoded in evaluate()), so split(1) produces
            # a list of N tensors each (1,384,96) → N dataset entries → 1 Hub job.
            # Reduces rotation head Hub jobs from O(N) to 1 (same as detectron2/bevdet).
            rot_out = self.rotation_head(
                pre_pose_feat.split(model_batch_size, dim=0),
                c.split(model_batch_size, dim=0),
            )
            yield rot_out
            pred_rotation, pred_translation, pred_size = rot_out.wait()
            return pred_rotation, pred_translation, pred_size
        else:
            pre_pose_feat, c = backbone_output  # type: ignore[misc]
            rot_out = self.rotation_head(pre_pose_feat, c)
            pred_rotation, pred_translation, pred_size = rot_out  # type: ignore[misc]
            combined = cast(
                tuple[torch.Tensor, ...],
                (pred_rotation, pred_translation, pred_size),
            )
            yield combined
            return combined

    def predict(
        self,
        rgb_crop: np.ndarray,
        depth: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask: np.ndarray,
        category_id: int,
    ) -> dict:
        """
        Run end-to-end pose estimation on a single object instance.

        Parameters
        ----------
        rgb_crop
            Cropped RGB patch (bbox region), shape (rmax-rmin, cmax-cmin, 3), dtype uint8.
        depth
            Depth image in raw sensor units, shape (H, W), dtype uint16.
        bbox
            Bounding box (rmin, rmax, cmin, cmax) in pixel coordinates.
        mask
            Binary instance mask, shape (H, W), dtype bool.
        category_id
            0-indexed category ID (0=bottle, 1=bowl, 2=camera, 3=can, 4=laptop, 5=mug).

        Returns
        -------
        dict
            dict with rotation (3,3), translation (3,), size (3,).
        """
        rgb_tensor = preprocess_rgb(rgb_crop).unsqueeze(0)
        pts, choose = self.preprocess_depth(depth, bbox, mask)
        cat = torch.tensor([category_id], dtype=torch.int32)
        fps_indices = compute_fps_indices(pts)

        with torch.no_grad():
            backbone_output = self.backbone(
                cat,
                choose,
                fps_indices[0],
                fps_indices[1],
                fps_indices[2],
                fps_indices[3],
                pts,
                rgb_tensor,
            )
            if self.uses_ondevice_model:
                pre_pose_feat, c = backbone_output.wait()
            else:
                pre_pose_feat, c = backbone_output  # type: ignore[misc]

            rotation_output = self.rotation_head(pre_pose_feat, c)
            if self.uses_ondevice_model:
                pred_rotation, pred_translation, pred_size = rotation_output.wait()
            else:
                pred_rotation, pred_translation, pred_size = rotation_output  # type: ignore[misc]

        R = pred_rotation[0].detach().cpu().numpy()
        t = pred_translation[0].detach().cpu().numpy()
        s = pred_size[0].detach().cpu().numpy()
        return {
            "rotation": R,
            "translation": t,
            "size": s,
        }
