# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    transform_to_matrix,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)

CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


class CVT_GKTApp:
    """Lightweight application wrapper for CVT and GKT."""

    def __init__(
        self,
        ckpt_name: str,
        target_height: int = 224,
        target_width: int = 480,
    ) -> None:
        self.ckpt_name = ckpt_name
        ckpt_type = "road" if "road" in ckpt_name else "vehicles"
        self.color = [196, 199, 192] if ckpt_type == "road" else [52, 101, 154]

        self.target_height = target_height
        self.target_width = target_width

    def preprocess_images(
        self,
        images: list[Image.Image],
        cam_metadata: dict[str, dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert 6-camera PIL images to normalized tensors and compute inverse camera matrices for BEV transformation.
        Source:https://github.com/bradyz/cross_view_transformers/blob/master/cross_view_transformer/data/transforms.py#L118C9-L118C20

        Parameters
        ----------
        images
            List of 6 PIL images in RGB format, ordered as: CAM_FRONT_LEFT, CAM_FRONT,
            CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT.
        cam_metadata
            Dictionary mapping camera names to camera parameter dictionaries.
            Each camera dictionary contains:

            intrins
                Camera intrinsics matrix, shape (3, 3).
            sensor2ego_translation
                Translation vector [x, y, z] in meters from sensor to ego frame,
                shape (3,).
            sensor2ego_rotation
                Quaternion [w, x, y, z] for rotation from sensor to ego frame,
                shape (4,).
            ego2global_translation
                Translation vector [x, y, z] in meters from ego to global frame,
                shape (3,).
            ego2global_rotation
                Quaternion [w, x, y, z] for rotation from ego to global frame,
                shape (4,).

        Returns
        -------
        images_tensor : torch.Tensor
            Pre-processed image tensor, shape [1, 6, 3, H, W].
        intrinsics_tensor : torch.Tensor
            Intrinsics tensor mapping 2D pixel coordinates to 3D camera-space rays,
            shape [1, 6, 3, 3].
        extrinsics_tensor : torch.Tensor
            Extrinsics tensor mapping world coordinates to camera coordinates,
            shape [1, 6, 4, 4].
        inv_intrinsics_tensor : torch.Tensor
            Inverse intrinsics tensor mapping 2D pixel coordinates to 3D camera-space
            rays, shape [1, 6, 3, 3].
        inv_extrinsics_tensor : torch.Tensor
            Inverse extrinsics tensor mapping world coordinates to camera coordinates,
            shape [1, 6, 4, 4].
        """
        intrins_list = [
            np.array(cam_metadata[cam]["intrins"], dtype=np.float32)
            for cam in CAM_NAMES
        ]
        sensor2egos_list = [
            transform_to_matrix(
                cam_metadata[cam]["sensor2ego_translation"],
                cam_metadata[cam]["sensor2ego_rotation"],
                inv=True,
            )
            for cam in CAM_NAMES
        ]
        ego2globals_list = [
            transform_to_matrix(
                cam_metadata[cam]["ego2global_translation"],
                cam_metadata[cam]["ego2global_rotation"],
                inv=True,
            )
            for cam in CAM_NAMES
        ]
        world_from_egoidarflat_np = transform_to_matrix(
            cam_metadata["CAM_FRONT_LEFT"]["ego2global_translation"],
            cam_metadata["CAM_FRONT_LEFT"]["ego2global_rotation"],
            flat=True,
        )

        sensor2egos = torch.tensor(np.array(sensor2egos_list), dtype=torch.float32)
        ego2globals = torch.tensor(np.array(ego2globals_list), dtype=torch.float32)
        world_from_egoidarflat = torch.tensor(
            world_from_egoidarflat_np, dtype=torch.float32
        )

        extrinsics = sensor2egos @ ego2globals @ world_from_egoidarflat

        images_tensor_list = []
        intrinsics_list = []

        h, w = self.target_height, self.target_width

        for i, image in enumerate(images):
            # Ensure image is RGB
            image = image.convert("RGB")
            original_w, original_h = image.size

            scale_x = w / original_w
            h_resize = int(original_h * scale_x)
            # Calculate top_crop to match target height
            top_crop = h_resize - h
            w_resize = w

            image_resized = image.resize((w_resize, h_resize))
            image_cropped = image_resized.crop((0, top_crop, w_resize, h_resize))

            _, img_tensor = app_to_net_image_inputs(image_cropped)
            images_tensor_list.append(img_tensor.squeeze(0))

            intrinsics = np.array(intrins_list[i], dtype=np.float32)

            # Scale intrinsics for resize
            scale_y = h_resize / original_h
            intrinsics[0, 0] *= scale_x  # fx
            intrinsics[0, 2] *= scale_x  # cx
            intrinsics[1, 1] *= scale_y  # fy
            intrinsics[1, 2] *= scale_y  # cy
            intrinsics[1, 2] -= top_crop  # cy adjustment for crop

            intrinsics_list.append(torch.tensor(intrinsics))

        # Stack tensors
        images_tensor = torch.stack(images_tensor_list).unsqueeze(0)
        intrinsics_tensor = torch.stack(intrinsics_list).unsqueeze(0)
        extrinsics_tensor = extrinsics.unsqueeze(0)
        inv_intrinsics_tensor = torch.inverse(intrinsics_tensor)
        inv_extrinsics_tensor = torch.inverse(extrinsics_tensor)
        return (
            images_tensor,
            intrinsics_tensor,
            extrinsics_tensor,
            inv_intrinsics_tensor,
            inv_extrinsics_tensor,
        )

    def smooth(
        self, x: np.ndarray, t1: float = 0.6, c: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Smooth heatmap and apply color threshold.

        Parameters
        ----------
        x
            Input heatmap array.
        t1
            Threshold for heatmap. Default is 0.6.
        c
            Color for heatmap visualization as [R, G, B]. If None, uses default color.

        Returns
        -------
        x_viz : np.ndarray
            Visualized heatmap with applied color.
        opacity : np.ndarray
            Opacity mask for the heatmap.
        """
        c = self.color if c is None else c
        c_array = np.array(c, dtype=np.float32)[None, None]
        m1 = x > t1
        x_viz = 255 * np.ones((*x.shape, 3), dtype=np.float32)
        x_viz[m1] = c_array
        opacity = m1.astype(np.float32)
        return x_viz, opacity

    @staticmethod
    def get_view_matrix(
        h: int = 200,
        w: int = 200,
        h_meters: float = 100.0,
        w_meters: float = 100.0,
        offset: float = 0.0,
    ) -> list[list[float]]:
        """
        Generate view matrix for BEV visualization.

        Parameters
        ----------
        h
            Height of the output image. Default is 200.
        w
            Width of the output image. Default is 200.
        h_meters
            Height in meters for the BEV map. Default is 100.0.
        w_meters
            Width in meters for the BEV map. Default is 100.0.
        offset
            Offset for the BEV map. Default is 0.0.

        Returns
        -------
        view_matrix : list[list[float]]
            3x3 view matrix for transforming coordinates.
        """
        sh = h / h_meters
        sw = w / w_meters

        return [[0.0, -sw, w / 2.0], [-sh, 0.0, h * offset + h / 2.0], [0.0, 0.0, 1.0]]

    def postprocess_bev_to_image(self, bev: torch.Tensor) -> list[Image.Image]:
        """
        Convert BEV heatmap to map-view PIL images for each camera view.

        Parameters
        ----------
        bev
            BEV tensor with predictions, shape [B, 1, H, W].

        Returns
        -------
        images : list[Image.Image]
            List of PIL Images (RGB) with heatmap and ego polygon.
        """
        bev = bev.detach().cpu()
        view = self.get_view_matrix()

        images = []
        batch_size = bev.shape[0]
        b_max = 8

        for b in range(min(batch_size, b_max)):
            pred = bev[b].sigmoid().numpy().transpose(1, 2, 0).squeeze()
            viz, opacity = self.smooth(pred, t1=0.4, c=self.color)

            canvas_base = opacity[..., None] * viz + (1 - opacity[..., None]) * viz
            canvas_base = np.uint8(canvas_base)

            points = np.array(
                [
                    [-4.0 / 2 + 0.3, -1.73 / 2, 1],
                    [-4.0 / 2 + 0.3, 1.73 / 2, 1],
                    [4.0 / 2 + 0.3, 1.73 / 2, 1],
                    [4.0 / 2 + 0.3, -1.73 / 2, 1],
                ]
            )

            points_transformed = view @ points.T
            points_transformed = points_transformed[:2].T.astype(np.int32)
            points_transformed = points_transformed.reshape((-1, 1, 2))
            cv2.fillPoly(canvas_base, [points_transformed], color=(164, 0, 0))  # type: ignore[call-overload]
            images.append(Image.fromarray(canvas_base, mode="RGB"))

        return images
