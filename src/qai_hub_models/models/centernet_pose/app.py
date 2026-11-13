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

from qai_hub_models.models.templates.corevision.draw import (
    draw_box_from_xyxy,
    draw_connections,
    draw_points,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
    pre_process_with_affine,
)


class CenterNetPoseApp:
    """
    This class is required to perform end to end inference for CenterNetPose Model

    For a given images input, the app will:
        * pre-process the inputs (convert to range[0, 1])
        * Run the inference
        * Convert the hm, wh, hps, reg, hm_hp, hm_offset into 2d_bbox and keypoints
        * Draw 2d_bbox and keypoints in the image.
    """

    def __init__(
        self,
        model: Callable[
            [
                torch.Tensor,
            ],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
        height: int = 512,
        width: int = 512,
        max_dets: int = 100,
    ) -> None:
        """
        Initialize CenterNetPoseApp.

        Parameters
        ----------
        model
            CenterNetPose Model.
        decode
            Function to decode the raw model outputs
            into detected objects/detections and keypoints.
        height
            Input height for the model.
        width
            Input width for the model.
        max_dets
            Maximum number of detections per image.
        """
        self.model = model
        self.decode = decode
        self.height = height
        self.width = width
        self.max_dets = max_dets
        self.vis_threshold = 0.3
        self.edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (3, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | Image.Image:
        # See predict_pose_from_image.
        return self.predict_pose_from_image(*args, **kwargs)

    def predict_pose_from_image(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """
        Run the CenterNetPose model and predict 2D bounding boxes with keypoints.

        Parameters
        ----------
        image
            PIL image in RGB format.
        raw_output
            If True, returns raw detections array. If False, returns annotated image.

        Returns
        -------
        result : np.ndarray | Image.Image
            If raw_output is True:
                dets array with shape (max_det, 40).
            Otherwise:
                PIL image with 2D bounding boxes and keypoints drawn.
        """
        image_array = np.array(image)
        height, width = image_array.shape[0:2]
        c = np.array([width / 2, height / 2], dtype=np.float32)
        s = np.array([max(height, width), max(height, width)], dtype=np.float32)

        image_tensor = pre_process_with_affine(
            image_array, c, s, 0, (self.height, self.width)
        )

        # model supports only single batch
        assert image_tensor.shape[0] == 1
        hm, wh, hps, reg, hm_hp, hm_offset = self.model(image_tensor)

        dets = self.decode(hm, wh, hps, reg, hm_hp, hm_offset, self.max_dets).numpy()
        dets = dets.reshape(-1, dets.shape[2])

        if raw_output:
            return dets

        bboxes = denormalize_coordinates_affine(
            dets[:, :4].reshape(-1, 2), c, s, 0, (hm.shape[2], hm.shape[3])
        )
        kps = denormalize_coordinates_affine(
            dets[:, 5:39].reshape(-1, 2), c, s, 0, (hm.shape[2], hm.shape[3])
        )
        scores = dets[:, 4]
        bboxes = bboxes.astype(int).reshape(-1, 4)
        kps = kps.reshape(self.max_dets, -1, 2)

        for bbox, kp, score in zip(bboxes, kps, scores, strict=False):
            if score > self.vis_threshold:
                draw_box_from_xyxy(
                    image_array,
                    bbox[0:2],
                    bbox[2:4],
                    color=(0, 0, 255),
                    size=2,
                    text="person",
                )

                draw_connections(image_array, kp, self.edges, (255, 0, 0), 2)
                draw_points(image_array, kp, (255, 255, 0), 5)

        return Image.fromarray(image_array)
