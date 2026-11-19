# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.sixd_repnet.model import SCORE_THRESHOLD
from qai_hub_models.models.templates.corevision.image_processing_3d import (
    rotation_matrix_to_euler,
)
from qai_hub_models.models.templates.imagenet_classifier.app import preprocess_image
from qai_hub_models.models.templates.retinaface.app import Detection, detect_faces


class SixDRepNetApp:
    """
    End-to-end two-stage head pose estimation application.

    Stage 1 — RetinaFaceDetector: detects face bounding boxes in the scene image.
    Stage 2 — PoseEstimator: estimates head pose (pitch, yaw, roll) per face crop.
    """

    def __init__(
        self,
        face_detector: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        pose_estimator: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Parameters
        ----------
        face_detector
            RetinaFaceDetector (or compatible callable) that takes an RGB
            (1, 3, H, W) float32 tensor and returns
            (bbox_regressions, classifications, landmark_regressions).
        pose_estimator
            SixDRepNet (or compatible callable) that takes a [1, 3, 224, 224]
            float tensor and returns a [1, 3, 3] rotation matrix tensor.
        """
        self.face_detector = face_detector
        self.pose_estimator = pose_estimator

    def _detect_faces(self, image: np.ndarray) -> list[Detection]:
        """
        Run face detection on an HWC uint8 RGB numpy image.

        Parameters
        ----------
        image
            RGB image as a numpy array of shape (H, W, 3), dtype uint8.

        Returns
        -------
        list[Detection]
            List of (box [x1, y1, x2, y2], landmarks [5, 2], score) in original
            image pixel coordinates, sorted by score descending.
        """
        return detect_faces(self.face_detector, image, score_threshold=SCORE_THRESHOLD)

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> Image.Image | np.ndarray | list[np.ndarray]:
        return self.predict_pose(*args, **kwargs)

    def predict_pose(
        self,
        image: np.ndarray | Image.Image,
        raw_output: bool = False,
    ) -> Image.Image | np.ndarray | list[np.ndarray]:
        """
        Detect faces and predict head pose for each.

        Parameters
        ----------
        image
            Full scene image as a PIL Image or RGB numpy array (H x W x 3, uint8).
        raw_output
            If True, return list of [pitch, yaw, roll] arrays (degrees).
            If False, return a PIL image with pose axes and bounding boxes drawn.

        Returns
        -------
        Image.Image | np.ndarray | list[np.ndarray]
            Annotated PIL image (``raw_output=False``), a single
            ``[pitch, yaw, roll]`` array in degrees (``raw_output=True``,
            one face), or a list of such arrays (``raw_output=True``,
            multiple faces).
        """
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        detections = self._detect_faces(image)

        # Fall back to full image if no face detected
        if not detections:
            h, w = image.shape[:2]
            detections = [
                (np.array([0, 0, w, h]), np.zeros((5, 2), dtype=np.float32), 1.0)
            ]

        all_angles = []
        vis = None
        h_img, w_img = image.shape[:2]

        for box, _landmarks, _score in detections:
            x_min, y_min, x_max, y_max = (int(v) for v in box)
            bw = x_max - x_min
            bh = y_max - y_min

            # 20% padding, matching the upstream 6DRepNet demo
            x_min_p = max(0, x_min - int(0.2 * bw))
            y_min_p = max(0, y_min - int(0.2 * bh))
            x_max_p = min(w_img, x_max + int(0.2 * bw))
            y_max_p = min(h_img, y_max + int(0.2 * bh))

            crop = image[y_min_p:y_max_p, x_min_p:x_max_p]
            # preprocess_image resizes to 224x224 and converts to a float [0,1] tensor.
            # normalize=False because PoseEstimator.forward applies ImageNet normalization internally.
            tensor = preprocess_image(Image.fromarray(crop), normalize=False)

            rot_mat = self.pose_estimator(tensor)

            euler = rotation_matrix_to_euler(rot_mat) * 180.0 / np.pi
            pitch = float(euler[0, 0])
            yaw = float(euler[0, 1])
            roll = float(euler[0, 2])
            all_angles.append(np.array([pitch, yaw, roll], dtype=np.float32))

            if not raw_output:
                if vis is None:
                    vis = image.copy()
                cx = x_min + bw / 2.0
                cy = y_min + bh / 2.0
                vis = draw_pose_axes(vis, yaw, pitch, roll, cx, cy, bw * 0.6)
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

        if raw_output:
            return all_angles[0] if len(all_angles) == 1 else all_angles

        return Image.fromarray(vis if vis is not None else image)


def draw_pose_axes(
    image: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    tdx: float,
    tdy: float,
    size: float = 100.0,
) -> np.ndarray:
    """
    Draw 3-axis head pose visualization on an RGB image.

    Parameters
    ----------
    image
        RGB image to draw on.
    yaw
        Yaw angle in degrees.
    pitch
        Pitch angle in degrees.
    roll
        Roll angle in degrees.
    tdx
        X coordinate of the axis origin (face centre).
    tdy
        Y coordinate of the axis origin (face centre).
    size
        Axis length in pixels.

    Returns
    -------
    np.ndarray
        Image with axes drawn (RGB: red=X, green=Y, blue=Z).
    """
    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180

    x1 = size * (np.cos(y) * np.cos(r)) + tdx
    y1 = size * (np.cos(p) * np.sin(r) + np.cos(r) * np.sin(p) * np.sin(y)) + tdy

    x2 = size * (-np.cos(y) * np.sin(r)) + tdx
    y2 = size * (np.cos(p) * np.cos(r) - np.sin(p) * np.sin(y) * np.sin(r)) + tdy

    x3 = size * (np.sin(y)) + tdx
    y3 = size * (-np.cos(y) * np.sin(p)) + tdy

    cv2.line(image, (int(tdx), int(tdy)), (int(x1), int(y1)), (255, 0, 0), 3)
    cv2.line(image, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(image, (int(tdx), int(tdy)), (int(x3), int(y3)), (0, 0, 255), 2)

    return image
