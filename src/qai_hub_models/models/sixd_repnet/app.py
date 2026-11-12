# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from math import ceil
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.sixd_repnet.model import SCORE_THRESHOLD
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.models.templates.corevision.image_processing_3d import (
    rotation_matrix_to_euler,
)
from qai_hub_models.models.templates.imagenet_classifier.app import preprocess_image

# RetinaFace MobileNet config (matches cfg_mnet in face_detection/alignment.py)
_CFG_MNET: dict[str, Any] = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}


def _make_priors(image_size: tuple[int, int]) -> torch.Tensor:
    """
    Generate anchor prior boxes for the given image size.

    Parameters
    ----------
    image_size
        (height, width) of the image in pixels.

    Returns
    -------
    torch.Tensor
        Prior boxes of shape (num_anchors, 4) in (cx, cy, w, h) normalised format.
    """
    cfg = _CFG_MNET
    feature_maps = [
        [ceil(image_size[0] / step), ceil(image_size[1] / step)]
        for step in cfg["steps"]
    ]
    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes = cfg["min_sizes"][k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                cx = (j + 0.5) * cfg["steps"][k] / image_size[1]
                cy = (i + 0.5) * cfg["steps"][k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    priors = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
    if cfg["clip"]:
        priors.clamp_(max=1, min=0)
    return priors


def _decode_boxes(
    loc: np.ndarray, priors: np.ndarray, variances: list[float]
) -> np.ndarray:
    """
    Decode box regression deltas into (x1, y1, x2, y2) pixel coordinates.

    Parameters
    ----------
    loc
        Box regression deltas of shape (num_anchors, 4).
    priors
        Prior anchor boxes of shape (num_anchors, 4) in (cx, cy, w, h) format.
    variances
        Two-element list [var_xy, var_wh] used to scale the regression deltas.

    Returns
    -------
    np.ndarray
        Decoded boxes of shape (num_anchors, 4) in (x1, y1, x2, y2) format.
    """
    boxes = np.concatenate(
        [
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
        ],
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def postprocess_detections(
    bbox_regressions: torch.Tensor,
    classifications: torch.Tensor,
    landmark_regressions: torch.Tensor,
    image_hw: tuple[int, int],
    confidence_threshold: float = 0.02,
    nms_threshold: float = 0.4,
    top_k: int = 5000,
    keep_top_k: int = 750,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Decode raw RetinaFace outputs into filtered face detections.

    Replicates the post_process() logic from face_detection/alignment.py.

    Parameters
    ----------
    bbox_regressions
        Shape (1, num_anchors, 4) — raw box regression deltas.
    classifications
        Shape (1, num_anchors, 2) — softmax class scores [background, face].
    landmark_regressions
        Shape (1, num_anchors, 10) — raw landmark regression deltas.
    image_hw
        (height, width) of the input image in pixels.
    confidence_threshold
        Minimum score to keep a detection before NMS.
    nms_threshold
        IoU threshold for NMS.
    top_k
        Keep at most this many detections before NMS.
    keep_top_k
        Keep at most this many detections after NMS.
    score_threshold
        Final minimum score threshold applied after NMS.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, float]]
        List of (box [x1, y1, x2, y2], landmarks [5, 2], score) for each
        detected face passing score_threshold.
    """
    im_h, im_w = image_hw
    cfg = _CFG_MNET
    variances = cfg["variance"]

    priors = _make_priors(image_hw).numpy()
    scale = np.array([im_w, im_h, im_w, im_h], dtype=np.float32)
    scale1 = np.tile(np.array([im_w, im_h], dtype=np.float32), 5)

    loc = bbox_regressions[0].cpu().numpy()
    conf = classifications[0].cpu().numpy()
    landms = landmark_regressions[0].cpu().numpy()

    boxes_np = _decode_boxes(loc, priors, variances) * scale
    scores_np = conf[:, 1]
    landmarks_np = (
        np.concatenate(
            [
                priors[:, :2] + landms[:, 0:2] * variances[0] * priors[:, 2:],
                priors[:, :2] + landms[:, 2:4] * variances[0] * priors[:, 2:],
                priors[:, :2] + landms[:, 4:6] * variances[0] * priors[:, 2:],
                priors[:, :2] + landms[:, 6:8] * variances[0] * priors[:, 2:],
                priors[:, :2] + landms[:, 8:10] * variances[0] * priors[:, 2:],
            ],
            axis=1,
        )
        * scale1
    )

    # Keep top-K by confidence before NMS
    order = scores_np.argsort()[::-1][:top_k]
    boxes_np = boxes_np[order]
    scores_np = scores_np[order]
    landmarks_np = landmarks_np[order]

    # Use shared batched_nms utility; pass landmarks as an extra arg so they
    # stay in sync with the selected boxes.
    boxes_t = torch.from_numpy(boxes_np).unsqueeze(0)  # (1, N, 4)
    scores_t = torch.from_numpy(scores_np).unsqueeze(0)  # (1, N)
    landmarks_t = torch.from_numpy(landmarks_np).unsqueeze(0)  # (1, N, 10)
    nms_result = batched_nms(
        nms_threshold, confidence_threshold, boxes_t, scores_t, None, landmarks_t
    )
    boxes_sel = nms_result[0][0].numpy()
    scores_sel = nms_result[1][0].numpy()
    landmarks_sel = nms_result[2][0].numpy()

    # Keep top-K after NMS, then apply final score threshold
    boxes_sel = boxes_sel[:keep_top_k]
    scores_sel = scores_sel[:keep_top_k]
    landmarks_sel = landmarks_sel[:keep_top_k]

    results = []
    for box, lm, score in zip(boxes_sel, landmarks_sel, scores_sel, strict=False):
        if float(score) < score_threshold:
            break
        results.append((box, lm.reshape(5, 2), float(score)))
    return results


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

    def _detect_faces(
        self, image: np.ndarray, input_hw: tuple[int, int] = (640, 640)
    ) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """
        Run face detection on an HWC uint8 RGB numpy image.

        The image is resized to ``input_hw`` before being fed to the detector.
        Detected box and landmark coordinates are scaled back to the original
        image dimensions before being returned.

        Parameters
        ----------
        image
            RGB image as a numpy array of shape (H, W, 3), dtype uint8.
        input_hw
            (height, width) to resize the image to before running the detector.
            Defaults to (640, 640) matching the RetinaFace MobileNet training resolution.

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray, float]]
            List of (box [x1, y1, x2, y2], landmarks [5, 2], score) in original
            image pixel coordinates, sorted by score descending.
        """
        orig_h, orig_w = image.shape[:2]
        det_h, det_w = input_hw

        # Resize to detector input size
        if (orig_h, orig_w) != (det_h, det_w):
            resized = np.array(
                Image.fromarray(image).resize((det_w, det_h), Image.BILINEAR)
            )
        else:
            resized = image

        img_tensor = numpy_image_to_torch(resized)  # (1, 3, H, W) float32 [0, 1]
        bbox_regressions, classifications, landmark_regressions = self.face_detector(
            img_tensor
        )
        detections = postprocess_detections(
            bbox_regressions,
            classifications,
            landmark_regressions,
            image_hw=(det_h, det_w),
        )

        # Scale boxes and landmarks back to original image size
        if (orig_h, orig_w) != (det_h, det_w):
            sx = orig_w / det_w
            sy = orig_h / det_h
            scaled = []
            for box, lm, score in detections:
                box_s = box * np.array([sx, sy, sx, sy], dtype=np.float32)
                lm_s = lm * np.array([sx, sy], dtype=np.float32)
                scaled.append((box_s, lm_s, score))
            return scaled
        return detections

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
