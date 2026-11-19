# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from math import ceil
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.models.templates.retinaface.model import (
    DEFAULT_DETECTOR_HEIGHT,
    DEFAULT_DETECTOR_WIDTH,
)

# RetinaFace MobileNet config (matches cfg_mnet in face_detection/alignment.py
# and in the vendored retinaface/config.py).
_CFG_MNET: dict[str, Any] = {
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
}

Detection = tuple[np.ndarray, np.ndarray, float]


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
    score_threshold: float,
    confidence_threshold: float = 0.02,
    nms_threshold: float = 0.4,
    top_k: int = 5000,
    keep_top_k: int = 750,
) -> list[Detection]:
    """
    Decode raw RetinaFace outputs into filtered face detections.

    Replicates the post_process() logic from face_detection/alignment.py.

    Parameters
    ----------
    bbox_regressions
        Shape (1, num_anchors, 4) -- raw box regression deltas.
    classifications
        Shape (1, num_anchors, 2) -- softmax class scores [background, face].
    landmark_regressions
        Shape (1, num_anchors, 10) -- raw landmark regression deltas.
    image_hw
        (height, width) of the tensor fed to the detector, in pixels.
    score_threshold
        Final minimum score threshold applied after NMS.
    confidence_threshold
        Minimum score to keep a detection before NMS.
    nms_threshold
        IoU threshold for NMS.
    top_k
        Keep at most this many detections before NMS.
    keep_top_k
        Keep at most this many detections after NMS.

    Returns
    -------
    list[Detection]
        List of (box [x1, y1, x2, y2], landmarks [5, 2], score) for each
        detected face passing score_threshold, sorted by score descending.
    """
    im_h, im_w = image_hw
    variances = _CFG_MNET["variance"]

    priors = _make_priors(image_hw).numpy()
    scale = np.array([im_w, im_h, im_w, im_h], dtype=np.float32)
    scale1 = np.tile(np.array([im_w, im_h], dtype=np.float32), 5)

    loc = bbox_regressions[0].detach().cpu().numpy()
    conf = classifications[0].detach().cpu().numpy()
    landms = landmark_regressions[0].detach().cpu().numpy()

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


def detect_faces(
    face_detector: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    image: np.ndarray,
    score_threshold: float,
    input_hw: tuple[int, int] = (DEFAULT_DETECTOR_HEIGHT, DEFAULT_DETECTOR_WIDTH),
    **postprocess_kwargs: Any,
) -> list[Detection]:
    """
    Run the detector on an HWC uint8 RGB numpy image.

    The image is resized to ``input_hw`` before being fed to the detector, since
    the exported graph has a fixed input shape. Detected box and landmark
    coordinates are scaled back to the original image dimensions.

    Parameters
    ----------
    face_detector
        RetinaFaceDetectorBase (or compatible callable) taking an RGB
        (1, 3, H, W) float32 tensor in [0, 1].
    image
        RGB image as a numpy array of shape (H, W, 3), dtype uint8.
    score_threshold
        Final minimum score threshold applied after NMS.
    input_hw
        (height, width) to resize to before running the detector.
    **postprocess_kwargs
        Forwarded to postprocess_detections.

    Returns
    -------
    list[Detection]
        List of (box [x1, y1, x2, y2], landmarks [5, 2], score) in original
        image pixel coordinates, sorted by score descending.
    """
    orig_h, orig_w = image.shape[:2]
    det_h, det_w = input_hw

    if (orig_h, orig_w) != (det_h, det_w):
        resized = np.array(
            Image.fromarray(image).resize((det_w, det_h), Image.BILINEAR)
        )
    else:
        resized = image

    img_tensor = numpy_image_to_torch(resized)  # (1, 3, H, W) float32 [0, 1]
    bbox_regressions, classifications, landmark_regressions = face_detector(img_tensor)
    detections = postprocess_detections(
        bbox_regressions,
        classifications,
        landmark_regressions,
        image_hw=(det_h, det_w),
        score_threshold=score_threshold,
        **postprocess_kwargs,
    )

    if (orig_h, orig_w) == (det_h, det_w):
        return detections

    sx = orig_w / det_w
    sy = orig_h / det_h
    return [
        (
            box * np.array([sx, sy, sx, sy], dtype=np.float32),
            lm * np.array([sx, sy], dtype=np.float32),
            score,
        )
        for box, lm, score in detections
    ]
