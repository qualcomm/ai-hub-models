# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Repo-owned replacements for the top-down crop that mmpose performs inside
MMPoseInferencer.preprocess(). Keeping this here means a top-down pose recipe
can consume boxes from a compiled detector component instead of running an
uncompiled detector in the app layer.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
    pre_process_with_affine,
)

# mmpose pads the detector box before cropping. This is the default of
# mmpose.datasets.transforms.GetBBoxCenterScale and is what every top-down
# config in mmpose 1.x uses.
BBOX_PADDING = 1.25

# COCO class index for "person". Matches det_cat_ids=(0,) on mmpose's inferencer.
PERSON_CLASS_ID = 0

# RTMDet's mmdet test pipeline is Resize(keep_ratio=True) + Pad(pad_val=114),
# i.e. a letterbox, not a stretch. Getting this wrong shifts the boxes enough to
# move keypoints by tens of pixels.
LETTERBOX_PAD_VALUE = 114


def letterbox(
    image: np.ndarray, target_hw: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """
    Aspect-preserving resize plus bottom-right pad, as mmdet does for RTMDet.

    Parameters
    ----------
    image
        RGB image, shape (H, W, 3), dtype uint8.
    target_hw
        Target (height, width).

    Returns
    -------
    padded : np.ndarray
        Letterboxed image of shape target_hw, dtype uint8.
    ratio : float
        Scale factor applied; divide detected coordinates by this to map back.
    """
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = target_hw
    ratio = min(dst_h / src_h, dst_w / src_w)
    new_w = int(src_w * ratio + 0.5)
    new_h = int(src_h * ratio + 0.5)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((dst_h, dst_w, 3), LETTERBOX_PAD_VALUE, dtype=image.dtype)
    padded[:new_h, :new_w] = resized
    return padded, ratio


def box_xyxy_to_cs(
    bbox: np.ndarray,
    aspect_ratio: float,
    padding: float = BBOX_PADDING,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an xyxy box to mmpose's (center, scale) crop parameters.

    Equivalent to mmpose's bbox_xyxy2cs() followed by
    TopdownAffine._fix_aspect_ratio(). mmpose applies padding before fixing the
    aspect ratio; because the aspect-ratio fix is scale-invariant the two orders
    agree, so padding is applied once at the end here.

    Parameters
    ----------
    bbox
        Box as [x1, y1, x2, y2].
    aspect_ratio
        Target width / height of the model input.
    padding
        Box padding factor applied to the scale.

    Returns
    -------
    center : np.ndarray
        Box center (x, y), shape (2,).
    scale : np.ndarray
        Box scale (w, h), shape (2,), aspect-corrected and padded.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)

    w = x2 - x1
    h = y2 - y1
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32) * padding
    return center, scale


def crop_to_center_scale(
    image: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    input_hw: tuple[int, int],
) -> torch.Tensor:
    """
    Affine-crop an image to a model input using mmpose's warp convention.

    Parameters
    ----------
    image
        RGB image, shape (H, W, 3), dtype uint8.
    center
        Crop center (x, y), shape (2,).
    scale
        Crop scale (w, h), shape (2,).
    input_hw
        Model input (height, width).

    Returns
    -------
    torch.Tensor
        Cropped image, shape (1, 3, H, W), float32 in [0, 1].
    """
    # compute_affine_transform (used by pre_process_with_affine) is identical to
    # mmpose's get_warp_matrix, including the dst_w-based direction vector.
    return pre_process_with_affine(image, center, scale, 0, input_hw)


def select_person_box(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_idx: torch.Tensor,
    score_threshold: float = 0.3,
) -> np.ndarray | None:
    """
    Pick the highest-scoring person box from a detector's decoded output.

    Parameters
    ----------
    boxes
        Decoded boxes, shape (1, num_preds, 4) as (x1, y1, x2, y2).
    scores
        Per-box confidence, shape (1, num_preds).
    class_idx
        Per-box class index, shape (1, num_preds).
    score_threshold
        Minimum confidence to accept a detection.

    Returns
    -------
    np.ndarray | None
        The selected box as [x1, y1, x2, y2], or None if no person was found.
    """
    boxes_np = boxes[0].detach().cpu().numpy()
    scores_np = scores[0].detach().cpu().numpy()
    class_np = class_idx[0].detach().cpu().numpy()

    keep = (class_np == PERSON_CLASS_ID) & (scores_np >= score_threshold)
    if not keep.any():
        return None
    candidates = boxes_np[keep]
    return candidates[int(np.argmax(scores_np[keep]))]


def detect_and_crop(
    person_detector: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    image: np.ndarray,
    input_hw: tuple[int, int],
    detector_hw: tuple[int, int],
    score_threshold: float = 0.3,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Run the person detector, then crop the image to the best person box.

    Falls back to the whole frame when no person is detected, matching what
    mmpose's inferencer does (pose2d_inferencer.py uses the image bounds).

    Parameters
    ----------
    person_detector
        Callable taking an RGB (1, 3, H, W) float32 tensor in [0, 1] and
        returning (boxes, scores, class_idx).
    image
        RGB image, shape (H, W, 3), dtype uint8.
    input_hw
        Pose model input (height, width).
    detector_hw
        Detector input (height, width); the image is resized to this.
    score_threshold
        Minimum detector confidence.

    Returns
    -------
    cropped : torch.Tensor
        Pose model input, shape (1, 3, H, W), float32 in [0, 1].
    center : np.ndarray
        Crop center used, in original image coordinates.
    scale : np.ndarray
        Crop scale used, in original image coordinates.
    """
    orig_h, orig_w = image.shape[:2]

    padded, ratio = letterbox(image, detector_hw)
    boxes, scores, class_idx = person_detector(numpy_image_to_torch(padded))
    box = select_person_box(boxes, scores, class_idx, score_threshold)

    if box is None:
        box = np.array([0, 0, orig_w, orig_h], dtype=np.float32)
    else:
        # Letterbox pads bottom-right only, so undoing it is a pure divide.
        box = box / ratio

    aspect_ratio = input_hw[1] / input_hw[0]
    center, scale = box_xyxy_to_cs(box, aspect_ratio)
    return crop_to_center_scale(image, center, scale, input_hw), center, scale
