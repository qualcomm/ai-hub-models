# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms as tv_batched_nms
from torchvision.ops import nms

from qai_hub_models.extern.numba import njit, prange


def batched_nms(
    iou_threshold: float,
    score_threshold: float | None,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_indices: torch.Tensor | None = None,
    *gather_additional_args: torch.Tensor,
) -> tuple[list[torch.Tensor], ...]:
    """
    Non maximum suppression over several batches.

    Parameters
    ----------
    iou_threshold
        Intersection over union (IoU) threshold.
    score_threshold
        Score threshold (throw away any boxes with scores under this threshold).
    boxes
        Boxes to run NMS on. Shape is [B, N, 4], B == batch, N == num boxes,
        and 4 == (x1, x2, y1, y2).
    scores
        Scores for each box. Shape is [B, N], range is [0:1].
    class_indices
        Class for each box. Shape is [B, N].
        If set, NMS is applied per-class rather than globally.
    *gather_additional_args
        Additional tensor(s) to be gathered in the same way as boxes and scores.
        In other words, each arg is returned with only the elements for the boxes
        selected by NMS. Should be shape [B, N, ...].

    Returns
    -------
    tuple[list[torch.Tensor], ...]
        boxes_out
            Output boxes. This is list of tensors--one tensor per batch.
            Each tensor is shape [S, 4], where S == number of selected boxes,
            and 4 == (x1, x2, y1, y2).
        scores_out
            Output scores. This is list of tensors--one tensor per batch.
            Each tensor is shape [S], where S == number of selected boxes.
        class_indices_out
            Output classes. This is list of tensors--one tensor per batch.
            Each tensor is shape [S], where S == number of selected boxes.
            Only returned when class_indices is provided; omitted from return tuple otherwise.
        *args_out
            Filtered versions of gather_additional_args, containing only elements
            corresponding to boxes selected by NMS. Each is a list of tensors
            (one per batch). Only returned when gather_additional_args is provided.
    """
    scores_out: list[torch.Tensor] = []
    boxes_out: list[torch.Tensor] = []
    class_indices_out: list[torch.Tensor] = []
    args_out: list[list[torch.Tensor]] = (
        [[] for _ in gather_additional_args] if gather_additional_args else []
    )

    for batch_idx in range(boxes.shape[0]):
        # Index to current batch.
        batch_scores = scores[batch_idx]
        batch_boxes = boxes[batch_idx]
        batch_args = [arg[batch_idx] for arg in gather_additional_args or []]
        batch_class_indices = (
            class_indices[batch_idx] if class_indices is not None else None
        )

        # Clip outputs to valid scores
        if score_threshold is not None:
            scores_idx = torch.nonzero(scores[batch_idx] >= score_threshold).squeeze(-1)
            batch_scores = batch_scores[scores_idx]
            batch_boxes = batch_boxes[scores_idx]
            batch_class_indices = (
                batch_class_indices[scores_idx]
                if batch_class_indices is not None
                else None
            )
            batch_args = [arg[scores_idx] for arg in batch_args or []]

        if len(batch_scores > 0):
            # Apply NMS
            if batch_class_indices is not None:
                # class dependent
                nms_indices = tv_batched_nms(
                    batch_boxes[..., :4],
                    batch_scores,
                    batch_class_indices,
                    iou_threshold,
                )
            else:
                # class agnostic
                nms_indices = nms(batch_boxes[..., :4], batch_scores, iou_threshold)

            # Apply NMS indices
            batch_boxes = batch_boxes[nms_indices]
            batch_scores = batch_scores[nms_indices]
            batch_class_indices = (
                batch_class_indices[nms_indices]
                if batch_class_indices is not None
                else None
            )
            batch_args = [arg[nms_indices] for arg in batch_args]

        # Append to outputs
        boxes_out.append(batch_boxes)
        scores_out.append(batch_scores)
        if batch_class_indices is not None:
            class_indices_out.append(batch_class_indices)
        for arg_idx, arg in enumerate(batch_args):
            args_out[arg_idx].append(arg)

    if class_indices is None:
        return boxes_out, scores_out, *args_out
    return boxes_out, scores_out, class_indices_out, *args_out


def compute_box_corners_with_rotation(
    xc: torch.Tensor,
    yc: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    theta: torch.Tensor,
) -> torch.Tensor:
    """
    From the provided information, compute the (x, y) coordinates of the box's corners.

    Parameters
    ----------
    xc
        Center of box (x). Shape is [Batch].
    yc
        Center of box (y). Shape is [Batch].
    w
        Width of box. Shape is [Batch].
    h
        Height of box. Shape is [Batch].
    theta
        Rotation of box (in radians). Shape is [Batch].

    Returns
    -------
    corners : torch.Tensor
        Computed corners. Shape is (B x 4 x 2), where 2 == (x, y).
    """
    batch_size = xc.shape[0]

    # Construct unit square
    points = torch.tensor([[-1, -1, 1, 1], [-1, 1, -1, 1]], dtype=torch.float32).repeat(
        batch_size, 1, 1
    )  # Construct Unit Square. Shape [B, 2, 4], where 2 == (X, Y)
    points *= torch.stack((w / 2, h / 2), dim=-1).unsqueeze(
        dim=2
    )  # Scale unit square to appropriate height and width

    # Rotate unit square to new coordinate system
    R = torch.stack(
        (
            torch.stack((torch.cos(theta), -torch.sin(theta)), dim=1),
            torch.stack((torch.sin(theta), torch.cos(theta)), dim=1),
        ),
        dim=1,
    )  # Construct rotation matrix
    points = R @ points  # Apply Rotation

    # Adjust box to center around the original center
    points = points + torch.stack((xc, yc), dim=1).unsqueeze(dim=2)

    return points.transpose(-1, -2)


def compute_box_affine_crop_resize_matrix(
    box_corners: torch.Tensor, output_image_size: tuple[int, int]
) -> list[np.ndarray]:
    """
    Computes the affine transform matrices required to crop, rescale,
    and pad the box described by box_corners to fit into an image of the given size without warping.

    Parameters
    ----------
    box_corners
        Bounding box corners. These coordinates will be mapped to the output image.
        Shape is [B, 3, 2], where B = batch, 3 = (top left point, bottom left point,
        top right point), and 2 = (x, y).
    output_image_size
        Size of image to which the box should be resized and cropped. Layout is (W, H).

    Returns
    -------
    affines : list[np.ndarray]
        Computed affine transform matrices. Shape is (2 x 3).
    """
    # Define coordinates for translated image
    network_input_points = np.array(
        [[0, 0], [0, output_image_size[1] - 1], [output_image_size[0] - 1, 0]],
        dtype=np.float32,
    )

    # Compute affine transformation that will map the square to the point
    affines: list[np.ndarray] = []
    for batch in range(box_corners.shape[0]):
        src = box_corners[batch][..., :3].detach().numpy()
        affines.append(cv2.getAffineTransform(src, network_input_points))
    return affines


def box_xywh_to_xyxy(box_cwh: torch.Tensor) -> torch.Tensor:
    """
    Convert center, W, H to top left / bottom right bounding box values.

    Parameters
    ----------
    box_cwh
        Bounding box.

        Shape is either...
        [..., 4]
        Box layout is [xc, yc, w, h]

        [..., 2, 2]
        Box layout is [[xc, yc], [w, h]]

    Returns
    -------
    box_xyxy: torch.Tensor
        if input shape is [..., 4]:
            Output shape is [..., 4]
            Output box layout is [x0, y0, x1, y1]
        if input shape is [..., 2, 2]:
            Output shape is [..., 2, 2]
            Output box layout is [[x0, y0], [x1, y1]]
    """
    if box_cwh.shape[-1] == 4:
        # box_cwh is [..., 4] with [cx, cy, w, h]
        # Output is [..., 4] with [x0, y0, x1, y1]
        center = box_cwh[..., :2]  # [cx, cy]
        wh_half = box_cwh[..., 2:] * 0.5  # [w/2, h/2]
        return torch.cat((center - wh_half, center + wh_half), dim=-1)

    center = box_cwh[..., 0, :]  # [cx, cy]
    wh_half = box_cwh[..., 1, :] * 0.5  # [w/2, h/2]
    return torch.stack((center - wh_half, center + wh_half), dim=-2)


def box_xyxy_to_xywh(box_xy: torch.Tensor) -> torch.Tensor:
    """
    Converts box coordinates to center / width / height notation.

    Parameters
    ----------
    box_xy
        Bounding box.

        Shape is either...
        [..., 4]
        Box layout is [x0, y0, x1, y1]

        [..., 2, 2]
        Box layout is [[x0, y0], [x1, y1]]

    Returns
    -------
    box_cwh: torch.Tensor
        if input shape is [..., 4]:
            Output shape is [..., 4]
            Output box layout is [xc, yc, w, h]
        if input shape is [..., 2, 2]:
            Output shape is [..., 2, 2]
            Output box layout is [[xc, yc], [w, h]]
    """
    if box_xy.shape[-1] == 4:
        # Input is [..., 4] with [x0, y0, x1, y1]
        # Output is [..., 4] with [xc, yc, w, h]
        xy1 = box_xy[..., :2]
        xy2 = box_xy[..., 2:]
        center = (xy1 + xy2) * 0.5
        wh = xy2 - xy1
        return torch.cat((center, wh), dim=-1)

    # Input is [..., 2, 2] with [[x0, y0], [x1, y1]]
    # Output is [..., 2, 2] with [[xc, yc], [w, h]]
    xy1 = box_xy[..., 0, :]
    xy2 = box_xy[..., 1, :]
    center = (xy1 + xy2) * 0.5
    wh = xy2 - xy1
    return torch.stack((center, wh), dim=-2)


def box_xywh_to_cs(
    box_cwh: list, aspect_ratio: float, padding_factor: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert bbox to center-scale format while maintaining aspect ratio.

    Parameters
    ----------
    box_cwh
        Bounding box. Shape is [4,] with layout [xc, yc, w, h].
    aspect_ratio
        Ratio between width and height.
    padding_factor
        Factor to apply additional padding to the scale. Defaults to 1.0.

    Returns
    -------
    center : np.ndarray
        Center for bbox. Shape is [2,].
    scale : np.ndarray
        Scale for bbox. Shape is [2,].
    """
    x, y, w, h = box_cwh
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32) * padding_factor
    return center, scale


def apply_directional_box_offset(
    offset: float | torch.Tensor,
    vec_start: torch.Tensor,
    vec_end: torch.Tensor,
    xc: torch.Tensor,
    yc: torch.Tensor,
) -> None:
    """
    Offset the bounding box defined by [xc, yc] by a pre-determined length.
    The offset will be applied in the direction of the supplied vector.

    Parameters
    ----------
    offset
        Floating point offset to apply to the bounding box, in absolute values.
    vec_start
        Starting point of the vector. Shape is [B, 2], where 2 == (x, y).
    vec_end
        Ending point of the vector. Shape is [B, 2], where 2 == (x, y).
    xc
        X center of box. Modified in place.
    yc
        Y center of box. Modified in place.

    Returns
    -------
    None
        No return value; xc and yc are modified in place.
    """
    xlen = vec_end[..., 0] - vec_start[..., 0]
    ylen = vec_end[..., 1] - vec_start[..., 1]
    vec_len = torch.sqrt(torch.float_power(xlen, 2) + torch.float_power(ylen, 2))

    xc += offset * (xlen / vec_len)
    yc += offset * (ylen / vec_len)


def get_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """
    Compute the IoU between two boxes.

    Parameters
    ----------
    boxA
        First bounding box in xyxy format. Shape is (4,).
    boxB
        Second bounding box in xyxy format. Shape is (4,).

    Returns
    -------
    iou : float
        Intersection over union value between the two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return inter_area / float(boxA_area + boxB_area - inter_area)


@njit(parallel=True)
def get_bbox_iou_matrix(
    boxes: np.ndarray, query_boxes: np.ndarray, criterion: int = -1
) -> np.ndarray:
    """
    Calculate axis-aligned 2D IoU between sets of bounding boxes.

    Parameters
    ----------
    boxes
        Predicted boxes in format [x1, y1, x2, y2], of shape (N, 4).
    query_boxes
        Ground truth boxes in format [x1, y1, x2, y2], of shape (K, 4).
    criterion
        If 0, use area of box only; otherwise standard IoU. Defaults to -1.

    Returns
    -------
    iou_matrix : np.ndarray
        IoU matrix of shape (N, K).
    """
    N, K = boxes.shape[0], query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in prange(K):
        qbox_area = (query_boxes[k, 2] - query_boxes[k, 0]) * (
            query_boxes[k, 3] - query_boxes[k, 1]
        )
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - max(
                boxes[n, 0], query_boxes[k, 0]
            )
            ih = min(boxes[n, 3], query_boxes[k, 3]) - max(
                boxes[n, 1], query_boxes[k, 1]
            )
            if iw > 0 and ih > 0:
                ua = (
                    (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    + qbox_area
                    - iw * ih
                )
                if criterion == 0:
                    ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                overlaps[n, k] = iw * ih / ua
    return overlaps


def _wrap_pi(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def canonicalize_xywhr_radians(xywhr: torch.Tensor) -> torch.Tensor:
    """
    Canonicalize representation (cx,cy,w,h,theta[radians]) without changing geometry:
      enforce w >= h by swapping and adding pi/2, then wrap to (-pi, pi]
    Helps eliminate slight “extra rotation” appearance and improves matching stability.
    """
    out = xywhr.clone()
    w, h, t = out[..., 2], out[..., 3], out[..., 4]
    swap = h > w
    out[..., 2] = torch.where(swap, h, w)
    out[..., 3] = torch.where(swap, w, h)
    out[..., 4] = torch.where(swap, t + math.pi / 2.0, t)
    out[..., 4] = _wrap_pi(out[..., 4])
    return out


def rotated_batched_nms(
    pred_boxes_xywh: torch.Tensor,  # (B,N,4) center-based xywh
    pred_angles_rad: torch.Tensor,  # (B,N,1) or (B,N) radians
    pred_scores: torch.Tensor,  # (B,N)
    pred_class_idx: torch.Tensor,  # (B,N)
    score_thr: float,
    iou_thr: float,
    class_aware: bool = True,
    canonicalize: bool = False,
) -> tuple[
    list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]
]:
    """
    Common rotated NMS for both evaluator and demo.

    Returns per-image lists:
      boxes_xywh: (K,4) torch.float32
      angles_rad: (K,)  torch.float32  (still radians!)
      scores:     (K,)  torch.float32
      classes:    (K,)  torch.int64

    Uses probiou-based greedy NMS to match Ultralytics OBB evaluation exactly.
    """
    # Normalize angles to (B,N)
    ang = pred_angles_rad[..., 0] if pred_angles_rad.ndim == 3 else pred_angles_rad

    # Optional canonicalization on the *representation* (still radians)
    if canonicalize:
        xywhr = torch.cat([pred_boxes_xywh, ang.unsqueeze(-1)], dim=-1)
        xywhr = canonicalize_xywhr_radians(xywhr)
        pred_boxes_xywh = xywhr[..., :4]
        ang = xywhr[..., 4]

    boxes_t = pred_boxes_xywh.detach().cpu().float()  # (B,N,4)
    ang_t = ang.detach().cpu().float()  # (B,N)
    scores_t = pred_scores.detach().cpu().float()  # (B,N)
    cls_t = pred_class_idx.detach().cpu().long()  # (B,N)

    B = boxes_t.shape[0]

    out_boxes: list[torch.Tensor] = []
    out_scores: list[torch.Tensor] = []
    out_angles: list[torch.Tensor] = []
    out_classes: list[torch.Tensor] = []

    for i in range(B):
        b = boxes_t[i]  # (N,4)
        a = ang_t[i]  # (N,)
        s = scores_t[i]  # (N,)
        c = cls_t[i]  # (N,)

        m = s > score_thr
        if not m.any():
            out_boxes.append(torch.empty((0, 4), dtype=torch.float32))
            out_scores.append(torch.empty((0,), dtype=torch.float32))
            out_angles.append(torch.empty((0,), dtype=torch.float32))
            out_classes.append(torch.empty((0,), dtype=torch.int64))
            continue

        fb, fa, fs, fc = b[m], a[m], s[m], c[m]

        keep_ = _probiou_nms(fb, fa, fs, fc, iou_thr, class_aware)

        if keep_.numel() == 0:
            out_boxes.append(torch.empty((0, 4), dtype=torch.float32))
            out_scores.append(torch.empty((0,), dtype=torch.float32))
            out_angles.append(torch.empty((0,), dtype=torch.float32))
            out_classes.append(torch.empty((0,), dtype=torch.int64))
            continue

        out_boxes.append(fb[keep_])
        out_scores.append(fs[keep_])
        out_angles.append(fa[keep_])
        out_classes.append(fc[keep_])

    return out_boxes, out_scores, out_angles, out_classes


def _probiou_nms(
    boxes: torch.Tensor,
    angles: torch.Tensor,
    scores: torch.Tensor,
    classes: torch.Tensor,
    iou_thr: float,
    class_aware: bool,
) -> torch.Tensor:
    """
    Greedy NMS using batch_probiou, matching Ultralytics OBB evaluation.
    Returns indices (into the input arrays) of kept boxes, sorted by score descending.
    """
    order = torch.argsort(scores, descending=True)
    boxes_r = torch.cat([boxes, angles.unsqueeze(-1)], dim=-1)  # (N,5) xywhr

    if class_aware:
        kept: list[int] = []
        for cls_id in classes.unique():
            cls_mask = classes == cls_id
            cls_idx = order[cls_mask[order]]
            kept.extend(_greedy_probiou(boxes_r, cls_idx, iou_thr).tolist())
        if not kept:
            return torch.empty((0,), dtype=torch.long)
        kept_t = torch.tensor(kept, dtype=torch.long)
        return kept_t[torch.argsort(scores[kept_t], descending=True)]
    return _greedy_probiou(boxes_r, order, iou_thr)


def _greedy_probiou(
    boxes_r: torch.Tensor,
    order: torch.Tensor,
    iou_thr: float,
) -> torch.Tensor:
    """Greedy NMS over a single set of pre-sorted indices using probiou."""
    from ultralytics.utils.metrics import batch_probiou

    keep: list[int] = []
    for idx in order.tolist():
        if not keep:
            keep.append(idx)
            continue
        iou = batch_probiou(boxes_r[idx].unsqueeze(0), boxes_r[keep])
        if iou.max().item() < iou_thr:
            keep.append(idx)
    return torch.tensor(keep, dtype=torch.long)


def transform_boxes_to_original_size(
    boxes: torch.Tensor,
    pad: tuple[int, int],
    scale: float,
    original_height: int,
    original_width: int,
) -> torch.Tensor:
    """
    Transform bounding boxes back to original image size.

    Parameters
    ----------
    boxes
        Bounding boxes tensors, shape (num_detections, 4).
        Each box represented by (x1, y1, x2, y2) coordinates.

    pad
        Padding applied during resizing (x_pad, y_pad).

    scale
        Scale factor applied during resizing.

    original_height
        Original image height.

    original_width
        Original image width.

    Returns
    -------
    torch.Tensor
        Bounding boxes tensors, shape (num_detections, 4).
        Each box represented by (x1, y1, x2, y2) coordinates.
    """
    if len(boxes) > 0:
        # Adjust for padding (subtract padding) and scale
        boxes = boxes.clone().float()
        boxes[:, 0] = (boxes[:, 0] - pad[0]) / scale
        boxes[:, 1] = (boxes[:, 1] - pad[1]) / scale
        boxes[:, 2] = (boxes[:, 2] - pad[0]) / scale
        boxes[:, 3] = (boxes[:, 3] - pad[1]) / scale
        # Ensure boxes are within image boundaries
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)
        boxes[:, 2] = torch.clamp(boxes[:, 2], max=original_width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], max=original_height)
    return boxes
