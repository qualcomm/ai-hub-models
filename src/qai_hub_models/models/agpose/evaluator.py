# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import glob
import math
import os
import pickle
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from qai_hub_models.models.agpose.dataset import CATEGORY_NAMES
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.bounding_box_processing_3d import (
    get_3d_bbox,
)
from qai_hub_models.utils.image_processing_3d import transform_coordinates_3d
from qai_hub_models.utils.metrics import MEAN_AVERAGE_PRECISION_5DEG_2CM, MetricMetadata


def _asymmetric_3d_iou(
    RT_1: np.ndarray,
    RT_2: np.ndarray,
    scales_1: np.ndarray,
    scales_2: np.ndarray,
) -> float:
    """
    Local asymmetric 3D IoU matching the upstream AG-Pose evaluation protocol.

    IMPORTANT — do not "fix" this to axis=1:
    bbox_3d has shape (3, 8).  axis=0 reduces over the 3-coordinate dimension,
    producing an **8-element vector** (one value per corner), NOT an AABB.
    np.prod of that 8-element vector is therefore NOT a geometric bounding-box
    volume.  The computation is numerically nonsensical as a volume measure,
    but it exactly reproduces the upstream evaluation_utils.py behaviour that
    yielded the published 57.0 % 5deg2cm baseline.  Changing axis would silently
    shift all IoU metrics and invalidate comparisons with the reported numbers.
    """
    bbox_3d_1 = transform_coordinates_3d(get_3d_bbox(scales_1, 0), RT_1)
    bbox_3d_2 = transform_coordinates_3d(get_3d_bbox(scales_2, 0), RT_2)

    bbox_1_max = np.amax(bbox_3d_1, axis=0)
    bbox_1_min = np.amin(bbox_3d_1, axis=0)
    bbox_2_max = np.amax(bbox_3d_2, axis=0)
    bbox_2_min = np.amin(bbox_3d_2, axis=0)

    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)

    if np.amin(overlap_max - overlap_min) < 0:
        intersection = 0.0
    else:
        intersection = float(np.prod(overlap_max - overlap_min))

    union = (
        float(np.prod(bbox_1_max - bbox_1_min))
        + float(np.prod(bbox_2_max - bbox_2_min))
        - intersection
    )
    return intersection / (union + 1e-8)


SYNSET_NAMES = ["BG", *CATEGORY_NAMES]

_DEGREE_THRESHOLDS = [5, 10]  # degrees
_SHIFT_THRESHOLDS = [0.02, 0.05]  # metres (2cm, 5cm official NOCS thresholds)

# Indices into the threshold lists used when reporting the PRIMARY metric (5°2cm)
_PRIMARY_DEG_IDX = _DEGREE_THRESHOLDS.index(5)
_PRIMARY_SHIFT_IDX = _SHIFT_THRESHOLDS.index(0.02)


def _compute_3d_iou(
    RT_1: np.ndarray,
    RT_2: np.ndarray,
    scales_1: np.ndarray,
    scales_2: np.ndarray,
    handle_visibility: int,
    class_name_1: str,
    class_name_2: str,
) -> float:
    """
    Compute 3D IoU with symmetry handling for rotationally symmetric objects.

    Parameters
    ----------
    RT_1
        Predicted pose matrix, shape (4, 4).
    RT_2
        Ground truth pose matrix, shape (4, 4).
    scales_1
        Predicted object size, shape (3,).
    scales_2
        Ground truth object size, shape (3,).
    handle_visibility
        1 if mug handle is visible, 0 if hidden (enables y-axis symmetry).
    class_name_1
        Predicted class name.
    class_name_2
        Ground truth class name.

    Returns
    -------
    float
        Maximum IoU over sampled rotations for symmetric classes, else direct IoU.
    """
    if RT_1 is None or RT_2 is None:
        return -1.0

    symmetric_classes = {"bottle", "bowl", "can"}
    is_symmetric = (
        class_name_1 in symmetric_classes and class_name_1 == class_name_2
    ) or (
        class_name_1 == "mug"
        and class_name_1 == class_name_2
        and handle_visibility == 0
    )

    if is_symmetric:
        n = 20
        max_iou = 0.0
        for i in range(n):
            theta = 2 * math.pi * i / n
            y_rot = np.array(
                [
                    [math.cos(theta), 0, math.sin(theta), 0],
                    [0, 1, 0, 0],
                    [-math.sin(theta), 0, math.cos(theta), 0],
                    [0, 0, 0, 1],
                ],
                dtype=np.float32,
            )
            max_iou = max(
                max_iou, _asymmetric_3d_iou(RT_1 @ y_rot, RT_2, scales_1, scales_2)
            )
        return max_iou
    return _asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)


def _compute_rt_degree_cm_symmetry(
    RT_1: np.ndarray,
    RT_2: np.ndarray,
    class_id: int,
    handle_visibility: int,
    synset_names: list[str],
) -> np.ndarray:
    """
    Returns [theta_degrees, shift_metres].
    shift = ||T1 - T2|| in metres, compared against thresholds [0.02, 0.05].
    """
    if RT_1 is None or RT_2 is None:
        return np.array([-1.0, -1.0])

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]) + 1e-8)
    T1 = RT_1[:3, 3]
    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]) + 1e-8)
    T2 = RT_2[:3, 3]

    cls_name = synset_names[class_id]
    symmetric_y = {"bottle", "can", "bowl"}

    if cls_name in symmetric_y or (cls_name == "mug" and handle_visibility == 0):
        y = np.array([0.0, 1.0, 0.0])
        y1, y2 = R1 @ y, R2 @ y
        cos_val = np.clip(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2) + 1e-8), -1.0, 1.0
        )
        theta = float(np.arccos(cos_val))
    else:
        R = R1 @ R2.T
        theta = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))

    theta_deg = theta * 180.0 / math.pi
    shift = float(np.linalg.norm(T1 - T2))  # metres, matches [0.02, 0.05] thresholds
    return np.array([theta_deg, shift])


def _compute_rt_overlaps(
    gt_class_ids: np.ndarray,
    gt_RTs: np.ndarray,
    gt_handle_visibility: np.ndarray,
    pred_class_ids: np.ndarray,
    pred_RTs: np.ndarray,
    synset_names: list[str],
) -> np.ndarray:
    """
    Compute rotation and translation overlaps between all pred/GT pairs.

    Parameters
    ----------
    gt_class_ids
        GT class IDs, shape (num_gt,).
    gt_RTs
        GT pose matrices, shape (num_gt, 4, 4).
    gt_handle_visibility
        Mug handle visibility flags, shape (num_gt,).
    pred_class_ids
        Predicted class IDs, shape (num_pred,).
    pred_RTs
        Predicted pose matrices, shape (num_pred, 4, 4).
    synset_names
        List of class names indexed by class ID.

    Returns
    -------
    np.ndarray
        Overlaps array, shape (num_pred, num_gt, 2) containing [degrees, metres].
    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    overlaps = np.zeros((num_pred, num_gt, 2), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = _compute_rt_degree_cm_symmetry(
                pred_RTs[i],
                gt_RTs[j],
                int(gt_class_ids[j]),
                int(gt_handle_visibility[j]),
                synset_names,
            )
    return overlaps


def _compute_match_from_degree_cm(
    overlaps: np.ndarray,
    pred_class_ids: np.ndarray,
    gt_class_ids: np.ndarray,
    degree_thres_list: list[float],
    shift_thres_list: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to GT instances using degree and shift thresholds.

    Parameters
    ----------
    overlaps
        Rotation/translation error matrix, shape (num_pred, num_gt, 2).
    pred_class_ids
        Predicted class IDs, shape (num_pred,).
    gt_class_ids
        GT class IDs, shape (num_gt,).
    degree_thres_list
        List of rotation error thresholds in degrees.
    shift_thres_list
        List of translation error thresholds in metres.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        gt_matches: shape (num_d, num_s, num_gt), pred_matches: shape (num_d, num_s, num_pred).
    """
    num_d = len(degree_thres_list)
    num_s = len(shift_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_d, num_s, num_pred))
    gt_matches = -1 * np.ones((num_d, num_s, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    for d, deg_thresh in enumerate(degree_thres_list):
        for s, shift_thresh in enumerate(shift_thres_list):
            for i in range(num_pred):
                sum_err = np.sum(overlaps[i, :, :], axis=-1)
                sorted_j = np.argsort(sum_err)
                for j in sorted_j:
                    if gt_matches[d, s, j] > -1:
                        continue
                    if pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    if (
                        overlaps[i, j, 0] > deg_thresh
                        or overlaps[i, j, 1] > shift_thresh
                    ):
                        continue
                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches


def _compute_ap_from_matches_scores(
    pred_match: np.ndarray,
    pred_scores: np.ndarray,
    gt_match: np.ndarray,
) -> float:
    """
    Compute average precision from match arrays and confidence scores.

    Parameters
    ----------
    pred_match
        Match array for predictions, shape (num_pred,). -1 means unmatched.
    pred_scores
        Confidence scores for predictions, shape (num_pred,).
    gt_match
        Match array for GT instances, shape (num_gt,).

    Returns
    -------
    float
        Average precision value in range [0, 1].
    """
    assert pred_match.shape[0] == pred_scores.shape[0]
    sort_idx = np.argsort(pred_scores)[::-1]
    pred_match = pred_match[sort_idx]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / max(len(gt_match), 1)

    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    idx = np.where(recalls[:-1] != recalls[1:])[0] + 1
    return float(np.sum((recalls[idx] - recalls[idx - 1]) * precisions[idx]))


def _compute_iou_overlaps(
    gt_class_ids: np.ndarray,
    gt_RTs: np.ndarray,
    gt_scales: np.ndarray,
    gt_handle_vis: np.ndarray,
    pred_class_ids: np.ndarray,
    pred_RTs: np.ndarray,
    pred_scales: np.ndarray,
    synset_names: list[str],
) -> np.ndarray:
    """
    Compute 3D IoU matrix between all predicted and GT bounding boxes.

    Parameters
    ----------
    gt_class_ids
        GT class IDs, shape (num_gt,).
    gt_RTs
        GT pose matrices, shape (num_gt, 4, 4).
    gt_scales
        GT object sizes, shape (num_gt, 3).
    gt_handle_vis
        Mug handle visibility flags, shape (num_gt,).
    pred_class_ids
        Predicted class IDs, shape (num_pred,).
    pred_RTs
        Predicted pose matrices, shape (num_pred, 4, 4).
    pred_scales
        Predicted object sizes, shape (num_pred, 3).
    synset_names
        List of class names indexed by class ID.

    Returns
    -------
    np.ndarray
        IoU matrix, shape (num_pred, num_gt), dtype float32.
    """
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    iou_mat = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            iou_mat[i, j] = _compute_3d_iou(
                pred_RTs[i],
                gt_RTs[j],
                pred_scales[i],
                gt_scales[j],
                int(gt_handle_vis[j]),
                synset_names[pred_class_ids[i]]
                if pred_class_ids[i] < len(synset_names)
                else "",
                synset_names[gt_class_ids[j]]
                if gt_class_ids[j] < len(synset_names)
                else "",
            )
    return iou_mat


def _compute_match_from_iou(
    iou_mat: np.ndarray,
    pred_class_ids: np.ndarray,
    gt_class_ids: np.ndarray,
    iou_thres_list: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to GT instances using 3D IoU thresholds.

    Parameters
    ----------
    iou_mat
        IoU matrix, shape (num_pred, num_gt).
    pred_class_ids
        Predicted class IDs, shape (num_pred,).
    gt_class_ids
        GT class IDs, shape (num_gt,).
    iou_thres_list
        List of IoU thresholds for matching.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        gt_matches: shape (num_i, num_gt), pred_matches: shape (num_i, num_pred).
    """
    num_i = len(iou_thres_list)
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    pred_matches = -1 * np.ones((num_i, num_pred))
    gt_matches = -1 * np.ones((num_i, num_gt))
    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches
    for t, iou_thresh in enumerate(iou_thres_list):
        for i in range(num_pred):
            best_iou = -1.0
            best_j = -1
            for j in range(num_gt):
                if gt_matches[t, j] > -1:
                    continue
                if pred_class_ids[i] != gt_class_ids[j]:
                    continue
                if iou_mat[i, j] > iou_thresh and iou_mat[i, j] > best_iou:
                    best_iou = iou_mat[i, j]
                    best_j = j
            if best_j >= 0:
                gt_matches[t, best_j] = i
                pred_matches[t, i] = best_j
    return gt_matches, pred_matches


def _compute_independent_mAP(
    final_results: list[dict],
    synset_names: list[str],
    degree_thresholds: Sequence[float],
    shift_thresholds: Sequence[float],
    iou_3d_thresholds: Sequence[float],
    iou_pose_thres: float = 0.1,
) -> np.ndarray:
    """
    Two modes:
      IoU-only mode : degree_thresholds=[360], shift_thresholds=[100]
                      Matches by 3D IoU only (for IoU25/IoU50 metrics).
      Pose mode     : normal degree/cm thresholds.
                      iou_pose_thres gates matching before degree/cm check.
    Returns aps shape: (num_classes+1, num_degree+1, num_shift+1, num_iou).
    """
    num_classes = len(synset_names)
    degree_thres_list = [*list(degree_thresholds), 360]
    shift_thres_list = [*list(shift_thresholds), 100]
    iou_thres_list = list(iou_3d_thresholds)
    iou_only_mode = degree_thresholds == [360] and shift_thresholds == [100]
    num_d = len(degree_thres_list)
    num_s = len(shift_thres_list)
    num_i = len(iou_thres_list)
    aps = np.zeros((num_classes + 1, num_d, num_s, num_i), dtype=np.float32)

    pred_matches_all: Any
    gt_matches_all: Any
    pred_scores_all: Any
    if iou_only_mode:
        pred_matches_all = [
            [np.zeros((0,)) for _ in range(num_i)] for _ in range(num_classes)
        ]
        gt_matches_all = [
            [np.zeros((0,)) for _ in range(num_i)] for _ in range(num_classes)
        ]
        pred_scores_all = [
            [np.zeros((0,)) for _ in range(num_i)] for _ in range(num_classes)
        ]
    else:
        pred_matches_all = [np.zeros((num_d, num_s, 0)) for _ in range(num_classes)]
        gt_matches_all = [np.zeros((num_d, num_s, 0)) for _ in range(num_classes)]
        pred_scores_all = [np.zeros((num_d, num_s, 0)) for _ in range(num_classes)]

    for result in final_results:
        gt_class_ids = np.array(result["gt_class_ids"], dtype=np.int32)
        gt_RTs = np.array(result["gt_RTs"], dtype=np.float32)
        gt_scales = np.array(result["gt_scales"], dtype=np.float32)
        gt_handle_vis = np.array(result["gt_handle_visibility"], dtype=np.int32)
        pred_class_ids = np.array(result["pred_class_ids"], dtype=np.int32)
        pred_RTs = np.array(result["pred_RTs"], dtype=np.float32)
        pred_scales = np.array(result["pred_scales"], dtype=np.float32)
        pred_scores = np.array(result["pred_scores"], dtype=np.float32)

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            gt_mask = gt_class_ids == cls_id
            pred_mask = pred_class_ids == cls_id
            cls_gt_ids = (
                gt_class_ids[gt_mask] if gt_mask.any() else np.zeros(0, dtype=np.int32)
            )
            cls_gt_RTs = (
                gt_RTs[gt_mask]
                if gt_mask.any()
                else np.zeros((0, 4, 4), dtype=np.float32)
            )
            cls_gt_scales = (
                gt_scales[gt_mask]
                if gt_mask.any()
                else np.zeros((0, 3), dtype=np.float32)
            )
            cls_gt_hv = (
                gt_handle_vis[gt_mask] if gt_mask.any() else np.zeros(0, dtype=np.int32)
            )
            cls_pred_ids = (
                pred_class_ids[pred_mask]
                if pred_mask.any()
                else np.zeros(0, dtype=np.int32)
            )
            cls_pred_RTs = (
                pred_RTs[pred_mask]
                if pred_mask.any()
                else np.zeros((0, 4, 4), dtype=np.float32)
            )
            cls_pred_scales = (
                pred_scales[pred_mask]
                if pred_mask.any()
                else np.zeros((0, 3), dtype=np.float32)
            )
            cls_pred_scores = (
                pred_scores[pred_mask]
                if pred_mask.any()
                else np.zeros(0, dtype=np.float32)
            )
            if synset_names[cls_id] != "mug":
                cls_gt_hv = np.ones_like(cls_gt_ids)

            # Compute IoU matrix — used in both modes
            iou_mat = _compute_iou_overlaps(
                cls_gt_ids,
                cls_gt_RTs,
                cls_gt_scales,
                cls_gt_hv,
                cls_pred_ids,
                cls_pred_RTs,
                cls_pred_scales,
                synset_names,
            )

            if iou_only_mode:
                # Pure IoU matching for IoU25/IoU50 metrics
                gt_match, pred_match = _compute_match_from_iou(
                    iou_mat,
                    cls_pred_ids,
                    cls_gt_ids,
                    iou_thres_list,
                )
                for t in range(num_i):
                    pred_matches_all[cls_id][t] = np.concatenate(
                        [pred_matches_all[cls_id][t], pred_match[t]]
                    )
                    gt_matches_all[cls_id][t] = np.concatenate(
                        [gt_matches_all[cls_id][t], gt_match[t]]
                    )
                    pred_scores_all[cls_id][t] = np.concatenate(
                        [pred_scores_all[cls_id][t], cls_pred_scores]
                    )
            else:
                # Sort predictions by descending confidence before matching
                # so greedy matching always considers the highest-confidence
                # prediction first, matching the upstream protocol.
                sort_idx = np.argsort(cls_pred_scores)[::-1]
                cls_pred_ids_s = cls_pred_ids[sort_idx]
                cls_pred_RTs_s = cls_pred_RTs[sort_idx]
                cls_pred_scores_s = cls_pred_scores[sort_idx]
                iou_mat_s = iou_mat[sort_idx]

                # Step 1: one-to-one IoU matching first, matching upstream
                # compute_combination_3d_matches flow. Each GT can only be
                # claimed by one prediction.
                _iou_gt_match, iou_pred_match = _compute_match_from_iou(
                    iou_mat_s,
                    cls_pred_ids_s,
                    cls_gt_ids,
                    [iou_pose_thres],
                )

                # Step 2: Filter BOTH pred and GT to IoU-matched instances only.
                # Matches upstream compute_independent_mAP with use_matches_for_pose=True:
                #   cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1]
                #   cls_gt_RTs   = cls_gt_RTs[iou_thres_gt_match > -1]
                # This ensures the pose AP recall denominator is only the
                # IoU-matched GT instances, not the full GT set.
                iou_thres_pred_match = iou_pred_match[0, :]  # shape (num_pred,)
                iou_thres_gt_match = _iou_gt_match[0, :]  # shape (num_gt,)

                pose_pred_ids_s = cls_pred_ids_s[iou_thres_pred_match > -1]
                pose_pred_RTs_s = cls_pred_RTs_s[iou_thres_pred_match > -1]
                pose_pred_scores_s = cls_pred_scores_s[iou_thres_pred_match > -1]

                pose_gt_ids = cls_gt_ids[iou_thres_gt_match > -1]
                pose_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1]
                pose_gt_hv = cls_gt_hv[iou_thres_gt_match > -1]

                rt_overlaps = _compute_rt_overlaps(
                    pose_gt_ids,
                    pose_gt_RTs,
                    pose_gt_hv,
                    pose_pred_ids_s,
                    pose_pred_RTs_s,
                    synset_names,
                )

                gt_match, pred_match = _compute_match_from_degree_cm(
                    rt_overlaps,
                    pose_pred_ids_s,
                    pose_gt_ids,
                    degree_thres_list,
                    shift_thres_list,
                )
                pred_matches_all[cls_id] = np.concatenate(
                    [pred_matches_all[cls_id], pred_match], axis=-1
                )
                gt_matches_all[cls_id] = np.concatenate(
                    [gt_matches_all[cls_id], gt_match], axis=-1
                )
                scores_tile = np.tile(pose_pred_scores_s, (num_d, num_s, 1))
                pred_scores_all[cls_id] = np.concatenate(
                    [pred_scores_all[cls_id], scores_tile], axis=-1
                )

    # Compute AP
    for cls_id in range(1, num_classes):
        if iou_only_mode:
            for t in range(num_i):
                aps[cls_id, 0, 0, t] = _compute_ap_from_matches_scores(
                    pred_matches_all[cls_id][t],
                    pred_scores_all[cls_id][t],
                    gt_matches_all[cls_id][t],
                )
        else:
            for d in range(num_d):
                for s in range(num_s):
                    aps[cls_id, d, s, 0] = _compute_ap_from_matches_scores(
                        pred_matches_all[cls_id][d, s, :],
                        pred_scores_all[cls_id][d, s, :],
                        gt_matches_all[cls_id][d, s, :],
                    )

    aps[-1] = np.mean(aps[1:-1], axis=0)
    return aps


class AGPoseEvaluator(BaseEvaluator):
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.synset_names = SYNSET_NAMES
        self._frame_gt: dict = {}
        self._load_pkl_index()
        self.reset()

    def _load_pkl_index(self) -> None:
        pkl_pattern = os.path.join(
            self.data_dir, "segmentation_results", "REAL275", "results_*.pkl"
        )
        pkl_files = sorted(glob.glob(pkl_pattern))
        if not pkl_files:
            raise FileNotFoundError(
                f"No REAL275 PKL files found at:\n  {pkl_pattern}\n"
                "Please check that segmentation_results/REAL275/ exists under data_dir."
            )
        print(f"[AGPoseEvaluator] Indexing {len(pkl_files)} REAL275 PKL files...")
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            img_path: str = data["image_path"]
            img_path = img_path.removeprefix("data/")
            parts = img_path.split("/")
            scene_name = parts[-2]
            frame_idx = int(parts[-1])
            self._frame_gt[(scene_name, frame_idx)] = {
                "gt_class_ids": np.array(data["gt_class_ids"], dtype=np.int32),
                "gt_RTs": np.array(data["gt_RTs"], dtype=np.float32),
                "gt_scales": np.array(data["gt_scales"], dtype=np.float32),
                "gt_handle_visibility": np.array(
                    data["gt_handle_visibility"], dtype=np.int32
                ),
                "gt_bboxes": np.array(data["gt_bboxes"], dtype=np.float32),
                "seg_pred_class_ids": np.array(data["pred_class_ids"], dtype=np.int32),
                "seg_pred_bboxes": np.array(data["pred_bboxes"], dtype=np.float32),
                "seg_pred_scores": np.array(data["pred_scores"], dtype=np.float32),
            }
        print(f"[AGPoseEvaluator] Indexed {len(self._frame_gt)} frames.")

    def reset(self) -> None:
        """Clear all accumulated predictions — called before each evaluation run."""
        self._preds_per_frame: dict = defaultdict(list)

    def add_batch(self, output: tuple, gt: tuple | list) -> None:
        """
        Accumulate one batch of model predictions with ground-truth metadata.

        Parameters
        ----------
        output
            3-tuple of model outputs: (pred_rotation, pred_translation, pred_size).
            Each element is either a torch.Tensor or np.ndarray of shape (B, ...) or
            unbatched (3,3), (3,), (3,) for a single sample.
        gt
            5-field ground-truth tuple whose order must match
            ``NocsReal275Dataset.__getitem__`` metadata:
            ``(scene_name, frame_idx, mask_id, cat_id_0idx, cat_id_1idx)``.
            Each field is either a scalar (single sample) or a list/tuple of
            length B (batched). cat_id_1idx is 1-indexed NOCS category ID.
        """
        pred_R, pred_t, pred_s = output

        # Handle tensor or numpy input
        def to_np(x: torch.Tensor | np.ndarray) -> np.ndarray:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().float().numpy()
            return np.array(x, dtype=np.float32)

        pred_R_arr = to_np(pred_R)
        pred_t_arr = to_np(pred_t)
        pred_s_arr = to_np(pred_s)
        # Normalize to batch dimension
        if pred_R_arr.ndim == 2:  # (3,3) → single sample
            pred_R_arr = pred_R_arr[np.newaxis]  # (1,3,3)
            pred_t_arr = pred_t_arr[np.newaxis]  # (1,3)
            pred_s_arr = pred_s_arr[np.newaxis]  # (1,3)

        batch_size = pred_R_arr.shape[0]

        for b in range(batch_size):
            pred_R_np = pred_R_arr[b].reshape(3, 3)
            pred_t_np = pred_t_arr[b].reshape(
                3,
            )
            pred_s_np = pred_s_arr[b].reshape(
                3,
            )

            pred_scale = np.linalg.norm(pred_s_np)
            pred_size = pred_s_np / (pred_scale + 1e-8)

            pred_RT = np.eye(4, dtype=np.float32)
            pred_RT[:3, 3] = pred_t_np
            pred_RT[:3, :3] = pred_R_np * pred_scale

            # Parse GT for this batch item
            if isinstance(gt[0], (list, tuple)):
                scene_name = gt[0][b]
                frame_idx = int(gt[1][b])
                mask_idx = int(gt[2][b])
                cat_1idx = int(gt[4][b])
            else:
                scene_name = gt[0]
                frame_idx = int(gt[1])
                mask_idx = int(gt[2])
                cat_1idx = int(gt[4])

            frame_key = (scene_name, frame_idx)
            pred_score = float(self._frame_gt[frame_key]["seg_pred_scores"][mask_idx])
            self._preds_per_frame[frame_key].append(
                {
                    "pred_RT": pred_RT,
                    "pred_size_normalized": pred_size,
                    "pred_class_id": cat_1idx,
                    "pred_score": pred_score,
                }
            )

    def _build_final_results(self) -> list:
        final_results = []
        for frame_key, preds in self._preds_per_frame.items():
            _scene_name, _frame_idx = frame_key
            if frame_key not in self._frame_gt:
                print(f"  [WARN] No PKL GT found for frame {frame_key}, skipping.")
                continue
            gt_data = self._frame_gt[frame_key]
            pred_RTs = np.stack([p["pred_RT"] for p in preds], axis=0)
            pred_scales = np.stack([p["pred_size_normalized"] for p in preds], axis=0)
            pred_cids = np.array([p["pred_class_id"] for p in preds], dtype=np.int32)
            pred_scores = np.array([p["pred_score"] for p in preds], dtype=np.float32)
            final_results.append(
                {
                    "gt_class_ids": gt_data["gt_class_ids"],
                    "gt_RTs": gt_data["gt_RTs"],
                    "gt_scales": gt_data["gt_scales"],
                    "gt_handle_visibility": gt_data["gt_handle_visibility"],
                    "pred_class_ids": pred_cids,
                    "pred_RTs": pred_RTs,
                    "pred_scales": pred_scales,
                    "pred_scores": pred_scores,
                }
            )
        return final_results

    def get_accuracy_score(self) -> float:
        final_results = self._build_final_results()
        if not final_results:
            return 0.0
        aps = _compute_independent_mAP(
            final_results,
            synset_names=self.synset_names,
            degree_thresholds=_DEGREE_THRESHOLDS,
            shift_thresholds=_SHIFT_THRESHOLDS,
            iou_3d_thresholds=[0.1],
            iou_pose_thres=0.1,
        )
        return float(aps[-1, _PRIMARY_DEG_IDX, _PRIMARY_SHIFT_IDX, 0]) * 100

    def formatted_accuracy(self) -> str:
        final_results = self._build_final_results()
        if not final_results:
            return "No predictions accumulated."

        aps_pose = _compute_independent_mAP(
            final_results,
            synset_names=self.synset_names,
            degree_thresholds=_DEGREE_THRESHOLDS,
            shift_thresholds=_SHIFT_THRESHOLDS,
            iou_3d_thresholds=[0.1],
            iou_pose_thres=0.1,
        )
        aps_iou = _compute_independent_mAP(
            final_results,
            synset_names=self.synset_names,
            degree_thresholds=[360],
            shift_thresholds=[100],
            iou_3d_thresholds=[0.25, 0.50, 0.75],
            iou_pose_thres=0.25,
        )

        deg_list = [*_DEGREE_THRESHOLDS, 360]
        shift_list = [*_SHIFT_THRESHOLDS, 100]

        def p(d: int, s: float) -> float:
            return float(aps_pose[-1, deg_list.index(d), shift_list.index(s), 0]) * 100

        iou25 = float(aps_iou[-1, 0, 0, 0]) * 100
        iou50 = float(aps_iou[-1, 0, 0, 1]) * 100
        iou75 = float(aps_iou[-1, 0, 0, 2]) * 100

        lines = [
            "── AG-Pose NOCS REAL275 Results ──────────────────",
            f"  IoU25      :  {iou25:6.2f}%",
            f"  IoU50      :  {iou50:6.2f}%",
            f"  IoU75      :  {iou75:6.2f}%",
            f"  5 deg 2cm :  {p(5, 0.02):6.2f}%",
            f"  5 deg 5cm :  {p(5, 0.05):6.2f}%",
            f"  10 deg 2cm :  {p(10, 0.02):6.2f}%",
            f"  10 deg 5cm :  {p(10, 0.05):6.2f}%",
            "───────────────────────────────────────────────────",
            "  Per-class 5deg2cm:",
        ]
        for cls_id in range(1, len(self.synset_names)):
            cls_ap = (
                float(aps_pose[cls_id, deg_list.index(5), shift_list.index(0.02), 0])
                * 100
            )
            lines.append(f"    {self.synset_names[cls_id]:<10}: {cls_ap:6.2f}%")
        return "\n".join(lines)

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_AVERAGE_PRECISION_5DEG_2CM
