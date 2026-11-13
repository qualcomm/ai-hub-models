# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np

from qai_hub_models.extern.numba import njit
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    get_bbox_iou_matrix,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    get_bev_iou_matrix,
)


@njit
def compute_statistics_jit(
    overlaps: np.ndarray,
    gt_datas: np.ndarray,
    dt_datas: np.ndarray,
    ignored_gt: np.ndarray,
    ignored_det: np.ndarray,
    dc_bboxes: np.ndarray,
    metric: int,
    min_overlap: float,
    thresh: float = 0.0,
    compute_fp: bool = False,
) -> tuple[int, int, int, float, np.ndarray]:
    """
    Compute evaluation statistics (TP, FP, FN, AOS) for a single frame between
    ground truth and detections, following KITTI's evaluation rules.

    Parameters
    ----------
    overlaps
        IoU matrix of shape [num_dets, num_gts] between detections and ground truths.
    gt_datas
        Ground truth boxes with shape [num_gts, 5] in format [x1, y1, x2, y2, alpha].
    dt_datas
        Detection boxes with shape [num_dets, 6] in format [x1, y1, x2, y2, alpha, score].
    ignored_gt
        Array indicating which GT boxes to ignore (-1, 0, 1), of shape (num_gts,).
    ignored_det
        Array indicating which DET boxes to ignore (-1, 0, 1), of shape (num_gts,).
    dc_bboxes
        Don't care regions (ignored in FP count) with shape [K, 4].
    metric
        0 for bbox IoU, 1 for BEV IoU.
    min_overlap
        Minimum IoU threshold to consider a match.
    thresh
        Score threshold to filter detections.
    compute_fp
        Whether to compute false positives and AOS (True for final eval).

    Returns
    -------
    tp : int
        Number of true positives.
    fp : int
        Number of false positives.
    fn : int
        Number of false negatives.
    similarity : float
        Orientation similarity (AOS), or -1 if no matches.
    thresholds : np.ndarray
        Score thresholds for matched true positives with shape (num_matched_tps,).
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if dt_scores[i] < thresh:
                ignored_threshold[i] = True

    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0.0

    thresholds = np.zeros((gt_size,))
    delta = np.zeros((gt_size,))
    delta_idx, thresh_idx = 0, 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0.0
        assigned_ignored_det = False

        for j in range(det_size):
            if ignored_det[j] == -1 or assigned_detection[j] or ignored_threshold[j]:
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]

            if not compute_fp and overlap > min_overlap and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score

            elif (
                compute_fp
                and overlap > min_overlap
                and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

            elif (
                compute_fp
                and overlap > min_overlap
                and valid_detection == NO_DETECTION
                and ignored_det[j] == 1
            ):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if valid_detection == NO_DETECTION and ignored_gt[i] == 0:
            fn += 1
        elif valid_detection != NO_DETECTION and (
            ignored_gt[i] == 1 or ignored_det[det_idx] == 1
        ):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
            delta_idx += 1
            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if not (
                assigned_detection[i]
                or ignored_det[i] in [-1, 1]
                or ignored_threshold[i]
            ):
                fp += 1

        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = get_bbox_iou_matrix(dt_bboxes, dc_bboxes, 0)
            for i in range(dc_bboxes.shape[0]):
                for j in range(det_size):
                    if (
                        assigned_detection[j]
                        or ignored_det[j] in [-1, 1]
                        or ignored_threshold[j]
                    ):
                        continue
                    if overlaps_dt_dc[j, i] > min_overlap:
                        assigned_detection[j] = True
                        nstuff += 1
        fp -= nstuff

        tmp = np.zeros((fp + delta_idx,))
        for i in range(delta_idx):
            tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0

        similarity = float(np.sum(tmp)) if (tp > 0 or fp > 0) else -1.0

    return tp, fp, fn, similarity, thresholds[:thresh_idx]


def _prepare_data(
    gt_annos: list[dict], dt_annos: list[dict], difficulty: int
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    int,
]:
    """
    Prepares data for KITTI evaluation by filtering, packing and summarizing GT and DT annotations.

    Parameters
    ----------
    gt_annos
        List of GT annotations.
    dt_annos
        List of DT annotations.
    difficulty
        Difficulty level.

    Returns
    -------
    gt_datas_list : list[np.ndarray]
        List of concatenated GT bboxes and alpha, each element of shape (N_i, 5).
    dt_datas_list : list[np.ndarray]
        List of concatenated DT bboxes, alpha, and score, each element of shape (M_i, 6).
    ignored_gts : list[np.ndarray]
        List of GT ignore flags per frame, each element of shape (N_i,).
    ignored_dets : list[np.ndarray]
        List of DT ignore flags per frame, each element of shape (M_i,).
    dontcares : list[np.ndarray]
        List of Don't Care bboxes per frame, each element of shape (K_i, 4).
    dc_bboxes_num : np.ndarray
        Count of don't-care objects per frame with shape (num_frames,).
    total_num_valid_gt : int
        Total number of valid GTs across all frames.
    """
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]

    gt_datas_list, dt_datas_list = [], []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_dc_num, total_num_valid_gt = [], 0

    for gt_anno, dt_anno in zip(gt_annos, dt_annos, strict=False):
        current_cls_name = "car"
        ignored_gt, ignored_dt, dc_bboxes_list = [], [], []
        num_valid_gt = 0

        for i, name in enumerate(gt_anno["name"]):
            name = name.lower()
            bbox = gt_anno["bbox"][i]
            height = bbox[3] - bbox[1]

            if name == current_cls_name:
                ignore = (
                    gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty]
                    or gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty]
                    or height <= MIN_HEIGHT[difficulty]
                )
                if ignore:
                    ignored_gt.append(1)
                else:
                    ignored_gt.append(0)
                    num_valid_gt += 1
            elif name == "van":
                ignored_gt.append(1)
            else:
                ignored_gt.append(-1)

            if name == "dontcare":
                dc_bboxes_list.append(bbox)

        for i, name in enumerate(dt_anno["name"]):
            name = name.lower()
            height = dt_anno["bbox"][i][3] - dt_anno["bbox"][i][1]
            if height < MIN_HEIGHT[difficulty]:
                ignored_dt.append(1)
            elif name == current_cls_name:
                ignored_dt.append(0)
            else:
                ignored_dt.append(-1)

        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_dt, dtype=np.int64))
        dc_bboxes = (
            np.stack(dc_bboxes_list, 0).astype(np.float64)
            if dc_bboxes_list
            else np.zeros((0, 4), dtype=np.float64)
        )
        total_dc_num.append(len(dc_bboxes))
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt

        gt_data = np.concatenate([gt_anno["bbox"], gt_anno["alpha"][..., None]], axis=1)
        dt_data = np.concatenate(
            [dt_anno["bbox"], dt_anno["alpha"][..., None], dt_anno["score"][..., None]],
            axis=1,
        )
        gt_datas_list.append(gt_data)
        dt_datas_list.append(dt_data)

    return (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        np.array(total_dc_num, dtype=np.int64),
        total_num_valid_gt,
    )


def get_thresholds(
    scores: np.ndarray, num_gt: int, num_sample_pts: int = 41
) -> list[float]:
    """
    Computes confidence thresholds based on score distribution for recall interpolation.

    Parameters
    ----------
    scores
        Detection scores with shape (N,).
    num_gt
        Number of valid GT objects.
    num_sample_pts
        Number of precision samples.

    Returns
    -------
    thresholds : list[float]
        List of score thresholds.
    """
    scores = np.sort(scores)[::-1]
    thresholds, cur_recall = [], 0.0
    for i in range(len(scores)):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i + 1 < len(scores) else l_recall
        if (r_recall - cur_recall) < (cur_recall - l_recall) and i + 1 < len(scores):
            continue
        thresholds.append(scores[i])
        cur_recall += 1 / (num_sample_pts - 1)
    return thresholds


def eval_class(
    gt_annos: list[dict],
    dt_annos: list[dict],
    difficultys: list[int],
    z_axis: int = 1,
    num_parts: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform KITTI-style evaluation for the 'Car' class.

    Parameters
    ----------
    gt_annos
        List of ground truth annotations per sample (frame). Each dict has keys:
        'bbox' (np.ndarray), 'alpha' (np.ndarray), 'name' (list[str]),
        'occluded' (np.ndarray), 'truncated' (np.ndarray), 'location' (np.ndarray),
        'dimensions' (np.ndarray), 'rotation_y' (np.ndarray).
    dt_annos
        List of detection results per sample (frame). Each dict has keys:
        'bbox' (np.ndarray), 'alpha' (np.ndarray), 'score' (np.ndarray),
        'name' (list[str]), 'location' (np.ndarray), 'dimensions' (np.ndarray),
        'rotation_y' (np.ndarray).
    difficultys
        Difficulty levels to evaluate [0, 1, 2] for easy, moderate, hard.
    z_axis
        Axis representing height (default is 1 for KITTI).
    num_parts
        Number of chunks to split data for large eval sets.

    Returns
    -------
    bbox_ap : np.ndarray
        Average precision (AP) for 'Car' class based on 2D bounding box IoU
        with shape of (len(difficultys),).
    aos : np.ndarray
        Average orientation similarity (AOS) for 'Car' class
        with shape of (len(difficultys),).
    bev_ap : np.ndarray
        Average precision (AP) for 'Car' class based on Bird's Eye View IoU
        with shape of (len(difficultys),).
    """
    assert len(gt_annos) == len(dt_annos)
    same_part = len(gt_annos) // num_parts
    remain_num = len(gt_annos) % num_parts
    if remain_num == 0:
        split_parts = [same_part] * num_parts
    else:
        split_parts = [same_part] * num_parts + [remain_num]

    total_dt = np.array([len(a["name"]) for a in dt_annos])
    total_gt = np.array([len(a["name"]) for a in gt_annos])
    bev_overlaps, bev_parted, bbox_overlaps, bbox_parted = [], [], [], []
    bev_axes = [i for i in range(3) if i != z_axis]
    idx = 0

    for n in split_parts:
        gt = gt_annos[idx : idx + n]
        dt = dt_annos[idx : idx + n]

        g, d = (
            np.concatenate([a["bbox"] for a in gt]),
            np.concatenate([a["bbox"] for a in dt]),
        )
        bbox_o = get_bbox_iou_matrix(d, g)

        def pack(a: list[dict]) -> np.ndarray:
            loc = np.concatenate([x["location"][:, bev_axes] for x in a])
            dim = np.concatenate([x["dimensions"][:, bev_axes] for x in a])
            rot = np.concatenate([x["rotation_y"] for x in a])[:, None]
            return np.concatenate([loc, dim, rot], 1)

        bev_o = get_bev_iou_matrix(pack(dt), pack(gt)).astype(np.float64)

        bev_parted.append(bev_o)
        bbox_parted.append(bbox_o)
        idx += n

    idx = 0
    for j, n in enumerate(split_parts):
        g_idx = d_idx = 0
        for i in range(n):
            g, d = total_gt[idx + i], total_dt[idx + i]
            bev_overlaps.append(bev_parted[j][d_idx : d_idx + d, g_idx : g_idx + g])
            bbox_overlaps.append(bbox_parted[j][d_idx : d_idx + d, g_idx : g_idx + g])
            g_idx += g
            d_idx += d
        idx += n

    metrics = {}
    metric_types = ["bbox", "bev"]
    for metric, metric_type in enumerate(metric_types):
        if metric_type == "bbox":
            min_overlap = 0.7
            overlaps = bbox_overlaps
        else:
            min_overlap = 0.5
            overlaps = bev_overlaps

        N_SAMPLE_PTS = 41
        num_difficulty = len(difficultys)

        precision = np.zeros([num_difficulty, N_SAMPLE_PTS])
        aos = np.zeros([num_difficulty, N_SAMPLE_PTS])

        for j, difficulty in enumerate(difficultys):
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                _total_dc_num,
                total_num_valid_gt,
            ) = _prepare_data(gt_annos, dt_annos, difficulty)

            thresholdss = []
            for i in range(len(gt_annos)):
                tp, fp, fn, similarity, thresholds = compute_statistics_jit(
                    overlaps[i],
                    gt_datas_list[i],
                    dt_datas_list[i],
                    ignored_gts[i],
                    ignored_dets[i],
                    dontcares[i],
                    metric,
                    min_overlap,
                    thresh=0.0,
                    compute_fp=False,
                )
                thresholdss += thresholds.tolist()
            thresholds = get_thresholds(np.array(thresholdss), total_num_valid_gt)

            pr = np.zeros([len(thresholds), 4])
            for i in range(len(gt_annos)):
                for t, thresh in enumerate(thresholds):
                    tp, fp, fn, similarity, _ = compute_statistics_jit(
                        overlaps[i],
                        gt_datas_list[i],
                        dt_datas_list[i],
                        ignored_gts[i],
                        ignored_dets[i],
                        dontcares[i],
                        metric,
                        min_overlap=min_overlap,
                        thresh=thresh,
                        compute_fp=True,
                    )
                    pr[t, 0] += tp
                    pr[t, 1] += fp
                    pr[t, 2] += fn
                    if similarity != -1:
                        pr[t, 3] += similarity

            for i in range(len(thresholds)):
                denom = pr[i, 0] + pr[i, 1]
                precision[j, i] = pr[i, 0] / denom if denom > 0 else 0
                aos[j, i] = pr[i, 3] / denom if denom > 0 else 0

            for i in range(len(thresholds)):
                precision[j, i] = np.max(precision[j, i:], axis=-1)
                aos[j, i] = np.max(aos[j, i:], axis=-1)

        metrics[metric_types[metric]] = {"precision": precision, "orientation": aos}
    bbox = get_mAP(metrics["bbox"]["precision"])
    bev = get_mAP(metrics["bev"]["precision"])
    aos = get_mAP(metrics["bbox"]["orientation"])
    return bbox, aos, bev


def get_mAP(prec: np.ndarray) -> np.ndarray:
    """
    Computes mean average precision (mAP) using 11-point interpolation.

    Parameters
    ----------
    prec
        Interpolated precision values with shape (num_difficulties, num_sample_pts).

    Returns
    -------
    mAP_scores : np.ndarray
        mAP score for each difficulty level with shape (num_difficulties,).
    """
    sums = np.array([0.0] * prec.shape[0])
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100
