# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.datasets.kitti import KITTI_LABELS_ASSET
from qai_hub_models.models.centernet_3d.kitti_utils import eval_class
from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import (
    AVERAGE_PRECISION,
    MetricMetadata,
)


class Kitti3DDetectionEvaluator(BaseEvaluator):
    """
    KITTI 3D object detection evaluator.

    Computes 2D bounding box AP, Average Orientation Similarity (AOS),
    and Bird's Eye View (BEV) AP following the KITTI benchmark protocol.
    """

    def __init__(self) -> None:
        KITTI_LABELS_ASSET.fetch(extract=True)
        # The data_object_label_2 archive extracts to a single label_2/ folder
        # (no training/ level), matching centernet_3d's KittiEvaluator.
        self.labels_data_path = KITTI_LABELS_ASSET.extracted_path / "label_2"
        self.reset()

    def reset(self) -> None:
        self.dt_annos: list[dict[str, np.ndarray]] = []
        self.gt_annos: list[dict[str, np.ndarray]] = []

    def add_batch(
        self,
        output: tuple[list[dict]],
        gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        Add a batch of predictions and ground truth for evaluation.

        Parameters
        ----------
        output
            Single-element tuple containing a list of per-image detection dicts.
            Each dict has keys:
            orients : list[float]
                Global orientation angles (rotation_y) per detection.
            dims : list[np.ndarray]
                3D dimensions [h, w, l] per detection.
            locations : list[list[float]]
                3D locations [x, y, z] per detection.
            pred_boxes_2d : list[list[float]]
                2D bounding boxes [x1, y1, x2, y2] in YOLO input space.
            pred_scores : list[float]
                Confidence scores per detection.
            yolo_h : int
                Height of the YOLO input image.
            yolo_w : int
                Width of the YOLO input image.
        gt
            img_id
                Image ID tensor with shape [B].
            c
                Original image center with shape [B, 2].
            s
                Original image scale (width, height) with shape [B, 2].
            calib
                Camera calibration matrix with shape [B, 3, 4].
        """
        (per_image_detections,) = output
        img_id_gt, c_gt, s_gt, _ = gt

        for b, detections in enumerate(per_image_detections):
            img_id = int(img_id_gt[b].item())
            s = s_gt[b].numpy()
            yolo_h = detections["yolo_h"]
            yolo_w = detections["yolo_w"]

            orients = detections["orients"]
            dims = detections["dims"]
            locations = detections["locations"]
            pred_boxes_2d_yolo = detections["pred_boxes_2d"]
            pred_scores = detections["pred_scores"]

            # Remap 2D boxes from YOLO input space (yolo_w x yolo_h) to
            # original image space using the inverse affine transform.
            # c and s encode the original image center and scale used during
            # preprocessing; denormalize_coordinates_affine inverts this exactly.
            # For KittiNoWarpDataset (no affine warp) this is an identity op.
            if len(pred_boxes_2d_yolo) > 0:
                corners = np.array(
                    [[b2d[0], b2d[1], b2d[2], b2d[3]] for b2d in pred_boxes_2d_yolo],
                    dtype=np.float64,
                )
                tl = denormalize_coordinates_affine(
                    corners[:, :2], c_gt[b].numpy(), s, 0, (yolo_w, yolo_h)
                )
                br = denormalize_coordinates_affine(
                    corners[:, 2:], c_gt[b].numpy(), s, 0, (yolo_w, yolo_h)
                )
                pred_boxes_2d = np.concatenate([tl, br], axis=1)
            else:
                pred_boxes_2d = np.zeros((0, 4), dtype=np.float64)

            if len(orients) > 0:
                # alpha = rotation_y - arctan2(x, z) (observation angle)
                alpha = np.array(
                    [
                        float(o) - float(np.arctan2(loc[0], loc[2]))
                        for o, loc in zip(orients, locations, strict=False)
                    ],
                    dtype=np.float64,
                )
                dt_annotations: dict[str, np.ndarray] = {
                    "name": np.array(["Car"] * len(orients)),
                    "truncated": np.zeros(len(orients), dtype=np.float64),
                    "occluded": np.zeros(len(orients), dtype=np.int32),
                    "alpha": alpha,
                    "bbox": pred_boxes_2d.reshape(-1, 4),
                    # KITTI dimensions format: [l, h, w] (length, height, width)
                    # dims from DeepBox are [h, w, l]
                    "dimensions": np.array(
                        [[d[2], d[0], d[1]] for d in dims], dtype=np.float64
                    ).reshape(-1, 3),
                    "location": np.array(locations, dtype=np.float64).reshape(-1, 3),
                    "rotation_y": np.array(orients, dtype=np.float64),
                    "score": np.array(pred_scores, dtype=np.float64),
                }
            else:
                dt_annotations = {
                    "name": np.array([], dtype=object),
                    "truncated": np.array([], dtype=np.float64),
                    "occluded": np.array([], dtype=np.int32),
                    "alpha": np.array([], dtype=np.float64),
                    "bbox": np.zeros((0, 4), dtype=np.float64),
                    "dimensions": np.zeros((0, 3), dtype=np.float64),
                    "location": np.zeros((0, 3), dtype=np.float64),
                    "rotation_y": np.array([], dtype=np.float64),
                    "score": np.array([], dtype=np.float64),
                }
            self.dt_annos.append(dt_annotations)

            # Load ground truth annotation from KITTI labels
            label_path = self.labels_data_path / f"{img_id:06d}.txt"
            with open(label_path) as f:
                lines = f.readlines()
            content = [line.strip().split(" ") for line in lines if line.strip()]
            gt_annotations: dict[str, np.ndarray] = {
                "name": np.array([x[0] for x in content]),
                "truncated": np.array([float(x[1]) for x in content], dtype=np.float64),
                "occluded": np.array([int(x[2]) for x in content], dtype=np.int32),
                "alpha": np.array([float(x[3]) for x in content], dtype=np.float64),
                "bbox": np.array(
                    [[float(v) for v in x[4:8]] for x in content], dtype=np.float64
                ).reshape(-1, 4),
                # KITTI label dimensions are [h, w, l]; convert to [l, h, w]
                "dimensions": np.array(
                    [[float(v) for v in x[8:11]] for x in content], dtype=np.float64
                ).reshape(-1, 3)[:, [2, 0, 1]],
                "location": np.array(
                    [[float(v) for v in x[11:14]] for x in content], dtype=np.float64
                ).reshape(-1, 3),
                "rotation_y": np.array(
                    [float(x[14]) for x in content], dtype=np.float64
                ),
                "score": (
                    np.array([float(x[15]) for x in content], dtype=np.float64)
                    if content and len(content[0]) == 16
                    else np.zeros(len(content), dtype=np.float64)
                ),
            }
            self.gt_annos.append(gt_annotations)

    def _has_any_detections(self) -> bool:
        """False if add_batch was never called, or every image had zero detections."""
        return any(len(anno["bbox"]) > 0 for anno in self.dt_annos)

    def get_accuracy_score(self) -> float:
        if not self._has_any_detections():
            return 0.0
        num_parts = max(1, len(self.dt_annos) // 100 + 1)
        bbox, _, _ = eval_class(
            self.gt_annos,
            self.dt_annos,
            difficultys=[0],
            num_parts=num_parts,
        )
        return float(bbox[0])

    def formatted_accuracy(self) -> str:
        if not self._has_any_detections():
            return "No detections"
        num_parts = max(1, len(self.dt_annos) // 100 + 1)
        bbox, aos, bev = eval_class(
            self.gt_annos,
            self.dt_annos,
            difficultys=[0, 1, 2],
            num_parts=num_parts,
        )
        bbox_str = ", ".join([f"{v:.2f}" for v in bbox])
        bev_str = ", ".join([f"{v:.2f}" for v in bev])
        aos_str = ", ".join([f"{v:.2f}" for v in aos])
        return f"{bbox_str} AP-(E,M,H) {aos_str} AOS-(E,M,H) {bev_str} BEV-(E,M,H)"

    def get_metric_metadata(self) -> MetricMetadata:
        return AVERAGE_PRECISION
