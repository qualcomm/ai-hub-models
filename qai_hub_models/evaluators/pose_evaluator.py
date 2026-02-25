# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from qai_hub_models.datasets.cocobody import CocoBodyDataset
from qai_hub_models.evaluators.base_evaluators import BaseEvaluator, MetricMetadata
from qai_hub_models.evaluators.utils.pose import get_final_preds
from qai_hub_models.extern.xtcocotools.cocoeval import COCOeval
from qai_hub_models.utils.printing import suppress_stdout


class CocoBodyPoseEvaluator(BaseEvaluator):
    """Evaluator for keypoint-based pose estimation using COCO-style mAP."""

    def __init__(self, in_vis_thre: float = 0.2) -> None:
        """
        Parameters
        ----------
        in_vis_thre
            Visibility threshold for keypoints.
        """
        self.reset()
        self.in_vis_thre = in_vis_thre
        self.coco_gt = CocoBodyDataset().cocoGt

    def reset(self) -> None:
        """Resets the collected predictions."""
        self.predictions: list[dict[str, Any]] = []

    def add_batch(
        self, output: tuple[torch.Tensor] | torch.Tensor, gt: list[torch.Tensor]
    ) -> None:
        """Processes a batch of model outputs and ground truth"""
        raise NotImplementedError("add_batch must be implemented in subclass")

    def _store_predictions(
        self,
        preds: np.ndarray,
        maxvals: np.ndarray,
        image_ids: torch.Tensor | list[int],
        category_ids: torch.Tensor | list[int],
    ) -> None:
        """
        Store pose predictions in COCO evaluation format.

        Parameters
        ----------
        preds
            Array of predicted keypoints in image coordinates.
            Shape: [batch_size, num_joints, 2] where last dim is (x,y)
        maxvals
            Array of confidence scores for each keypoint.
            Shape: [batch_size, num_joints, 1]
        image_ids
            Tensor containing COCO image_id for each prediction.
            Shape: [batch_size]
        category_ids
            Tensor containing COCO category IDs (typically all 1 for person).
            Shape: [batch_size]
        """
        for idx in range(preds.shape[0]):
            image_id = int(image_ids[idx])
            category_id = int(category_ids[idx])

            # Convert keypoints to COCO format [x1, y1, v1, ..., x17, y17, v17]
            keypoints_list = []
            for joint_idx in range(preds.shape[1]):
                x, y = preds[idx][joint_idx]
                v = float(maxvals[idx][joint_idx])
                keypoints_list.extend([float(x), float(y), v])

            # Compute keypoint-based confidence score

            box_score = np.mean(maxvals[idx])
            kpt_score = float(0)
            valid_num = float(0)
            for n_jt in range(preds.shape[1]):
                t_s = maxvals[idx][n_jt]
                if t_s > self.in_vis_thre:
                    kpt_score += t_s
                    valid_num += 1
            if valid_num > 0:
                kpt_score /= valid_num
            final_score = kpt_score * box_score

            # Store prediction in correct format
            pred_dict = {
                "image_id": image_id,
                "category_id": category_id,
                "keypoints": keypoints_list,
                "score": float(final_score),
            }

            self.predictions.append(pred_dict)

    def get_coco_mAP(self) -> dict[str, Any]:
        """
        Computes COCO-style mAP using COCOeval.

        Returns
        -------
        metrics : dict[str, Any]
            A dictionary with AP values (mAP, AP@0.5, etc.).
        """
        pred_image_ids = [p["image_id"] for p in self.predictions]

        res = copy.deepcopy(self.predictions)
        with suppress_stdout():
            coco_dt = self.coco_gt.loadRes(res)
            coco_eval = COCOeval(self.coco_gt, coco_dt, "keypoints")
            coco_eval.params.useSegm = None
            coco_eval.params.imgIds = pred_image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        return {"AP": coco_eval.stats[0] * 100, "AP@.5": coco_eval.stats[1] * 100}

    def get_accuracy_score(self) -> float:
        """Returns the overall mAP score."""
        return self.get_coco_mAP()["AP"]

    def formatted_accuracy(self) -> str:
        """Formats the mAP score for display."""
        results = self.get_coco_mAP()
        return f"mAP: {results['AP']:.3f}, AP@.5: {results['AP@.5']:.3f}"

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Mean Average Precision",
            unit="mAP",
            description="Percentage of keypoints that are close to the expected location.",
            range=(0.0, 100.0),
            float_vs_device_threshold=10.0,
        )


class MPIIPoseEvaluator(BaseEvaluator):
    """Evaluator for tracking accuracy of a Pose Estimation Model using MPII."""

    def __init__(self) -> None:
        self.reset()

    def add_batch(self, output: torch.Tensor, gt: list[torch.Tensor]) -> None:
        gt_keypoints, headboxes, joint_missing, center, scale = gt

        preds, _ = get_final_preds(output.numpy(), center.numpy(), scale.numpy())
        self.preds.append(preds)
        self.gt_keypoints.append(gt_keypoints)
        self.headboxes.append(headboxes)
        self.joint_missing.append(joint_missing)

    def reset(self) -> None:
        self.preds: list[np.ndarray] = []
        self.gt_keypoints: list[torch.Tensor] = []
        self.headboxes: list[torch.Tensor] = []
        self.joint_missing: list[torch.Tensor] = []

    def get_accuracy_score(self) -> float:
        joint_missing = np.transpose(np.concatenate(self.joint_missing), (1, 0))
        gt_keypoints = np.transpose(np.concatenate(self.gt_keypoints), (1, 2, 0))
        headboxes = np.transpose(np.concatenate(self.headboxes), (1, 2, 0))

        # convert 0-based index to 1-based index
        pred_keypoints = np.transpose(np.concatenate(self.preds), [1, 2, 0]) + 1.0

        # Reference for metric calculation from
        # https://github.com/HRNet/HRNet-Human-Pose-Estimation/blob/00d7bf72f56382165e504b10ff0dddb82dca6fd2/lib/dataset/mpii.py#L107
        SC_BIAS = 0.6
        threshold = 0.5

        jnt_visible = 1 - joint_missing
        uv_error = pred_keypoints - gt_keypoints
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes[0, :, :] - headboxes[1, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
        PCKh = np.divide(100.0 * np.sum(less_than_threshold, axis=1), jnt_count)

        rng = np.arange(0, 0.5 + 0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
            pckAll[r, :] = np.divide(
                100.0 * np.sum(less_than_threshold, axis=1), jnt_count
            )

        pckh_mask = np.zeros(PCKh.shape, dtype=bool)
        pckh_mask[6:8] = True
        PCKh = np.ma.array(PCKh, mask=pckh_mask)

        jnt_count_mask = np.zeros(jnt_count.shape, dtype=bool)
        jnt_count_mask[6:8] = True
        jnt_count = np.ma.array(jnt_count, mask=jnt_count_mask)
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        mean = np.sum(PCKh * jnt_ratio)
        self.mean_ratio = np.sum(pckAll[11, :] * jnt_ratio)
        return mean * 100

    def formatted_accuracy(self) -> str:
        mean = self.get_accuracy_score()
        return (
            f"{mean:.3f} (Percentage Mean), {self.mean_ratio:.3f} (Percentage Mean@0.1)"
        )

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Percentage Correct Keypoints (head-normalized)",
            unit="PCKh",
            description="Percentage of keypoints within a certain distance of expected.",
            range=(0.0, 100.0),
            float_vs_device_threshold=10.0,
        )
