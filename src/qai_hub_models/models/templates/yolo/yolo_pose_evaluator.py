# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.datasets.coco.coco_keypoints import CocoKeypointsDataset
from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.pose.pose_evaluator import CocoBodyPoseEvaluator
from qai_hub_models.utils.printing import suppress_stdout


class YoloPoseEvaluator(CocoBodyPoseEvaluator):
    """
    Evaluator for YOLO Pose.

    Expects postprocessed model output (include_postprocessing=True):

    boxes : torch.Tensor
        Shape [batch, num_preds, 4] — (x1, y1, x2, y2) in model pixel space.
    scores : torch.Tensor
        Shape [batch, num_preds] — confidence scores.
    keypoints : torch.Tensor
        Shape [batch, num_preds, num_keypoints, 3] — (x, y, visibility) in
        model pixel space with sigmoid-activated visibility.

    Ground truth (from CocoKeypointsDataset)
    -----------------------------------------
    image_ids   : torch.Tensor  [batch]
    category_ids: torch.Tensor  [batch]
    scales      : torch.Tensor  [batch]  — resize scale (model_px / orig_px)
    pads        : torch.Tensor  [batch, 2]  — (left_pad, top_pad) in model pixels
    """

    def __init__(
        self,
        in_vis_thre: float = 0.2,
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
    ) -> None:
        self.reset()
        self.in_vis_thre = in_vis_thre
        with suppress_stdout():
            self.coco_gt = CocoKeypointsDataset().cocoGt
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def add_batch(
        self,
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        gt: list[torch.Tensor],
    ) -> None:
        """
        Process one batch of YOLO Pose outputs.

        Parameters
        ----------
        output
            (boxes, scores, keypoints) as returned by Yolo*PoseDetector.forward()
            with include_postprocessing=True.

            boxes      : [batch, num_preds, 4]  (x1, y1, x2, y2) in model input space
            scores     : [batch, num_preds]      confidence scores
            keypoints  : [batch, num_preds, num_keypoints, 3]  (x, y, visibility)
                         x,y in model pixel space; visibility is sigmoid-activated.

        gt
            [image_ids, category_ids, scales, pads] tensors.
            image_ids   : [batch]
            category_ids: [batch]
            scales      : [batch]  resize scale (model_px / orig_px)
            pads        : [batch, 2]  (left_pad, top_pad) in model pixels
        """
        boxes, scores, keypoints = output
        image_ids, category_ids, scales, pads = gt

        # Apply NMS to remove overlapping detections.
        (
            _,
            det_scores,
            det_kpts,
        ) = batched_nms(
            self.iou_threshold,
            self.conf_threshold,
            boxes.cpu().float(),
            scores.cpu().float(),
            None,
            keypoints.cpu().float(),
        )

        batch_size = keypoints.shape[0]
        for b in range(batch_size):
            image_id = int(image_ids[b].item())
            category_id = int(category_ids[b].item())
            scale = float(scales[b].item())
            left_pad = int(pads[b][0].item())
            top_pad = int(pads[b][1].item())

            # Get NMS results for the current batch entry
            scores = det_scores[b]
            kpts = det_kpts[b]

            if scores.shape[0] == 0:
                continue

            # Map keypoint x,y from model pixel space back to original image space.
            kpts[:, :, 0] = (kpts[:, :, 0] - left_pad) / scale
            kpts[:, :, 1] = (kpts[:, :, 1] - top_pad) / scale

            # COCO eval skips detections where all visibility values are 0.
            # Quantized models may output 0 for all visibility values due to
            # quantization of sigmoid outputs. Fall back to a small positive
            # value so the detection is not silently dropped.
            # Create a mask for detections where all visibilities are zero
            all_vis_zero_mask = torch.all(kpts[:, :, 2] == 0, dim=1)
            # Set visibilities to 0.5 for those detections
            kpts[all_vis_zero_mask, :, 2] = 0.5

            # Flatten keypoints and create prediction entries
            kpts_flat = kpts.reshape(kpts.shape[0], -1)
            for i in range(scores.shape[0]):
                self.predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "keypoints": kpts_flat[i].tolist(),
                        "score": float(scores[i].item()),
                    }
                )
