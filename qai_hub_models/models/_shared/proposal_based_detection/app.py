# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
import torch.nn.functional as F

from qai_hub_models.utils.bounding_box_processing import batched_nms


class ProposalBasedDetectionApp:
    """
    This class consists of light-weight "app code" that is required to
    end-to-end inference with proposal-based detection models:
      - Detectron2
      - MaskRCNN
    """

    def __init__(
        self,
        model_image_height: int = 800,
        model_image_width: int = 800,
        proposal_iou_threshold: float = 0.7,
        boxes_iou_threshold: float = 0.5,
        boxes_score_threshold: float = 0.8,
        max_det_pre_nms: int = 6000,
        max_det_post_nms: int = 200,
    ) -> None:
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width
        self.proposal_iou_threshold = proposal_iou_threshold
        self.boxes_iou_threshold = boxes_iou_threshold
        self.boxes_score_threshold = boxes_score_threshold
        self.max_det_pre_nms = max_det_pre_nms
        self.max_det_post_nms = max_det_post_nms

    def filter_proposals(
        self, proposals: list[torch.Tensor], objectness_logits: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """
        Filter and process region proposals based on objectness scores.

        Parameters
        ----------
        proposals
            List of tensors containing bounding box coordinates
            with shape [batch_size, num_proposals, 4]
        objectness_logits
            List of tensors containing objectness scores with shape
            [batch_size, num_proposals]

        Returns
        -------
        filtered_proposals : list[torch.Tensor]
            List of padded tensors containing filtered proposals
            with shape [batch_size, max_det_post_nms, 4]
        """
        padded_proposals = []
        for pred_proposals, pred_objectness_logits in zip(
            proposals, objectness_logits, strict=False
        ):
            pred_proposals[:, :, 0::2] = pred_proposals[:, :, 0::2].clamp(
                min=0, max=self.model_image_width
            )
            pred_proposals[:, :, 1::2] = pred_proposals[:, :, 1::2].clamp(
                min=0, max=self.model_image_height
            )

            # keep max_det_pre_nms
            batch_size = pred_proposals.shape[0]
            top_k = min(pred_objectness_logits.shape[1], self.max_det_pre_nms)
            topk_scores, topk_idx = pred_objectness_logits.topk(top_k, dim=1)
            topk_proposals = pred_proposals[
                torch.arange(batch_size).unsqueeze(1), topk_idx
            ]

            # filter empty boxes
            widths = topk_proposals[:, :, 2] - topk_proposals[:, :, 0]
            heights = topk_proposals[:, :, 3] - topk_proposals[:, :, 1]
            keep = (widths > 0) & (heights > 0)
            if keep.sum() < keep.shape[1]:
                topk_proposals = torch.stack(
                    [topk_proposals[b, keep[b]] for b in range(batch_size)]
                )
                topk_scores = torch.stack(
                    [topk_scores[b, keep[b]] for b in range(batch_size)]
                )

            selected_proposals, _ = batched_nms(
                self.proposal_iou_threshold,
                None,
                topk_proposals,
                topk_scores,
            )

            # Truncate to max_det_post_nms boxes and  if less, Zero-pad tensor
            # to make it shape static.
            padded_batch = []
            for b in range(batch_size):
                selected_proposals[b] = selected_proposals[b][: self.max_det_post_nms]
                padding = self.max_det_post_nms - selected_proposals[b].shape[0]
                selected_proposals[b] = F.pad(selected_proposals[b], (0, 0, 0, padding))
                padded_batch.append(selected_proposals[b])

            padded_proposals.append(torch.stack(padded_batch))
        return padded_proposals
