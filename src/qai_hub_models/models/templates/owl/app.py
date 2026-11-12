# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL.Image import Image
from transformers import AutoProcessor

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.draw import draw_box_from_xyxy
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


class OwlApp:
    """End-to-end inference app for the Owl open-vocabulary object detector."""

    def __init__(
        self,
        model: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        hf_model_id: str,
        nms_score_threshold: float = 0.1,
        nms_iou_threshold: float = 0.1,
    ) -> None:
        self.model = model
        self.processor = AutoProcessor.from_pretrained(
            hf_model_id,
            use_fast=True,
        )

        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def inference(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_boxes, pred_scores, pred_class_idx = self.model(
            pixel_values, input_ids, attention_mask
        )
        return pred_boxes, pred_scores, pred_class_idx

    def predict_boxes_from_image(
        self,
        pixel_values_or_image: (torch.Tensor | np.ndarray | Image | list[Image]),
        text_queries: list[str],
        raw_output: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[np.ndarray]
    ):
        """
        Run open-vocabulary object detection on a single image.

        Parameters
        ----------
        pixel_values_or_image
            A PIL Image or a path / URL to an image file.
        text_queries
            List of text descriptions to search for, e.g.
            ``["a photo of a cat", "a photo of a remote"]``.
        raw_output
           See "returns" doc section for details.

        Returns
        -------
        output : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[np.ndarray]
            If raw_output is True, returns:
                boxes : list[torch.Tensor]
                    Bounding box locations per batch.
                    List element shape is [num preds, 4] where 4 == (x1, y1, x2, y2).
                scores : list[torch.Tensor]
                    Class scores per batch multiplied by confidence.
                    List element shape is [num_preds, # of classes (typically 80)].
                class_idx : list[torch.Tensor]
                    Shape is [num_preds] where the values are the indices of the most probable class of the prediction.

            If raw_output is False, returns:
                images : list[np.ndarray]
                    A list of predicted RGB, [H, W, C] images (one list element per batch).
                    Each image will have bounding boxes drawn.
        """
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)

        inputs = self.processor(
            text=[text_queries],
            images=pixel_values_or_image,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"].to(torch.int32)
        attention_mask = inputs["attention_mask"].to(torch.int32)

        proc_H, proc_W = pixel_values.shape[2], pixel_values.shape[3]

        input_ids = input_ids.reshape(-1, input_ids.shape[-1])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1])
        # Inference
        pred_boxes, pred_scores, pred_class_idx = self.inference(
            pixel_values, input_ids, attention_mask
        )

        pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx = (
            batched_nms(
                self.nms_iou_threshold,
                self.nms_score_threshold,
                pred_boxes,
                pred_scores,
                pred_class_idx,
            )
        )
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return (pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx)

        for batch_idx in range(len(pred_post_nms_boxes)):
            pred_boxes_batch = pred_post_nms_boxes[batch_idx]
            pred_class_idx_batch = pred_post_nms_class_idx[batch_idx]

            orig_H, orig_W = NHWC_int_numpy_frames[batch_idx].shape[:2]
            pred_boxes_batch = self._map_boxes_to_original(
                pred_boxes_batch, proc_H, proc_W, orig_H, orig_W
            )

            for i, box in enumerate(pred_boxes_batch):
                class_idx = int(pred_class_idx_batch[i].item())

                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[batch_idx],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=(0, 255, 0),
                    size=2,
                    text=f"{text_queries[class_idx]}",
                )

        return NHWC_int_numpy_frames

    def _map_boxes_to_original(
        self,
        boxes: torch.Tensor,
        proc_h: int,
        proc_w: int,
        orig_h: int,
        orig_w: int,
    ) -> torch.Tensor:
        """
        Map boxes from processor pixel space back to original image pixel space.

        Default implementation assumes the processor did an aspect-distorting squash
        resize (OwlViTImageProcessor), which requires separate x/y scale factors.

        """
        if len(boxes) == 0:
            return boxes
        boxes = boxes.clone().float()
        boxes[:, 0] = (boxes[:, 0] * orig_w / proc_w).clamp(0, orig_w)
        boxes[:, 1] = (boxes[:, 1] * orig_h / proc_h).clamp(0, orig_h)
        boxes[:, 2] = (boxes[:, 2] * orig_w / proc_w).clamp(0, orig_w)
        boxes[:, 3] = (boxes[:, 3] * orig_h / proc_h).clamp(0, orig_h)
        return boxes
