# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from PIL.Image import Image
from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from torchvision.transforms import v2

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.draw import draw_box_from_xyxy
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.sam.app import SAMInputImageLayout


class SAM3App:
    """
    Light-weight app code for end-to-end inference with SAM3 (Segment Anything Model 3).

    SAM3 extends SAM2 with text-based prompting, allowing segmentation
    from natural language descriptions. This app supports text prompts only.

    The app uses 2 models:
      * vision_backbone (image -> FPN features)
      * head (tokenized + FPN -> pred_boxes, scores, pred_masks)
    """

    def __init__(
        self,
        input_image_channel_layout: SAMInputImageLayout,
        sam3_vision_backbone: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        sam3_head: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        tokenizer: Callable[..., torch.Tensor],
        context_length: int,
        image_height: int,
        image_width: int,
        mask_threshold: float = 0.0,
        nms_iou_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize the SAM3 segmentation model."""
        self.sam3_vision_backbone = sam3_vision_backbone
        self.sam3_head = sam3_head
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.input_image_channel_layout = input_image_channel_layout
        self.image_height = image_height
        self.image_width = image_width
        self.mask_threshold = mask_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self._device = torch.device(device)

    def predict(self, *args: Any, **kwargs: Any) -> Image:
        """Default prediction method - delegates to text-based segmentation."""
        return self.predict_mask_from_text(*args, **kwargs)

    def predict_mask_from_text(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        text_prompts: list[str],
        confidence_threshold: float = 0.5,
    ) -> Image:
        """
        Predicts segmentation masks from a single image and text prompts,
        and returns a PIL Image with masks and bounding boxes overlaid.

        Parameters
        ----------
        pixel_values_or_image
            PIL image, numpy array (H W C x uint8),
            or pyTorch tensor (1 C H W x float32, value range [0, 1]),
            with channel layout consistent with self.input_image_channel_layout.
        text_prompts
            List of text descriptions for objects to segment (e.g., ["cup", "person"]).
        confidence_threshold
            Minimum confidence score for keeping predictions.

        Returns
        -------
        Image
            PIL Image with predicted masks and bounding boxes overlaid.
        """
        numpy_frames, img_tensor = app_to_net_image_inputs(
            pixel_values_or_image,
            image_layout=self.input_image_channel_layout.name,
            to_float=True,
        )
        if len(numpy_frames) > 1:
            raise ValueError(
                f"predict_mask_from_text renders one image; got {len(numpy_frames)}."
            )
        canvas = numpy_frames[0]  # HWC uint8 RGB

        orig_height, orig_width = int(img_tensor.shape[-2]), int(img_tensor.shape[-1])

        # Resize to the backbone's expected resolution.
        image = v2.Resize(size=(self.image_height, self.image_width))(img_tensor)

        # Tokenize text input
        tokenized = self.tokenizer(text_prompts, context_length=self.context_length)

        text_ids = list(range(len(text_prompts)))

        image = image.to(self._device)
        tokenized = tokenized.to(self._device)

        backbone_fpn_0, backbone_fpn_1, backbone_fpn_2 = self.sam3_vision_backbone(
            image
        )

        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        all_class_idx: list[torch.Tensor] = []
        for class_pos, prompt_idx in enumerate(text_ids):
            prompt_tokens = tokenized[prompt_idx : prompt_idx + 1].long()
            pred_boxes_i, scores_i, pred_masks_i = self.sam3_head(
                prompt_tokens, backbone_fpn_0, backbone_fpn_1, backbone_fpn_2
            )
            all_boxes.append(pred_boxes_i)
            all_scores.append(scores_i)
            all_masks.append(pred_masks_i)
            all_class_idx.append(
                torch.full_like(scores_i, fill_value=class_pos, dtype=torch.long)
            )

        # Concatenate along the per-query dim, keeping batch=1.
        pred_boxes = box_ops.box_cxcywh_to_xyxy(torch.cat(all_boxes, dim=1))
        scores = torch.cat(all_scores, dim=1)
        pred_masks = torch.cat(all_masks, dim=1)
        class_idx = torch.cat(all_class_idx, dim=1)

        boxes_out, scores_out, class_idx_out, masks_out = batched_nms(
            self.nms_iou_threshold,
            confidence_threshold,
            pred_boxes,
            scores,
            class_idx,
            pred_masks,
        )
        pred_boxes, scores, class_idx, pred_masks = (
            boxes_out[0],
            scores_out[0],
            class_idx_out[0],
            masks_out[0],
        )
        scale_fct = torch.tensor(
            [orig_width, orig_height, orig_width, orig_height],
            dtype=pred_boxes.dtype,
            device=pred_boxes.device,
        )
        pred_boxes = pred_boxes * scale_fct[None, :]

        # Upscale masks from the head's native output size to the
        # original image; threshold on raw logits.
        pred_masks = interpolate(
            pred_masks.unsqueeze(1),
            (orig_height, orig_width),
            mode="bilinear",
            align_corners=False,
        )
        pred_masks = pred_masks > self.mask_threshold
        labels = [text_prompts[text_ids[idx.item()]] for idx in class_idx]
        return self._render_results(canvas, pred_boxes, pred_masks, scores, labels)

    def _render_results(
        self,
        canvas: np.ndarray,
        pred_boxes: torch.Tensor,
        pred_masks: torch.Tensor,
        scores: torch.Tensor,
        labels: list[str],
    ) -> Image:
        """
        Render predicted masks and bounding boxes onto the input image.

        Parameters
        ----------
        canvas
            Original image as an HWC uint8 RGB numpy array,
            shape ``(orig_height, orig_width, 3)``.
        pred_boxes
            Kept boxes in XYXY pixel coords, shape ``(num_kept, 4)``.
        pred_masks
            Kept binary masks at original resolution,
            shape ``(num_kept, 1, orig_height, orig_width)``.
        scores
            Confidence per kept prediction, shape ``(num_kept,)``.
        labels
            Per-kept-prediction text label, length ``num_kept``.

        Returns
        -------
        Image
            PIL Image with masks and bounding boxes overlaid.
        """
        canvas = np.ascontiguousarray(canvas)

        color_rgb = (30, 144, 255)  # dodgerblue
        mask_alpha = 0.5
        mask_color = np.array(color_rgb, dtype=np.uint8)

        masks_np = pred_masks.squeeze(1).cpu().numpy().astype(bool)
        boxes_np = pred_boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()

        for i in range(len(scores_np)):
            overlay = canvas.copy()
            overlay[masks_np[i]] = mask_color
            cv2.addWeighted(overlay, mask_alpha, canvas, 1 - mask_alpha, 0, canvas)

            box = boxes_np[i]
            draw_box_from_xyxy(
                canvas,
                box[:2],
                box[2:],
                color=color_rgb,
                size=2,
                text=f"{labels[i]}: {scores_np[i]:.2f}",
            )

        return PILImage.fromarray(canvas)
