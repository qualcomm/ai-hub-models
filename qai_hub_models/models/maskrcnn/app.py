# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection.roi_heads import paste_masks_in_image

from qai_hub_models.models._shared.proposal_based_detection.app import (
    ProposalBasedDetectionApp,
)
from qai_hub_models.utils.bounding_box_processing import batched_nms
from qai_hub_models.utils.draw import create_color_map, draw_box_from_xyxy
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    denormalize_coordinates,
    resize_pad,
    undo_resize_pad,
)
from qai_hub_models.utils.path_helpers import QAIHM_PACKAGE_ROOT


class MaskRCNNApp(ProposalBasedDetectionApp):
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with MaskRCNN.

    For a given image input, the app will:
        * Preprocess the image (normalize, resize, etc).
        * Run MaskRCNN Inference
        * Convert the raw output into box coordinates, masks, and corresponding label and confidence.
        * Return numpy image with boxes and masks overlaid.
    """

    def __init__(
        self,
        proposal_generator: Callable[
            [torch.Tensor],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        roi_head: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        model_image_height: int = 800,
        model_image_width: int = 800,
        proposal_iou_threshold: float = 0.7,
        boxes_iou_threshold: float = 0.5,
        boxes_score_threshold: float = 0.5,
        mask_threshold: float = 0.05,
        max_det_pre_nms: int = 6000,
        max_det_post_nms: int = 200,
        max_vis_boxes: int = 100,
    ) -> None:
        """
        Initialize MaskRCNNApp.

        Parameters
        ----------
        proposal_generator
            Callable that generates proposals from image tensor.
        roi_head
            Callable that processes ROI features and proposals.
        model_image_height
            Height of model input images.
        model_image_width
            Width of model input images.
        proposal_iou_threshold
            IOU threshold for proposal filtering.
        boxes_iou_threshold
            IOU threshold for NMS on boxes.
        boxes_score_threshold
            Score threshold for filtering boxes.
        mask_threshold
            Threshold for binarizing masks.
        max_det_pre_nms
            Maximum detections before NMS.
        max_det_post_nms
            Maximum detections after NMS.
        max_vis_boxes
            Maximum boxes to visualize.
        """
        super().__init__(
            model_image_height,
            model_image_width,
            proposal_iou_threshold,
            boxes_iou_threshold,
            boxes_score_threshold,
            max_det_pre_nms,
            max_det_post_nms,
        )
        self.proposal_generator = proposal_generator
        self.roi_head = roi_head
        self.max_vis_boxes = max_vis_boxes
        self.mask_threshold = mask_threshold
        self.num_classes = 91
        with open(QAIHM_PACKAGE_ROOT / "labels" / "coco_labels_91.txt") as f:
            self.labels_list = [line.strip() for line in f]

    def predict(
        self,
        images: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | list[Image.Image]
    ):
        """
        From the provided image, detect the objects and segment them.

        Parameters
        ----------
        images
            PIL image, numpy array (N H W C x uint8) or (H W C x uint8) with RGB channel layout,
            or pyTorch tensor (N C H W x fp32, value range is [0, 1]) with RGB channel layout.
        raw_output
            If true, returns raw detection outputs (boxes, scores, class indices, masks).
            Otherwise, returns annotated images.

        Returns
        -------
        result : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[Image.Image]
            If raw_output is True:
                boxes
                    List of bounding box tensors per batch. Each tensor has shape [num preds, 4]
                    where 4 represents (x1, y1, x2, y2).
                scores
                    List of class score tensors per batch multiplied by confidence.
                    Each tensor has shape [num_preds].
                class_idx
                    List of class index tensors. Each tensor has shape [num_preds] with values
                    indicating the most probable class.
                masks
                    List of mask tensors per batch. Each tensor has shape [num_preds, H, W].
            If raw_output is False:
                List of PIL images with detected objects and masks drawn and labeled.
        """
        NHWC_int_numpy_frames, image_tensor = app_to_net_image_inputs(images)

        image_tensor, scale_factor, pad = resize_pad(
            image_tensor, (self.model_image_height, self.model_image_width)
        )

        # Run Proposal Generator
        features_0, features_1, features_2, features_3, proposals, objectness_logits = (
            self.proposal_generator(image_tensor)
        )

        # Filter proposals with NMS
        padded_proposals = self.filter_proposals([proposals], [objectness_logits])[0]

        # Normalize proposals to [0, 1] range based on model input dimensions
        normalized_proposals = padded_proposals / torch.tensor(
            [
                self.model_image_width,
                self.model_image_height,
                self.model_image_width,
                self.model_image_height,
            ],
            dtype=padded_proposals.dtype,
            device=padded_proposals.device,
        )

        # Run ROI Head
        boxes_tensor, scores_tensor, classes_tensor, masks_tensor = self.roi_head(
            features_0, features_1, features_2, features_3, normalized_proposals
        )

        # Paste masks to full resolution for each batch
        batch_size = boxes_tensor.shape[0]
        pasted_masks_list = []
        for batch_idx in range(batch_size):
            pasted_masks = paste_masks_in_image(
                masks_tensor[batch_idx].unsqueeze(1),
                boxes_tensor[batch_idx],
                (self.model_image_height, self.model_image_width),
                padding=1,  # 1-pixel border to prevent edge artifacts
            )
            pasted_masks_list.append(pasted_masks)
        masks_tensor = torch.stack(pasted_masks_list)

        # Apply NMS with masks as additional argument
        batched_boxes, batched_scores, batched_classes, batched_masks = batched_nms(
            self.boxes_iou_threshold,
            self.boxes_score_threshold,
            boxes_tensor.float(),
            scores_tensor.float(),
            classes_tensor,
            masks_tensor,
        )

        # Visualization
        color = create_color_map(self.num_classes + 1)

        out_images = []
        for i, (boxes, scores, labels, masks) in enumerate(
            zip(
                batched_boxes,
                batched_scores,
                batched_classes,
                batched_masks,
                strict=True,
            )
        ):
            h, w, _ = NHWC_int_numpy_frames[i].shape

            boxes = boxes[: self.max_vis_boxes]
            scores = scores[: self.max_vis_boxes]
            labels = labels[: self.max_vis_boxes]
            masks = masks[: self.max_vis_boxes] if len(masks) > 0 else masks

            denormalize_coordinates(boxes.view(-1, 2, 2), (1, 1), scale_factor, pad)

            batched_boxes[i] = boxes
            batched_scores[i] = scores
            batched_classes[i] = labels

            combined_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
            output_array = NHWC_int_numpy_frames[i].copy()
            masks_resized_list = []

            for box, score, label, mask in zip(
                boxes, scores, labels, masks, strict=True
            ):
                label_idx = int(label)
                label_color = color[label_idx]
                label_name = self.labels_list[label_idx]

                mask_resized = undo_resize_pad(
                    mask.unsqueeze(0).float(), (w, h), scale_factor, pad
                ).squeeze()
                masks_resized_list.append(mask_resized)

                mask_binary = (mask_resized > self.mask_threshold).cpu().numpy()
                combined_mask[mask_binary] = label_color

                x1, y1, x2, y2 = box.cpu().numpy()
                draw_box_from_xyxy(
                    output_array,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    tuple(label_color.tolist()),
                    size=2,
                    text=f"{label_name}: {score:.2f}",
                )

            batched_masks[i] = torch.stack(masks_resized_list)

            # Blend once with the combined mask
            output_image = Image.blend(
                Image.fromarray(output_array),
                Image.fromarray(combined_mask),
                alpha=0.3,
            )

            out_images.append(output_image)

        if raw_output:
            return batched_boxes, batched_scores, batched_classes, batched_masks

        return out_images
