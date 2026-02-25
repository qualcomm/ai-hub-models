# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.utils.bounding_box_processing import batched_nms
from qai_hub_models.utils.draw import draw_box_from_xyxy
from qai_hub_models.utils.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)


class BodyDetectionApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with gear_guard_net object detection models.

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * resize and pad image to match model input size
        * Run model inference
        * if requested, post-process model output using non maximum suppression
        * if requested, draw the predicted bounding boxes on the input image
    """

    # Subclasses should override this with their own color map
    COLOR_MAP = {-1: (255, 255, 255)}

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        input_height: int = 320,
        input_width: int = 192,
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        pad_value: float = 0.0,
    ) -> None:
        """
        Initialize a BodyDetectionApp application.

        Parameters
        ----------
        model
            gear_guard_net object detection model.

        input_height
            Target height (pixel) for model input images. Default is 320 for GearGuardNet.

        input_width
            Target width (pixel) for model input images. Default is 192 for GearGuardNet.

        nms_score_threshold
            Score threshold for non maximum suppression. Range [0, 1]

        nms_iou_threshold
            Intersection over Union threshold for non maximum suppression.
            Range [0,1]

        pad_value
            Padding value used when resizing images to model input size.
            Should match the padding value used during model training.
            Default is 0.0 (black). Use 114/255.0 for YOLO-style gray padding.

        """
        self.model = model
        self.input_height = input_height
        self.input_width = input_width
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pad_value = pad_value

    def predict(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[np.ndarray]
    ):
        # See predict_boxes_from_image.
        return self.predict_boxes_from_image(*args, **kwargs)

    def predict_boxes_from_image(
        self,
        pixel_values_or_image: (torch.Tensor | np.ndarray | Image.Image),
        raw_output: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[np.ndarray]
    ):
        """
        From the provided image or tensor, predict the bounding boxes & classes of objects detected within.

        Parameters
        ----------
        pixel_values_or_image
            either one of below:
            - pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout
            - numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            - PIL image

        raw_output
            See "returns" doc section for details.

        Returns
        -------
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[np.ndarray]
            If raw_output is true, returns a tuple of (boxes, scores, class_idx):
                - boxes : list[torch.Tensor]
                    Bounding box coordinates (x1, y1, x2, y2) per batch.
                    List element shape (num_preds, 4), List length = N
                - scores : list[torch.Tensor]
                    Class scores per batch multiplied by confidence, range [0, 1].
                    List element shape (num_preds, num_classes), List length = N
                - class_idx : list[torch.tensor]
                    Indices of the most probable class of the prediction.
                    List element shape (num_preds), List length = N

            Otherwise, returns:
                - images: list[np.ndarray]
                    A list of predicted RGB, [H, W, C] images (one list element per batch).
                    Each image will have bounding boxes drawn. List length = N
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        # Resize to fit model input
        NCHW_fp32_torch_resized_frames, scale, pad = resize_pad(
            NCHW_fp32_torch_frames,
            (self.input_height, self.input_width),
            pad_value=self.pad_value,
        )

        # Run prediction
        pred_boxes, pred_scores, pred_class_idx = self.model(
            NCHW_fp32_torch_resized_frames
        )

        # Non Maximum Suppression on each batch
        pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx = (
            batched_nms(
                self.nms_iou_threshold,
                self.nms_score_threshold,
                pred_boxes,
                pred_scores,
                pred_class_idx,
            )
        )

        # Return raw output if requested
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return (pred_post_nms_boxes, pred_post_nms_scores, pred_post_nms_class_idx)

        # Add boxes to each batch with colors based on object type
        for batch_idx in range(len(pred_post_nms_boxes)):
            pred_boxes_batch = pred_post_nms_boxes[batch_idx]
            pred_class_idx_batch = pred_post_nms_class_idx[batch_idx]

            # Transform bounding boxes back to original image size
            pred_boxes_batch = self._transform_boxes_to_original_size(
                pred_boxes_batch,
                pad,
                scale,
                NCHW_fp32_torch_frames.shape[2],  # height
                NCHW_fp32_torch_frames.shape[3],  # width
            )

            for i, box in enumerate(pred_boxes_batch):
                class_idx = int(pred_class_idx_batch[i].item())
                _color = self.COLOR_MAP.get(class_idx, self.COLOR_MAP[-1])

                # Get label for the box (subclasses can override _get_box_label)
                label = self._get_box_label(class_idx)

                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[batch_idx],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=_color,
                    size=2,
                    text=label,
                )

        return NHWC_int_numpy_frames

    def _get_box_label(self, class_idx: int) -> str | None:
        """
        Get the label text for a detected object.

        Parameters
        ----------
        class_idx
            The class index of the detected object.

        Returns
        -------
        label : str | None
            The label text to display on the bounding box, or None to display no label.
            Subclasses can override this method to provide custom labels.
        """
        return None

    @staticmethod
    def _transform_boxes_to_original_size(
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
            boxes[:, 0] = (boxes[:, 0] - pad[0]) / scale  # x1
            boxes[:, 1] = (boxes[:, 1] - pad[1]) / scale  # y1
            boxes[:, 2] = (boxes[:, 2] - pad[0]) / scale  # x2
            boxes[:, 3] = (boxes[:, 3] - pad[1]) / scale  # y2

            # Ensure boxes are within image boundaries
            boxes[:, 0] = torch.clamp(boxes[:, 0], min=0)
            boxes[:, 1] = torch.clamp(boxes[:, 1], min=0)
            boxes[:, 2] = torch.clamp(boxes[:, 2], max=original_width)
            boxes[:, 3] = torch.clamp(boxes[:, 3], max=original_height)

        return boxes
