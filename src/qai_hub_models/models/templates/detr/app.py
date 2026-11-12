# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch
from PIL.Image import Image, Resampling

from qai_hub_models.models.templates.corevision.draw import draw_box_from_xyxy
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    preprocess_PIL_image,
)
from qai_hub_models.models.templates.detr.coco_label_map import LABEL_MAP


class DETRApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with DETR.

    For a given image input, the app will:
        * Preprocess the image (normalize, resize, etc) and get encoding to pass to the model.
        * Run DETR Inference
        * Convert the raw output into box coordinates and corresponding label and confidence.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        model_image_height: int | None = None,
        model_image_width: int | None = None,
    ) -> None:
        self.model = model
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width

    def _process_boxes_scores_classes(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_class_idx: torch.Tensor,
        NHWC_int_numpy_frames: list[npt.NDArray[np.uint8]],
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process outputs for (pred_boxes, pred_scores, pred_class_idx) format.

        Parameters
        ----------
        pred_boxes
            Predicted bounding boxes.
        pred_scores
            Predicted scores.
        pred_class_idx
            Predicted class indices.
        NHWC_int_numpy_frames
            Image in NHWC format as a list of numpy arrays.
        threshold
            Prediction score threshold.

        Returns
        -------
        scores : torch.Tensor
            Confidence scores for the predicted class.
        labels : torch.Tensor
            Labels (class number) for the predicted class.
        boxes : torch.Tensor
            Bounding boxes for the predicted class.
        """
        mask = pred_scores > threshold
        pred_boxes = torch.cat(
            [pred_boxes[i][mask[i]] for i in range(pred_boxes.shape[0])], dim=0
        )
        pred_class_idx = torch.cat(
            [pred_class_idx[i][mask[i]] for i in range(pred_class_idx.shape[0])], dim=0
        )

        # Add boxes to each batch
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            pred_boxes_batch = pred_boxes
            pred_class_idx_batch = pred_class_idx
            if len(pred_boxes_batch.shape) > 0 and len(pred_class_idx_batch.shape) > 0:
                for box, label in zip(
                    pred_boxes_batch, pred_class_idx_batch, strict=False
                ):
                    draw_box_from_xyxy(
                        NHWC_int_numpy_frames[batch_idx],
                        box[0:2].int(),
                        box[2:4].int(),
                        color=(0, 255, 0),
                        size=2,
                        text=f"{LABEL_MAP[int(label.item())]}",
                    )

        return pred_scores[mask], pred_class_idx, pred_boxes

    def predict(
        self,
        image: Image,
        threshold: float = 0.9,
    ) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From the provided image or tensor, generate the segmented mask.

        Parameters
        ----------
        image
            A PIL Image in NCHW, RGB format.
        threshold
            Prediction score threshold.

        Returns
        -------
        numpy_array : list[npt.NDArray[np.uint8]]
            Original image numpy array with the corresponding predictions.
        scores : torch.Tensor
            Confidence scores for the predicted class.
            Shape is [Number of predictions above threshold].
        labels : torch.Tensor
            Labels (class number) for the predicted class.
            Shape is [Number of predictions above threshold].
        boxes : torch.Tensor
            Bounding boxes for the predicted class.
            Shape is [Number of predictions above threshold, 4].
        """
        # The official detr demo uses resize instead of padding. There is an option
        # to do padding instead and pass a pixel mask to the model, but we opted for
        # the simpler route.
        # https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=ZRluxbQYYTEe
        if self.model_image_height is not None and self.model_image_width is not None:
            image = image.resize(
                (self.model_image_width, self.model_image_height),
                resample=Resampling.BILINEAR,
            )

        # Convert image to numpy array for drawing
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(image)

        # Run DETR Inference
        outputs = self.model(preprocess_PIL_image(image))

        # Process outputs for (pred_boxes, pred_scores, pred_class_idx) format
        scores, labels, boxes = self._process_boxes_scores_classes(
            outputs[0], outputs[1], outputs[2], NHWC_int_numpy_frames, threshold
        )

        return NHWC_int_numpy_frames, scores, labels, boxes
