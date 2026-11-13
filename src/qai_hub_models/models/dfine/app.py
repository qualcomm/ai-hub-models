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
from qai_hub_models.utils.labels import get_class_names


class DFineApp:
    """
    Light-weight "app code" for end-to-end inference with D-FINE.

    For a given image input, the app will:
        * Resize the image to the model's input resolution.
        * Run D-FINE inference (boxes, scores, class indices).
        * Threshold detections and draw the predicted boxes / 80-class COCO
          labels on the image.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        model_image_height: int = 640,
        model_image_width: int = 640,
        threshold: float = 0.3,
    ) -> None:
        self.model = model
        self.model_image_height = model_image_height
        self.model_image_width = model_image_width
        self.threshold = threshold

    def predict(
        self,
        image: Image,
    ) -> tuple[list[npt.NDArray[np.uint8]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect objects in the provided image.

        Parameters
        ----------
        image
            A PIL Image in RGB format.

        Returns
        -------
        numpy_array : list[npt.NDArray[np.uint8]]
            Original image (per batch) with predicted boxes drawn.
        scores : torch.Tensor
            Confidence scores for detections above the threshold.
        labels : torch.Tensor
            Class indices for detections above the threshold.
        boxes : torch.Tensor
            Bounding boxes (x1, y1, x2, y2) for detections above the threshold.
        """
        image = image.resize(
            (self.model_image_width, self.model_image_height),
            resample=Resampling.BILINEAR,
        )

        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(image)

        pred_boxes, pred_scores, pred_class_idx = self.model(
            preprocess_PIL_image(image)
        )

        mask = pred_scores > self.threshold
        boxes_per_batch = [pred_boxes[i][mask[i]] for i in range(pred_boxes.shape[0])]
        labels_per_batch = [
            pred_class_idx[i][mask[i]] for i in range(pred_class_idx.shape[0])
        ]
        boxes = torch.cat(boxes_per_batch, dim=0)
        labels = torch.cat(labels_per_batch, dim=0)

        coco_labels = get_class_names("coco")
        for batch_idx in range(len(NHWC_int_numpy_frames)):
            for box, label in zip(
                boxes_per_batch[batch_idx], labels_per_batch[batch_idx], strict=False
            ):
                draw_box_from_xyxy(
                    NHWC_int_numpy_frames[batch_idx],
                    box[0:2].int(),
                    box[2:4].int(),
                    color=(0, 255, 0),
                    size=2,
                    text=coco_labels[int(label.item())],
                )

        return NHWC_int_numpy_frames, pred_scores[mask], labels, boxes
