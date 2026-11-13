# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable

import torch

from qai_hub_models.models.templates.gear_guard_net import BodyDetectionApp
from qai_hub_models.utils.labels import get_class_names

# Define color map for gear guard classes
GEAR_GUARD_COLOR_MAP = {0: (255, 0, 0), 1: (0, 255, 0), -1: (255, 255, 255)}


class GearGuardNetApp(BodyDetectionApp):
    """
    This class consists of light-weight "app code" that is required to perform end to end inference
    with gear_guard_net object detection models.

    Inherits from BodyDetectionApp with customizations for:
    - 2 body/face classes
    - Custom color map for body/face visualization
    - Default model input size of 320x192

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * resize and pad image to match model input size
        * Run model inference
        * if requested, post-process model output using non maximum suppression
        * if requested, draw the predicted bounding boxes on the input image
    """

    COLOR_MAP = GEAR_GUARD_COLOR_MAP

    def __init__(
        self,
        model: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        input_height: int = 320,
        input_width: int = 192,
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
    ) -> None:
        """
        Initialize a GearGuardNetApp application.

        Parameters
        ----------
        model
            gear_guard_net object detection model.

        input_height
            Target height for model input images. Default is 320 for GearGuardNet.

        input_width
            Target width for model input images. Default is 192 for GearGuardNet.

        nms_score_threshold
            Score threshold for non maximum suppression.

        nms_iou_threshold
            Intersection over Union threshold for non maximum suppression.
        """
        super().__init__(
            model,
            input_height,
            input_width,
            nms_score_threshold,
            nms_iou_threshold,
        )

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
            The class name (e.g., "helmet", "vest") or "unknown" for invalid indices.
        """
        class_names = get_class_names("ppe")
        if 0 <= class_idx < len(class_names):
            return class_names[class_idx]
        return "unknown"
