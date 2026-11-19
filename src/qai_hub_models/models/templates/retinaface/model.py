# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec

# RetinaFace is fully convolutional, so the anchor count is a function of input
# size. Pinning the resolution is what makes the graph exportable at all
# (16800 anchors at 640x640).
DEFAULT_DETECTOR_HEIGHT = 640
DEFAULT_DETECTOR_WIDTH = 640

# BGR mean used by the upstream RetinaFace MobileNet. Matches both
# face_detection/alignment.py batch_detect() and the vendored
# RetinafaceDetector.detect_faces(): mean = [104, 117, 123].
RETINA_BGR_MEAN = torch.tensor([104.0, 117.0, 123.0]).view(1, 3, 1, 1)


class RetinaFaceDetectorBase(BaseModel):
    """
    RetinaFace MobileNet face detector, as an exportable CollectionModel component.

    Accepts an RGB image tensor (float32, [0, 1]) and returns the raw network
    outputs: (bbox_regressions, classifications, landmark_regressions). Anchor
    decoding, NMS and score filtering run on the host -- see
    qai_hub_models.models.templates.retinaface.app.postprocess_detections.

    Subclasses supply ``from_pretrained``; the weights differ per recipe even
    though the architecture does not.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.detector_model = model

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run RetinaFace on an RGB image tensor and return raw network outputs.

        Parameters
        ----------
        image
            RGB image tensor of shape (1, 3, H, W), float32, values in [0, 1].

        Returns
        -------
        bbox_regressions : torch.Tensor
            Shape (1, num_anchors, 4) -- raw box regression deltas.
        classifications : torch.Tensor
            Shape (1, num_anchors, 2) -- softmax class scores [background, face].
        landmark_regressions : torch.Tensor
            Shape (1, num_anchors, 10) -- raw landmark regression deltas.
        """
        img_bgr = torch.flip(image, dims=[1]) * 255.0
        mean = RETINA_BGR_MEAN.to(img_bgr.device, dtype=img_bgr.dtype)
        img_bgr = img_bgr - mean
        bbox_regressions, classifications, landmark_regressions = self.detector_model(
            img_bgr
        )
        return bbox_regressions, classifications, landmark_regressions

    def get_input_spec(
        self,
        batch_size: int = 1,
        detector_height: int = DEFAULT_DETECTOR_HEIGHT,
        detector_width: int = DEFAULT_DETECTOR_WIDTH,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, detector_height, detector_width),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "bbox_regressions": TensorSpec(),
            "classifications": TensorSpec(),
            "landmark_regressions": TensorSpec(),
        }
