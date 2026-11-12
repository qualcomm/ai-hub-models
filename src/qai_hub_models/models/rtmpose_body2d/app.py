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
from PIL.Image import Image, fromarray

from qai_hub_models.extern.mmpose import patch_mmpose_no_build_deps
from qai_hub_models.models.templates.corevision.draw import draw_points
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)

with patch_mmpose_no_build_deps():
    from mmpose.codecs.utils import get_simcc_maximum

# Defined the keypoint paires (coco) that from the human pose skeleton
skeleton = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (15, 17),
    (15, 18),
    (15, 19),
    (16, 20),
    (16, 21),
    (16, 22),
    (91, 92),
    (92, 93),
    (93, 94),
    (94, 95),
    (91, 96),
    (96, 97),
    (97, 98),
    (98, 99),
    (91, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    (91, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    (91, 108),
    (108, 109),
    (109, 110),
    (110, 111),
    (112, 113),
    (113, 114),
    (114, 115),
    (115, 116),
    (112, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    (112, 121),
    (121, 122),
    (122, 123),
    (123, 124),
    (112, 125),
    (125, 126),
    (126, 127),
    (127, 128),
    (112, 129),
    (129, 130),
    (130, 131),
    (131, 132),
]


def add_skeleton_edges(
    img: np.ndarray,
    points: list[tuple[int, int]],
    skeleton: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> None:
    """Draws Lines connecting specified keypoiny pairs to from a skeleton."""
    for p1, p2 in skeleton:
        x1, y1 = points[p1]
        x2, y2 = points[p2]
        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=thickness)


class RTMPosebody2dApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with RTMPose.

    The app uses 1 model:
        * RTMPose

    For a given image input, the app will:
        * pre-process the image
        * Run RTMPose inference
        * Convert the output into a list of keypoint coordiates
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        inferencer: Any,
    ) -> None:
        self.model = model
        self.inferencer = inferencer

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image]:
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def decode_output(
        self, pred_x: np.ndarray, pred_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from SimCC representations. The decoded
        coordinates are in the input image space.

        Parameters
        ----------
        pred_x
            SimCC label for x-axis.
        pred_y
            SimCC label for y-axis.

        Returns
        -------
        keypoints : np.ndarray
            Decoded coordinates in shape (N, K, D).
        scores : np.ndarray
            The keypoint scores in shape (N, K).
            It usually represents the confidence of the keypoint prediction.
        """
        keypoints, scores = get_simcc_maximum(pred_x, pred_y)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]
        keypoints /= 2.0

        return keypoints, scores

    def predict_pose_keypoints(
        self,
        pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image],
        raw_output: bool = False,
    ) -> np.ndarray | list[Image]:
        """
        Predicts pose keypoints for a person in the image.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout
        raw_output
            See "returns" doc section for details.

        Returns
        -------
        np.ndarray | list[Image]
            If raw_output is True, returns:
            keypoints
                np.ndarray, shape [B, N, 2].
                Numpy array of keypoints within the images. Each keypoint is an (x, y) pair of coordinates within the image.

            If raw_output is False, returns:
            predicted_images
                list[PIL.Image].
                Images with keypoints drawn.
        """
        # Preprocess image to get data required for post processing
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)
        inputs = self.inferencer.preprocess(NHWC_int_numpy_frames, batch_size=1)
        proc_inputs, _ = next(iter(inputs))
        proc_inputs_ = proc_inputs["inputs"][0]
        x = proc_inputs_[[2, 1, 0]]
        # Convert to expected model input distrubtion
        # Add batch dimension
        x = torch.unsqueeze(x, 0)
        x = x.to(dtype=torch.float32)
        pred_x, pred_y = self.model(x)
        keypoints, _scores = self.decode_output(pred_x.numpy(), pred_y.numpy())

        # center and scale to transform the coordinates back to the original image
        bbox = proc_inputs["data_samples"][0].gt_instances.bboxes[0]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        scale = proc_inputs["data_samples"][0].gt_instances.bbox_scales[0]
        # map the predicted coordinate back to the original input imag
        input_size = proc_inputs["data_samples"][0].metainfo["input_size"]
        keypoints = keypoints / input_size * scale + center - 0.5 * scale
        keypoints = np.round(keypoints).astype(np.int32)

        if raw_output:
            return keypoints
        predicted_images = []

        # draw keypoints and skeleton
        for i, img in enumerate(NHWC_int_numpy_frames):
            add_skeleton_edges(
                img, keypoints[i], skeleton, color=(255, 0, 255), thickness=2
            )
            draw_points(img, keypoints[i], color=(0, 255, 255), size=4)
            predicted_images.append(fromarray(img))
        return predicted_images
