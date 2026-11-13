# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.superpoint.model import DEFAULT_HEIGHT, DEFAULT_WIDTH
from qai_hub_models.models.superpoint.postprocess import (
    extract_keypoints,
    match_descriptors,
)
from qai_hub_models.models.templates.corevision.draw import (
    draw_connections,
    draw_points,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.input_spec import InputSpec

DEFAULT_DETECTION_THRESHOLD = 0.001
DEFAULT_TOP_K = 1000
DEFAULT_STRIDE = 8
DEFAULT_NMS_RADIUS = 4
DEFAULT_REMOVE_BORDERS = 4
DEFAULT_MAX_VISUALIZED_MATCHES = 80


class SuperPointApp:
    """
    Application wrapper for SuperPoint keypoint detection + matching.

    Given two images of the same scene, this app resizes both to the model's
    input resolution, runs the model's dense (scores, descriptors) forward
    pass, extracts a variable-length set of keypoints per image via
    ``postprocess.extract_keypoints``, and mutually matches their descriptors
    via ``postprocess.match_descriptors``.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        input_spec: InputSpec | None = None,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        stride: int = DEFAULT_STRIDE,
        nms_radius: int = DEFAULT_NMS_RADIUS,
        remove_borders: int = DEFAULT_REMOVE_BORDERS,
        max_visualized_matches: int = DEFAULT_MAX_VISUALIZED_MATCHES,
    ) -> None:
        self.model = model
        if input_spec is not None:
            _, _, self.height, self.width = input_spec["image_a"][0]
        else:
            self.height, self.width = DEFAULT_HEIGHT, DEFAULT_WIDTH
        self.detection_threshold = detection_threshold
        self.top_k = top_k
        self.stride = stride
        self.nms_radius = nms_radius
        self.remove_borders = remove_borders
        self.max_visualized_matches = max_visualized_matches

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> np.ndarray | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # See predict_matches.
        return self.predict_matches(*args, **kwargs)

    def predict_matches(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect and mutually match SuperPoint keypoints between two images.

        Parameters
        ----------
        image_a
            First image, any resolution/mode; resized to
            ``(self.height, self.width)`` before inference.
        image_b
            Second image, any resolution/mode; resized to
            ``(self.height, self.width)`` before inference.
        raw_output
            If ``True``, return the raw keypoints/matches tensors instead of
            an annotated visualization.

        Returns
        -------
        result : np.ndarray | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            If ``raw_output``: ``(keypoints_a, keypoints_b, matches)``, where
            ``keypoints_a``/``keypoints_b`` are ``(N, 2)`` ``(x, y)`` pixel
            coordinates in each image's own resized frame, and ``matches`` is
            ``(M, 2)`` of ``(index_into_a, index_into_b)`` pairs.
            Otherwise: an RGB ``uint8`` numpy array with ``image_a`` and
            ``image_b`` placed side by side, detected keypoints marked, and
            mutual matches drawn as connecting lines.
        """
        image_a_resized = image_a.convert("RGB").resize(
            (self.width, self.height), Image.BILINEAR
        )
        image_b_resized = image_b.convert("RGB").resize(
            (self.width, self.height), Image.BILINEAR
        )

        frames_a, tensor_a = app_to_net_image_inputs(image_a_resized)
        frames_b, tensor_b = app_to_net_image_inputs(image_b_resized)

        scores_a, descriptors_a, scores_b, descriptors_b = self.model(
            tensor_a, tensor_b
        )

        kp_a, _, d_a = extract_keypoints(
            scores_a[0].cpu(),
            descriptors_a[0].cpu(),
            self.detection_threshold,
            self.top_k,
            self.stride,
            self.nms_radius,
            self.remove_borders,
        )
        kp_b, _, d_b = extract_keypoints(
            scores_b[0].cpu(),
            descriptors_b[0].cpu(),
            self.detection_threshold,
            self.top_k,
            self.stride,
            self.nms_radius,
            self.remove_borders,
        )

        matches = match_descriptors(d_a, d_b)

        if raw_output:
            return kp_a, kp_b, matches

        frame_a, frame_b = frames_a[0], frames_b[0]
        canvas = np.ascontiguousarray(np.concatenate([frame_a, frame_b], axis=1))
        offset = frame_a.shape[1]

        kp_a_np = kp_a.numpy()
        kp_b_np = kp_b.numpy() + np.array([offset, 0], dtype=np.float32)

        if matches.shape[0] > 0:
            similarity = (d_a[matches[:, 0]] * d_b[matches[:, 1]]).sum(dim=1)
            k = min(self.max_visualized_matches, similarity.shape[0])
            keep = torch.topk(similarity, k).indices
            shown_matches = matches[keep]
            matched_a = kp_a_np[shown_matches[:, 0].numpy()]
            matched_b = kp_b_np[shown_matches[:, 1].numpy()]
            point_pairs = np.stack([matched_a, matched_b], axis=1)
            draw_connections(canvas, point_pairs, color=(255, 0, 0), size=1)

        draw_points(canvas, kp_a_np, (0, 255, 0), 3)
        draw_points(canvas, kp_b_np, (0, 255, 0), 3)

        return canvas
