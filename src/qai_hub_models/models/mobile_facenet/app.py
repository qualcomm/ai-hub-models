# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from collections.abc import Callable

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.align_faces import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.facenet_utils import (
    select_significant_face,
)
from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.retinaface.detector import (
    detector,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    numpy_image_to_torch,
)


class MobileFaceNetApp:
    """
    Application wrapper for MobileFaceNet face verification.

    Given two face image paths, this app detects and aligns each face,
    computes a 128-dimensional L2-normalised embedding for each using the
    MobileFaceNet backbone, and returns a same/different verdict by comparing
    the cosine angle between the two embeddings against a configurable threshold.

    Detection and alignment use RetinaFace to locate the most prominent face
    in each image and warp it to a canonical 112x112 crop. Each crop is
    embedded twice (original + horizontal flip) and the two results are
    summed before normalisation (test-time augmentation).
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        input_height: int,
        input_width: int,
        threshold: float,
    ) -> None:
        self.model = model
        self.input_height = input_height
        self.input_width = input_width
        self.threshold = threshold
        self.device = "cpu"
        self.default_square = True
        self.inner_padding_factor = 0.25
        self.outer_padding = (0, 0)

    def predict_features(
        self,
        img1: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        img2: torch.Tensor | np.ndarray | Image.Image | list[Image.Image],
        already_aligned: bool = False,
    ) -> tuple[bool, float, float]:
        """
        Verify whether two face images depict the same person.

        Both images are decoded, optionally detected and aligned via RetinaFace,
        then passed together in a single model call. The model applies TTA
        (original + horizontal flip) internally for each image and returns
        interleaved embeddings. The cosine angle between the two embeddings is
        compared against ``self.threshold`` to produce a same/different verdict.

        Parameters
        ----------
        img1
            First face image. Accepts a ``torch.Tensor``, ``np.ndarray``,
            ``PIL.Image``, or a list of ``PIL.Image`` objects (any resolution).
        img2
            Second face image. Accepts a ``torch.Tensor``, ``np.ndarray``,
            ``PIL.Image``, or a list of ``PIL.Image`` objects (any resolution).
        already_aligned
            If ``True``, skip face detection and alignment — each image is
            resized directly to ``(input_height, input_width)``. Set this when
            the inputs are pre-cropped face chips.

        Returns
        -------
        same : bool
            ``True`` if the cosine angle is below ``self.threshold`` (same person).
        angle : float
            Cosine angle in degrees between the two embeddings.
            Range ``[0°, 180°]``; lower means more similar.
        cosine : float
            Raw cosine similarity in ``[-1, 1]``; higher means more similar.
        """
        img1_bgr = app_to_net_image_inputs(img1, image_layout="BGR", to_float=False)[0][
            0
        ]
        img2_bgr = app_to_net_image_inputs(img2, image_layout="BGR", to_float=False)[0][
            0
        ]

        if already_aligned:
            img1_bgr = cv.resize(img1_bgr, (self.input_width, self.input_height))
            img2_bgr = cv.resize(img2_bgr, (self.input_width, self.input_height))
        else:
            img1_bgr = self.detect_and_align(img1_bgr)
            img2_bgr = self.detect_and_align(img2_bgr)

        img1_t = numpy_image_to_torch(img1_bgr[..., ::-1].copy())
        img2_t = numpy_image_to_torch(img2_bgr[..., ::-1].copy())
        output = self.model(img1_t, img2_t)  # (2, 128): [0]=emb1, [1]=emb2

        emb1 = output[0].cpu().detach().numpy()
        emb2 = output[1].cpu().detach().numpy()
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)
        cosine = float(np.clip(np.dot(emb1, emb2), -1.0, 1.0))
        angle = math.acos(cosine) * 180.0 / math.pi
        return angle < self.threshold, angle, cosine

    def align_face(self, raw: np.ndarray, facial5points: np.ndarray) -> np.ndarray:
        """
        Warp and crop a face image to a canonical ``(input_height, input_width)`` BGR crop.

        Parameters
        ----------
        raw
            Full-resolution source image as a ``uint8`` BGR NumPy array.
        facial5points
            Five 2-D facial landmark coordinates as a flat or ``(2, 5)`` array
            in the order: left-eye, right-eye, nose, left-mouth, right-mouth.

        Returns
        -------
        crop : np.ndarray
            Aligned face crop of shape ``(input_height, input_width, 3)`` in BGR.
        """
        facial5points = np.reshape(facial5points, (2, 5))

        crop_size = (self.input_height, self.input_width)
        output_size = (self.input_height, self.input_width)

        # get the reference 5 landmarks position in the crop settings
        reference_5pts = get_reference_facial_points(
            output_size,
            self.inner_padding_factor,
            self.outer_padding,
            self.default_square,
        )
        return warp_and_crop_face(
            raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size
        )

    def detect_and_align(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the most prominent face in ``image`` and return an aligned BGR crop.

        Uses RetinaFace to locate all faces, selects the most significant one via
        ``select_significant_face``, then warps the five detected landmarks to a
        canonical reference frame via ``align_face``.

        Parameters
        ----------
        image
            Input image as a ``uint8`` BGR NumPy array of any resolution.

        Returns
        -------
        crop : np.ndarray
            Aligned face crop of shape ``(input_height, input_width, 3)`` in BGR.

        """
        bounding_boxes, landmarks = detector.detect_faces(image)
        if len(landmarks) == 0:
            raise RuntimeError(
                "No face detected in the input image\n"
                "If the image is already a cropped face, pass --already_aligned to skip detection."
            )
        i = select_significant_face(bounding_boxes)
        return self.align_face(image, landmarks[i])
