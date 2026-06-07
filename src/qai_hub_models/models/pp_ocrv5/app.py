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

from qai_hub_models.utils.image_processing import app_to_net_image_inputs

DETECTION_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DETECTION_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ctc_greedy_decode(logits: np.ndarray, charlist: list[str]) -> str:
    """Greedy CTC decode a single region's logits into a text string.

    Parameters
    ----------
    logits
        CTC logits of shape ``[T, vocab]`` (or ``[1, T, vocab]``) where the
        last dimension indexes ``charlist``.
    charlist
        The character list, using the PaddleOCR layout
        ``["<blank>"] + dictionary + [" "]`` so index 0 is the CTC blank.

    Returns
    -------
    text : str
        The decoded text: per-timestep argmax, with consecutive duplicate
        indices collapsed and blank (index 0) dropped, then mapped to chars.
    """
    if logits.ndim == 3:
        logits = logits[0]
    best = logits.argmax(axis=-1)

    chars: list[str] = []
    previous = -1
    for index in best:
        index = int(index)
        # Collapse repeats, then drop the CTC blank (index 0).
        if index not in (previous, 0) and 0 <= index < len(charlist):
            chars.append(charlist[index])
        previous = index
    return "".join(chars)


class PPOCRv5App:
    """
    Light-weight "app code" for end-to-end inference with PP-OCRv5.

    The app uses two models:
        * detector   : produces a per-pixel text probability map
        * recognizer : reads a text-line crop and produces CTC logits

    For a given image, the app will:
        * pre-process the image (resize, ImageNet normalize) for detection
        * run the detector and threshold the probability map into text regions
        * pre-process each detected region for recognition
        * run the recognizer to produce CTC logits per region
        * greedy-CTC-decode the logits into recognized text strings
    """

    def __init__(
        self,
        detector: Callable[[torch.Tensor], torch.Tensor],
        recognizer: Callable[[torch.Tensor], torch.Tensor],
        detector_img_shape: tuple[int, int],
        recognizer_img_shape: tuple[int, int],
        charlist: list[str] | None = None,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.detector_img_shape = detector_img_shape
        self.recognizer_img_shape = recognizer_img_shape
        self.charlist = charlist
        # (scale, new_w, new_h) of the most recent detector letterbox, used to
        # map detected boxes from padded detector space back to the frame.
        self._last_letterbox: tuple[float, int, int] | None = None

    def detector_preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Resize and ImageNet-normalize an RGB uint8 image for detection.

        The image is resized while preserving its aspect ratio and then padded
        ("letterboxed") to the fixed detector input shape. Squishing a wide
        signboard into a square would distort glyphs and degrade the DB text
        map, so the letterbox geometry is stored for mapping detected boxes
        back to the original frame.
        """
        import cv2

        height, width = self.detector_img_shape
        frame_h, frame_w = frame.shape[:2]
        scale = min(width / frame_w, height / frame_h)
        new_w = max(1, round(frame_w * scale))
        new_h = max(1, round(frame_h * scale))
        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        self._last_letterbox = (scale, new_w, new_h)

        normalized = canvas.astype(np.float32) / 255.0
        normalized = (normalized - DETECTION_MEAN) / DETECTION_STD
        chw = normalized.transpose(2, 0, 1)[np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(chw))

    def recognizer_preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Resize a text-line crop (RGB uint8) to the recognizer input shape."""
        import cv2

        height, width = self.recognizer_img_shape
        resized = cv2.resize(crop, (width, height)).astype(np.float32) / 255.0
        chw = resized.transpose(2, 0, 1)[np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(chw))

    def predict(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.predict_text_from_image(*args, **kwargs)

    def predict_text_from_image(
        self,
        pixel_values_or_image: np.ndarray | Image.Image,
        charlist: list[str] | None = None,
    ) -> list[str]:
        """
        Detect text regions in an image and recognize the text in each region.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s), or a numpy array (N H W C or H W C, uint8, RGB).
        charlist
            Optional character list for CTC decoding. Defaults to the list
            supplied at construction time. Must use the PaddleOCR layout
            ``["<blank>"] + dictionary + [" "]``.

        Returns
        -------
        results : list[str]
            One recognized text string per detected text region, across all
            input images, ordered top-to-bottom then left-to-right.
        """
        charlist = charlist if charlist is not None else self.charlist
        if charlist is None:
            raise ValueError(
                "A character list is required to decode recognition logits. "
                "Pass `charlist=...` or construct the app with one "
                "(e.g. recognizer.get_charlist())."
            )

        nhwc_frames, _ = app_to_net_image_inputs(pixel_values_or_image)

        results: list[str] = []
        for frame in nhwc_frames:
            det_input = self.detector_preprocess(frame)
            prob_map = np.asarray(self.detector(det_input))

            for crop in self._extract_text_crops(frame, prob_map):
                rec_input = self.recognizer_preprocess(crop)
                logits = np.asarray(self.recognizer(rec_input))
                results.append(ctc_greedy_decode(logits, charlist))

        return results

    def _extract_text_crops(
        self,
        frame: np.ndarray,
        prob_map: np.ndarray,
        threshold: float = 0.3,
        min_area_frac: float = 1e-4,
    ) -> list[np.ndarray]:
        """Threshold the probability map and crop axis-aligned text regions.

        The DB-style probability map fires per-character; neighbouring
        characters are merged into a single text-line box with a horizontal
        morphological dilation before extracting connected components. Tiny
        components (noise) are dropped, and each box is padded slightly so the
        recognizer sees full glyphs. Regions are returned in reading order
        (top-to-bottom, then left-to-right) so the strings line up with the
        image.
        """
        import cv2

        mask = (prob_map[0, 0] > threshold).astype(np.uint8)
        if mask.sum() == 0:
            return []

        mask_h, mask_w = mask.shape
        # Dilate horizontally to bridge inter-character gaps into text lines.
        kernel_w = max(3, mask_w // 40)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        merged = cv2.dilate(mask, kernel, iterations=1)

        height, width = frame.shape[:2]
        det_h, det_w = self.detector_img_shape

        # Map mask-space -> detector-input space -> original frame, undoing the
        # aspect-preserving letterbox applied in `detector_preprocess`.
        if self._last_letterbox is not None:
            lb_scale, new_w, new_h = self._last_letterbox
        else:
            lb_scale, new_w, new_h = (1.0, det_w, det_h)
        mask_to_det_x = det_w / mask_w
        mask_to_det_y = det_h / mask_h

        min_area = min_area_frac * mask_h * mask_w
        pad_x = max(1, int(0.01 * width))
        pad_y = max(1, int(0.01 * height))

        def to_frame(mx: float, my: float) -> tuple[float, float]:
            # mask -> detector-input pixels, then -> original frame pixels.
            dx = min(mx * mask_to_det_x, new_w)
            dy = min(my * mask_to_det_y, new_h)
            return dx / lb_scale, dy / lb_scale

        contours, _ = cv2.findContours(
            merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            fx0, fy0 = to_frame(x, y)
            fx1, fy1 = to_frame(x + w, y + h)
            x0 = max(0, int(fx0) - pad_x)
            y0 = max(0, int(fy0) - pad_y)
            x1 = min(width, int(fx1) + pad_x)
            y1 = min(height, int(fy1) + pad_y)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            boxes.append((x0, y0, x1, y1))

        # Reading order: group by approximate text-line (top), then by x.
        boxes.sort(key=lambda b: (b[1], b[0]))
        return [frame[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes]
