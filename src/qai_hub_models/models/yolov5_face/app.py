# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
    box_xywh_to_xyxy,
)
from qai_hub_models.models.templates.corevision.draw import (
    draw_box_from_xyxy,
    draw_points,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.utils.input_spec import InputSpec

# Pad value (Neutral Gray) used by the YOLOv5-Face letterbox (must match model preprocessing)
PAD_VALUE = 114.0 / 255.0

# 5-point facial landmark colours  (left-eye, right-eye, nose, left-mouth, right-mouth)
_LM_COLORS = [
    (255, 0, 0),  # left eye   - red
    (0, 255, 0),  # right eye  - green
    (0, 0, 255),  # nose       - blue
    (255, 255, 0),  # left mouth - yellow
    (255, 0, 255),  # right mouth - magenta
]


class YoloV5FaceApp:
    """
    End-to-end application for YoloV5-Face.

    Handles:
      * letterbox preprocessing (resize-pad to 640x640, pad value 114/255)
      * model inference
      * NMS in 640x640 space
      * coordinate rescaling back to original image space
      * visualisation (bounding boxes + 5 facial landmarks)
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        conf_threshold: float = 0.6,
        iou_threshold: float = 0.5,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        if input_spec is not None:
            _, _, h, w = input_spec["image"][0]
            self.model_image_input_shape: tuple[int, int] | None = (h, w)
        else:
            self.model_image_input_shape = None

    def predict(self, *args: Any, **kwargs: Any) -> tuple[list[dict], Image.Image]:
        return self.run_inference_on_image(*args, **kwargs)

    @overload
    def run_inference_on_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: Literal[False] = ...,
    ) -> tuple[list[dict], Image.Image]: ...

    @overload
    def run_inference_on_image(
        self,
        pixel_values_or_image: torch.Tensor
        | np.ndarray
        | Image.Image
        | list[Image.Image],
        raw_output: Literal[True],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]: ...

    def run_inference_on_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> (
        tuple[list[dict], Image.Image]
        | tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
    ):
        """
        Detect faces and their 5 facial landmarks in the input image.

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
        output : tuple[list[dict], Image.Image] | tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
            If raw_output is True, returns:
                boxes : list[torch.Tensor]
                    Post-NMS boxes in 640x640 letterbox space, shape [N, 4] (x1,y1,x2,y2).
                scores : list[torch.Tensor]
                    Post-NMS scores, shape [N].
                landmarks : list[torch.Tensor]
                    Post-NMS landmarks in 640x640 letterbox space, shape [N, 10].

            If raw_output is False, returns:
                detections : list[dict]
                    Each dict contains:
                    'box'       - [x1, y1, x2, y2] in original image pixels
                    'score'     - float confidence
                    'landmarks' - list of 5 (x, y) tuples in original image pixels
                annotated_image : Image.Image
                    Original image with bounding boxes and landmark dots drawn.
        """
        [np_img], image_tensor = app_to_net_image_inputs(pixel_values_or_image)
        orig_h, orig_w = np_img.shape[:2]

        # Defaults used when model_image_input_shape is None (no letterboxing).
        scale: float = 1.0
        pad_left: float = 0.0
        pad_top: float = 0.0

        if self.model_image_input_shape is not None:
            image_tensor, scale, (pad_left, pad_top) = resize_pad(
                image_tensor,
                dst_size=self.model_image_input_shape,
                pad_value=PAD_VALUE,
            )

        boxes_raw, scores_raw, landmarks_raw = self.model(image_tensor)

        boxes_raw = boxes_raw[0]
        scores_flat = scores_raw[0, :, 0]
        landmarks_raw = landmarks_raw[0]

        # Convert cx,cy,w,h → x1,y1,x2,y2 then NMS
        xyxy_all = box_xywh_to_xyxy(boxes_raw)
        (xyxy_nms,), (scores_nms,), (lm_nms,) = batched_nms(
            self.iou_threshold,
            self.conf_threshold,
            xyxy_all.unsqueeze(0),
            scores_flat.unsqueeze(0),
            None,
            landmarks_raw.unsqueeze(0),
        )

        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return [xyxy_nms], [scores_nms], [lm_nms]

        if len(xyxy_nms) == 0:
            annotated = Image.fromarray(np_img)
            return [], annotated

        # Transform box coordinates back to original image space
        def _rescale_coords(coords_xy: torch.Tensor) -> torch.Tensor:
            """Invert the letterbox transform for a tensor of (x, y) coordinates.

            Parameters
            ----------
            coords_xy
                Tensor of shape ``(..., N)`` where even indices are x-coordinates
                and odd indices are y-coordinates (e.g. ``(N, 4)`` xyxy boxes or
                ``(N, 10)`` flattened landmarks).  Values are in letterboxed
                640x640 pixel space.

            Returns
            -------
            torch.Tensor
                Same shape as input, coordinates mapped back to original image
                pixel space by subtracting letterbox padding and dividing by scale.
            """
            out = coords_xy.clone().float()
            out[..., 0::2] = (out[..., 0::2] - pad_left) / scale
            out[..., 1::2] = (out[..., 1::2] - pad_top) / scale
            return out

        xyxy_orig = _rescale_coords(xyxy_nms)
        xyxy_orig[:, 0::2] = xyxy_orig[:, 0::2].clamp(0, orig_w)
        xyxy_orig[:, 1::2] = xyxy_orig[:, 1::2].clamp(0, orig_h)

        lm_orig = _rescale_coords(lm_nms)
        lm_orig[..., 0::2] = lm_orig[..., 0::2].clamp(0, orig_w)
        lm_orig[..., 1::2] = lm_orig[..., 1::2].clamp(0, orig_h)

        # Build detection list
        detections: list[dict[str, Any]] = []
        for i in range(len(xyxy_orig)):
            box = xyxy_orig[i].tolist()
            lm = lm_orig[i].view(5, 2).tolist()
            detections.append(
                {
                    "box": [round(v) for v in box],
                    "score": float(scores_nms[i].item()),
                    "landmarks": [(round(x), round(y)) for x, y in lm],
                }
            )

        # Visualise
        canvas = np_img.copy()
        for det in detections:
            x1v, y1v, x2v, y2v = det["box"]
            draw_box_from_xyxy(
                canvas,
                (x1v, y1v),
                (x2v, y2v),
                color=(0, 255, 0),
                size=2,
                text=f"{det['score']:.2f}",
            )

        for color_idx in range(5):
            pts = np.array(
                [det["landmarks"][color_idx] for det in detections],
                dtype=np.float32,
            )
            if len(pts):
                draw_points(canvas, pts, color=_LM_COLORS[color_idx], size=5)

        return detections, Image.fromarray(canvas)
