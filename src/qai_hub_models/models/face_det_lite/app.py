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
from skimage.color import rgb2gray

from qai_hub_models.models.face_det_lite.utils import detect
from qai_hub_models.models.templates.corevision.draw import draw_box_from_xyxy
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)


class FaceDetLiteApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FaceDetLite.

    The app uses 1 model:
        * FaceDetLite

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FaceDetLite inference
        * Output list of face Bounding Box objects.
    """

    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model = model

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> tuple[list[list[int | float]], Image.Image]:
        return self.run_inference_on_image(*args, **kwargs)

    def run_inference_on_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
    ) -> tuple[list[list[int | float]], Image.Image]:
        """
        Return the corresponding output by running inference on input image.

        Parameters
        ----------
        pixel_values_or_image
            PIL image(s)
            or
            numpy array (N H W C x uint8) or (H W C x uint8) -- both RGB channel layout
            or
            pyTorch tensor (N C H W x fp32, value range is [0, 1]), RGB channel layout

        Returns
        -------
        face_bounding_boxes : list[list[int | float]]
            A list of BBox for face.
        annotated_image : Image.Image
            Image with detected faces drawn.
        """
        assert pixel_values_or_image is not None, "pixel_values_or_image is None"
        [img_array_color], _ = app_to_net_image_inputs(pixel_values_or_image)

        img_array = rgb2gray(img_array_color)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
        hm, box, landmark = self.model(img_tensor)
        dets = detect(hm, box, landmark, threshold=0.55, nms_iou=-1, stride=8)
        res = []
        for n in range(len(dets)):
            xmin, ymin, w, h = dets[n].xywh
            score = dets[n].score

            L = int(xmin)
            R = int(xmin + w)
            T = int(ymin)
            B = int(ymin + h)
            W = int(w)
            H = int(h)

            img_H, img_W = img_array_color.shape[:2]
            if L < 0 or T < 0 or img_W <= R or img_H <= B:
                L = max(L, 0)
                T = max(T, 0)
                if img_W <= R:
                    R = img_W - 1
                if img_H <= B:
                    B = img_H - 1

            # Enlarge bounding box to cover more face area
            b_Left = L - int(W * 0.05)
            b_Top = T - int(H * 0.05)
            b_Width = int(W * 1.1)
            b_Height = int(H * 1.1)

            if (
                b_Left >= 0
                and b_Top >= 0
                and b_Width - 1 + b_Left < img_W
                and b_Height - 1 + b_Top < img_H
            ):
                L = b_Left
                T = b_Top
                W = b_Width
                H = b_Height
                R = W - 1 + L
                B = H - 1 + T

            res.append([L, T, W, H, score])

        np_out = np.asarray(img_array_color)
        np_out = torch.tensor(np_out).byte().numpy()
        for box in dets:
            box = box.box
            draw_box_from_xyxy(
                np_out,
                torch.tensor(box[0:2]),
                torch.tensor(box[2:4]),
                color=(0, 255, 0),
                size=2,
            )
        out = Image.fromarray(np_out)
        return res, out
