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

from qai_hub_models.models.templates.corevision.draw import (
    create_color_map,
    draw_box_from_xyxy,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    denormalize_coordinates_affine,
    pre_process_with_affine,
)
from qai_hub_models.models.templates.corevision.labels import get_class_names


class CenterNet2DApp:
    """
    This class is required to perform end to end inference for CenterNet2D Model

    For a given images input, the app will:
        * pre-process the inputs (convert to range[0, 1])
        * Run the inference
        * Convert the hm, wh, reg into 3D_bboxes
        * Draw 2D_bbox in the image.
    """

    def __init__(
        self,
        model: Callable[
            [
                torch.Tensor,
            ],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                bool,
                int,
            ],
            torch.Tensor,
        ],
        height: int = 512,
        width: int = 512,
        max_dets: int = 100,
        cat_spec_wh: bool = False,
    ) -> None:
        """
        Initialize CenterNet2DApp

        Parameters
        ----------
        model
            CenterNet2D Model.
        decode
            Function to decode the raw model outputs
            into detected objects/detections.
        height
            Input image height.
        width
            Input image width.
        max_dets
            Maximum number of detections per image.
        cat_spec_wh
            If True, indicates that the `wh` tensor is category-specific
            (i.e., its channel dimension is `2 * num_classes`). If False,
            `wh` is not category-specific. Defaults to False.
        """
        self.model = model
        self.decode = decode
        self.height = height
        self.width = width
        self.max_dets = max_dets
        self.cat_spec_wh = cat_spec_wh
        self.vis_threshold = 0.3
        self.num_classes = 80

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | Image.Image:
        # See predict_2d_boxes_from_image.
        return self.predict_2d_boxes_from_image(*args, **kwargs)

    def predict_2d_boxes_from_image(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | Image.Image:
        """
        Run the CenterNet2D model and predict 2d bounding boxes.

        Parameters
        ----------
        image
            PIL images in RGB format.
        raw_output
            If True, returns raw detection output.

        Returns
        -------
        detections_or_image : np.ndarray | Image.Image
            If raw_output is True:
                dets with shape (max_dets, 6)
            Otherwise:
                PIL image with 2d bounding boxes drawn.
        """
        image_array = np.array(image)
        height, width = image_array.shape[0:2]
        c = np.array([width / 2, height / 2], dtype=np.float32)
        s = np.array([max(height, width), max(height, width)], dtype=np.float32)

        image_tensor = pre_process_with_affine(
            image_array, c, s, 0, (self.height, self.width)
        )

        # model supports only single batch
        assert image_tensor.shape[0] == 1
        hm, wh, reg = self.model(image_tensor)

        dets = self.decode(hm, wh, reg, self.cat_spec_wh, self.max_dets).numpy()
        dets = dets.reshape(-1, dets.shape[2])

        if raw_output:
            return dets

        bbox_tl = denormalize_coordinates_affine(
            dets[:, 0:2], c, s, 0, (hm.shape[2], hm.shape[3])
        )
        bbox_br = denormalize_coordinates_affine(
            dets[:, 2:4], c, s, 0, (hm.shape[2], hm.shape[3])
        )
        scores = dets[:, 4]
        labels = dets[:, 5].astype(int)
        color = create_color_map(self.num_classes + 1)

        labels_list = get_class_names("coco")

        for tl, br, score, label in zip(bbox_tl, bbox_br, scores, labels, strict=False):
            if score > self.vis_threshold:
                draw_box_from_xyxy(
                    image_array,
                    tl,
                    br,
                    color=color[label + 1].tolist(),
                    size=2,
                    text=labels_list[label],
                )

        return Image.fromarray(image_array)
