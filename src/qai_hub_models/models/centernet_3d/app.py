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

from qai_hub_models.models.centernet_3d.util import ddd_post_process
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    compute_box_3d,
    draw_3d_bbox,
)
from qai_hub_models.models.templates.corevision.draw import draw_box_from_corners
from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.models.templates.corevision.image_processing_3d import (
    project_to_image,
)

OBJECT_CLASSES = {
    1: (0, 0, 230),  # pedestrian
    2: (255, 158, 0),  # car
    3: (220, 20, 60),  # cyclist
}


class CenterNet3DApp:
    """
    This class is required to perform end to end inference for CenterNet3D Model

    For a given images input, the app will:
        * pre-process the inputs (convert to range[0, 1])
        * Run the inference
        * Convert the hm, dep, rot, dim, wh, reg into 3D_bboxes
        * Draw 3D_bbox in the image and bev image.
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
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
            ],
            torch.Tensor,
        ],
        max_dets: int = 100,
        model_input_shape: tuple[int, int] = (384, 1280),
    ) -> None:
        """
        Initialize CenterNet3DApp

        Parameters
        ----------
        model
            CenterNet3D Model.
        decode
            Function to decode the raw model outputs
            into detected objects/detections.
        max_dets
            Maximum number of detections per image.
        model_input_shape
            (height, width) of the model input.
        """
        self.model = model
        self.decode = decode
        self.max_dets = max_dets
        self.model_input_shape = model_input_shape
        self.vis_threshold = 0.3
        self.out_bev_size = 384
        self.world_size = 64
        self.calib = np.array(
            [
                [707.0493, 0, 604.0814, 45.75831],
                [0, 707.0493, 180.5066, -0.3454157],
                [0, 0, 1.0, 0.004981016],
            ],
            dtype=np.float32,
        )

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> np.ndarray | tuple[Image.Image, Image.Image]:
        # See predict_3d_boxes_from_image.
        return self.predict_3d_boxes_from_image(*args, **kwargs)

    def predict_3d_boxes_from_image(
        self,
        image: Image.Image,
        raw_output: bool = False,
    ) -> np.ndarray | tuple[Image.Image, Image.Image]:
        """
        Run the CenterNet3D model and predict 3d bounding boxes.

        Parameters
        ----------
        image
            PIL images in RGB format.
        raw_output
            If True, returns raw detection output.

        Returns
        -------
        detections_or_images : np.ndarray | tuple[Image.Image, Image.Image]
            If raw_output is True:
                dets with shape (1, max_dets, 3)
            Otherwise:
                Tuple containing image with 3d bounding boxes and bev image.
        """
        image_array = np.array(image)
        h, w = self.model_input_shape
        height, width = image_array.shape[0:2]
        c = np.array([width / 2, height / 2])
        s = np.array([width, height])
        image_tensor = pre_process_with_affine(image_array, c, s, 0, (h, w))

        assert image_tensor.shape[0] == 1, "Model supports only single batch"
        hm, dep, rot, edim, wh, reg = self.model(image_tensor)

        dets = self.decode(hm, rot, dep, edim, wh, reg, self.max_dets).detach().numpy()

        if raw_output:
            return dets

        pp_dets = ddd_post_process(
            dets,
            [c],
            [s],
            tuple(hm.shape[2:]),  # type: ignore[arg-type]
            [self.calib],
        )

        bird_view = (
            np.ones((self.out_bev_size, self.out_bev_size, 3), dtype=np.uint8) * 255
        )
        coords = []
        labels = []
        det: np.ndarray
        for det in pp_dets[0]:
            if det[-2] > self.vis_threshold:
                dim = det[5:8]  # (h, w, l)
                loc = det[8:11]  # (x, y, z) in camera coordinates
                rot_y = det[11]  # rotation_y
                label = det[-1]

                box_3d = compute_box_3d(dim, loc, rot_y)

                if loc[2] > 1:  # Ensure object is in front of the camera
                    coords.append(project_to_image(box_3d, self.calib)[None])
                    labels.append(int(label))

                rect = box_3d[:4, [0, 2]]

                # Translate to positive coordinates for BEV visualization
                rect[:, 0] = rect[:, 0] + int(self.world_size / 2)
                rect[:, 1] = self.world_size - rect[:, 1]  # Invert z-axis for BEV

                # Scale to BEV image dimensions
                rect = rect * self.out_bev_size / self.world_size
                rect = rect[[0, 1, 3, 2]].astype(np.int32)
                draw_box_from_corners(bird_view, rect, OBJECT_CLASSES[label + 1])
        bbox_3d = draw_3d_bbox(
            image_array, np.concatenate(coords), np.array(labels), OBJECT_CLASSES
        )
        pil_image_bbox = Image.fromarray(bbox_3d)
        pil_image_bev = Image.fromarray(bird_view)
        return pil_image_bbox, pil_image_bev
