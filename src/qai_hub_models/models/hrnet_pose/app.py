# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from PIL.Image import Image, fromarray

from qai_hub_models.extern.mmengine import patch_mmengine_pkgresources
from qai_hub_models.extern.mmpose import patch_mmpose_no_build_deps
from qai_hub_models.models.templates.corevision.draw import draw_points
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.mmpose.silence import (
    set_mmpose_inferencer_show_progress,
)
from qai_hub_models.models.templates.pose.pose_utils import get_final_preds

with patch_mmpose_no_build_deps(), patch_mmengine_pkgresources():
    from mmpose.apis import MMPoseInferencer
    from mmpose.structures.pose_data_sample import PoseDataSample


# More inferencer architectures for litehrnet can be found at
# https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/topdown_heatmap/coco
DEFAULT_INFERENCER_ARCH = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"


def get_max_preds(batch_heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), (
        "batch_heatmaps should be numpy.ndarray"
    )
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]

    # shape [batch, num joints, image h * w]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

    # index of max pixel per joint
    idx = np.argmax(heatmaps_reshaped, 2)

    # value of max pixel per joint
    maxvals = np.amax(heatmaps_reshaped, 2)

    # Reshape to prep for tiling
    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    # Tile indices to make room for (x, y)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    # Convert index [..., 0] to x
    preds[:, :, 0] = (preds[:, :, 0]) % width
    # Convert [..., 1] to y
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # Tile mask back to (x, y) plane to ignore negatives
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    # apply mask
    preds *= pred_mask
    return preds, maxvals


class HRNetPoseApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with HRNet.

    The app uses 1 model:
        * HRNetPose

    For a given image input, the app will:
        * pre-process the image
        * Run HRNet inference
        * Convert the output into a list of keypoint coordinates
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.model = model
        # Use mmpose inferencer for example preprocessing
        self.inferencer = MMPoseInferencer(DEFAULT_INFERENCER_ARCH, device="cpu")
        set_mmpose_inferencer_show_progress(self.inferencer, False)
        self.pre_processor = self.inferencer.inferencer.model.data_preprocessor

    def predict(self, *args: Any, **kwargs: Any) -> np.ndarray | list[Image]:
        # See predict_pose_keypoints.
        return self.predict_pose_keypoints(*args, **kwargs)

    def preprocess_input(
        self, pixel_values_or_image: torch.Tensor | np.ndarray | Image | list[Image]
    ) -> tuple[list[np.ndarray], dict[str, torch.Tensor], torch.Tensor]:
        # Convert from PIL / torch/ etc. to NHWC, RGB numpy frames, which is the required input type.
        NHWC_int_numpy_frames, _ = app_to_net_image_inputs(pixel_values_or_image)

        # MMPose does a lot of heavy lifting here. The preprocessor does the following:
        # * runs a detector model to find people in each frame
        # * for each bounding box...
        # *     crop to the bounding box. resize bounding box to fit model input size using scaling factor
        # *     Save bounding box coordinates and box scaling factor for use later
        inputs = self.inferencer.preprocess(NHWC_int_numpy_frames, batch_size=1)

        # We only get the first (highest probability) box and ignore the others.
        # Other implementations may choose to run pose estimation on all boxes
        # if they want to support multiple people in the same frame.
        proc_inputs, _ = next(iter(inputs))
        proc_inputs_ = proc_inputs["inputs"][0]

        # BGR -> RGB
        x = proc_inputs_[[2, 1, 0]]
        # Convert to expected model input distrubtion
        x = x.float() / 255.0

        # Add batch dimension
        x = torch.unsqueeze(x, 0)

        return (NHWC_int_numpy_frames, proc_inputs, x)

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
        result : np.ndarray | list[Image]
            If raw_output is True, returns:
            keypoints
                Numpy array of keypoints with shape [B, N, 2].
                Each keypoint is an (x, y) pair of coordinates within the image.

            If raw_output is False, returns:
            predicted_images
                Images with keypoints drawn.
        """
        (NHWC_int_numpy_frames, proc_inputs, x) = self.preprocess_input(
            pixel_values_or_image
        )

        # run inference
        heatmaps = self.model(x).detach().numpy()

        # Coordinates are relative to the cropped bbox, not the original image.
        # We need to grab the box center and scale to transform the coordinates
        # back to the original image.

        # The 3 below lines are poorly typed in MMCV and need casts to work around the issue.
        # We use cast instead of assert here because PoseDataSample may not be defined (given
        # it is imported in a try-catch block), and pylance can't deal with that.
        data_sample = cast(PoseDataSample, proc_inputs["data_samples"][0])
        bbox = cast(np.ndarray, data_sample.gt_instances["bboxes"][0])
        scale = cast(np.ndarray, data_sample.gt_instances["bbox_scales"][0])
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

        # create predictions from heatmap
        keypoints, _ = get_final_preds(heatmaps, np.array([center]), np.array([scale]))
        keypoints = np.round(keypoints).astype(np.int32)
        if raw_output:
            return keypoints

        predicted_images = []
        for i, img in enumerate(NHWC_int_numpy_frames):
            draw_points(img, keypoints[i], color=(255, 0, 0), size=6)
            predicted_images.append(fromarray(img))
        return predicted_images
