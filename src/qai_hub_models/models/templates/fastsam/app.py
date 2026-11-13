# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics.models.fastsam.utils import adjust_bboxes_to_image_border
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import crop_mask
from ultralytics.utils.plotting import Annotator, colors

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    batched_nms,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    denormalize_coordinates,
    resize_pad,
)


class FastSAMApp:
    """
    This class consists of light-weight "app code" that is required to perform end to end inference with FastSAM.

    The app uses 1 model:
        * FastSAM

    For a given image input, the app will:
        * pre-process the image (convert to range[0, 1])
        * Run FastSAM inference
        * post-process the image
        * display the input and output side-by-side
    """

    def __init__(
        self,
        fastsam_model: Callable[
            [torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        retina_masks: bool = True,
        model_image_input_shape: tuple[int, int] = (640, 640),
    ) -> None:
        self.model = fastsam_model
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.retina_masks = retina_masks
        self.model_image_input_shape = model_image_input_shape

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[Image.Image]
    ):
        # See upscale_image.
        return self.segment_image(*args, **kwargs)

    def segment_image(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> (
        tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        | list[Image.Image]
    ):
        """
        Upscale provided images

        Parameters
        ----------
        pixel_values_or_image
            Input PIL image (before pre-processing) or pyTorch tensor (after image pre-processing).
        raw_output
            Whether to return raw output or annotated images.

        Returns
        -------
        output : tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | list[Image.Image]
            If raw_output is True, returns:
                boxes : list[torch.Tensor]
                    List of each batch of predicted bounding boxes.
                    Each tensor is shape [N, 4], where N is the number of boxes
                    and 4 == [x1, y1, x2, y2] (box coordinates in pixel space).
                scores : list[torch.Tensor]
                    List of each batch of predicted box scores.
                    Each tensor is shape [B], where N is the number of boxes.
                masks : list[torch.Tensor]
                    List of each batch of predicted masks.
                    Each tensor is shape [N, H, W], where N is the number of boxes,
                    and (H, W) is the network image input shape.

            If raw_output is False, returns:
                images : list[Image.Image]
                    A list of images with masks / boxes/ confidences drawn.
        """
        # Input Prep
        NHWC_int_numpy_frames, NCHW_fp32_torch_frames = app_to_net_image_inputs(
            pixel_values_or_image
        )

        source_image_size = (
            NCHW_fp32_torch_frames.shape[2],
            NCHW_fp32_torch_frames.shape[3],
        )
        images, scales, paddings = resize_pad(
            NCHW_fp32_torch_frames, self.model_image_input_shape
        )

        pred_boxes, pred_scores, pred_mask_coeffs, pred_mask_protos = self.model(images)

        # Non Maximum Suppression on each batch
        boxes: list[torch.Tensor]
        scores: list[torch.Tensor]
        mask_coeffs: list[torch.Tensor]
        masks: list[torch.Tensor] = []
        (
            boxes,
            scores,
            mask_coeffs,
        ) = batched_nms(
            self.iou_threshold,
            self.confidence,
            pred_boxes,
            pred_scores,
            None,
            pred_mask_coeffs,
        )

        for batch_idx in range(len(pred_boxes)):
            batch_boxes = boxes[batch_idx]
            batch_mask_coeffs = mask_coeffs[batch_idx]
            batch_mask_protos = pred_mask_protos[batch_idx]
            batch_scores = scores[batch_idx]

            # Compute masks from coeffs + protos
            # It's possible to do this once on all anchors (inside the network)
            # instead of running it after NMS for each batch. However, the batched matmul
            # in the network is very expensive; it increases network latency by 10x.
            #
            # Instead of computing the masks before NMS, computing after substantually
            # reduces computation needs because we have many fewer boxes to deal with.
            c, proto_h, proto_w = batch_mask_protos.shape
            batch_masks = (batch_mask_coeffs @ batch_mask_protos.view(c, -1)).view(
                -1, proto_h, proto_w
            )

            # Crop masks to associated bounding box
            ih, iw = source_image_size
            mh, mw = (batch_masks.shape[1], batch_masks.shape[2])
            width_ratio = mw / iw
            height_ratio = mh / ih
            downsampled_bboxes = batch_boxes.clone()
            downsampled_bboxes[:, 0] *= width_ratio
            downsampled_bboxes[:, 2] *= width_ratio
            downsampled_bboxes[:, 3] *= height_ratio
            downsampled_bboxes[:, 1] *= height_ratio
            batch_masks = cast(torch.Tensor, crop_mask(batch_masks, downsampled_bboxes))

            # Rescale masks to original image size
            batch_masks = F.interpolate(
                batch_masks.unsqueeze(1),
                self.model_image_input_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            # Make mask a 1-hot mapping
            batch_masks = torch.gt(batch_masks, 0.0).type(torch.uint8)

            # Denormalize coords to original image size, clip to image size
            denormalize_coordinates(
                batch_boxes.view(-1, 2, 2), (1, 1), scales, paddings
            )
            batch_boxes[..., [1, 3]] = torch.clip(
                batch_boxes[..., [1, 3]], 0, source_image_size[0]
            )  # y1, y2
            batch_boxes[..., [0, 2]] = torch.clip(
                batch_boxes[..., [0, 2]], 0, source_image_size[1]
            )  # x1, x2

            # Only keep predictions with masks
            keep = batch_masks.sum((-2, -1)) > 0  # only keep predictions with masks
            batch_boxes, batch_scores, batch_masks = (
                batch_boxes[keep],
                batch_scores[keep],
                batch_masks[keep],
            )

            # Adjust boxes to match the edge of the image, if close enough.
            full_box = torch.tensor([0, 0, *source_image_size], dtype=torch.float32)
            batch_boxes = adjust_bboxes_to_image_border(batch_boxes, source_image_size)
            idx = torch.nonzero(
                input=box_iou(full_box[None], batch_boxes) > self.iou_threshold
            ).flatten()
            if idx.numel() != 0:
                batch_boxes[idx] = full_box

            masks.append(batch_masks)
            scores[batch_idx] = batch_scores
            boxes[batch_idx] = batch_boxes

        # Return raw output if requested
        if raw_output or isinstance(pixel_values_or_image, torch.Tensor):
            return boxes, scores, masks

        # Overlay the segmentation masks on the image.
        out = []
        for batch_idx in range(len(pred_boxes)):
            batch_boxes = boxes[batch_idx]
            batch_masks = masks[batch_idx]
            batch_scores = scores[batch_idx]
            batch_frame = NHWC_int_numpy_frames[batch_idx]
            cc = [colors(x, True) for x in reversed(range(len(batch_masks)))]
            a = Annotator(batch_frame, pil=True)
            a.masks(batch_masks.numpy(), colors=cc)
            for box_idx, box in enumerate(batch_boxes):
                a.box_label(
                    box, f"{batch_scores[box_idx].item():.2f}", color=cc[box_idx]
                )
            out.append(a.im)

        return out
