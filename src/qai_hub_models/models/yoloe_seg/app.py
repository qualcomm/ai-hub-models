# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.templates.corevision.draw import create_color_map
from qai_hub_models.models.templates.yolo.app import (
    YoloSegmentationApp,
)
from qai_hub_models.utils.input_spec import InputSpec


def _process_masks_batch(
    pred_post_nms_masks: list[torch.Tensor],
    pred_post_nms_boxes: list[torch.Tensor],
    proto: torch.Tensor,
    input_height: int,
    input_width: int,
) -> list[np.ndarray]:
    """
    Apply proto coefficients to mask predictions and upsample to model input size.

    Wraps ``ultralytics.utils.ops.process_mask`` for every batch element and
    handles zero-detection batches gracefully (returns an empty array instead
    of crashing).

    Parameters
    ----------
    pred_post_nms_masks
        Per-batch mask coefficient tensors, each of shape [num_boxes, 32].
    pred_post_nms_boxes
        Per-batch bounding-box tensors, each of shape [num_boxes, 4].
    proto
        Proto tensor of shape [batch, 32, mask_h, mask_w].
    input_height
        Model input height (used as the upsample target).
    input_width
        Model input width (used as the upsample target).

    Returns
    -------
    list[np.ndarray]
        Per-batch float32 mask arrays of shape [num_boxes, input_height, input_width].
        Empty batches produce shape [0, input_height, input_width].
    """
    from ultralytics.utils.ops import process_mask

    processed: list[np.ndarray] = []
    for batch_idx in range(len(pred_post_nms_masks)):
        if pred_post_nms_masks[batch_idx].shape[0] == 0:
            processed.append(np.zeros((0, input_height, input_width), dtype=np.float32))
        else:
            processed.append(
                process_mask(
                    proto[batch_idx],
                    pred_post_nms_masks[batch_idx],
                    pred_post_nms_boxes[batch_idx],
                    (input_height, input_width),
                    upsample=True,
                ).numpy()
            )
    return processed


def _resize_masks_to_input(
    processed_masks: list[np.ndarray],
    input_h: int,
    input_w: int,
) -> list[np.ndarray]:
    """
    Resize processed mask arrays to the original input image dimensions.

    Uses bilinear interpolation via ``torch.nn.functional.interpolate`` and
    handles zero-detection batches gracefully.

    Parameters
    ----------
    processed_masks
        Per-batch float32 mask arrays of shape [num_boxes, mask_h, mask_w].
    input_h
        Target height (original image height before model preprocessing).
    input_w
        Target width (original image width before model preprocessing).

    Returns
    -------
    list[np.ndarray]
        Per-batch float32 mask arrays of shape [num_boxes, input_h, input_w].
        Empty batches produce shape [0, input_h, input_w].
    """
    resized: list[np.ndarray] = []
    for mask in processed_masks:
        if mask.shape[0] == 0:
            resized.append(np.zeros((0, input_h, input_w), dtype=np.float32))
        else:
            resized_tensor = F.interpolate(
                input=torch.from_numpy(mask).float().unsqueeze(0),
                size=(input_h, input_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            resized.append(resized_tensor.numpy())
    return resized


def _overlay_masks_on_image(
    img_tensor: np.ndarray,
    masks: np.ndarray,
    mask_threshold: float = 0.5,
) -> Image.Image:
    """
    Overlay segmentation masks on an image using threshold-based per-mask coloring.

    Each mask is rendered independently: pixels where mask value > mask_threshold
    are blended with a unique color. This ensures all detected instances are visible
    even when their mask confidence values differ significantly.

    Parameters
    ----------
    img_tensor
        RGB image array of shape [H, W, 3] (uint8).
    masks
        Float mask array of shape [num_masks, H, W] with values in [0, 1].
    mask_threshold
        Threshold above which a pixel is considered part of a mask.

    Returns
    -------
    Image.Image
        Annotated PIL image.
    """
    if masks.shape[0] == 0:
        return Image.fromarray(img_tensor)
    num_masks = masks.shape[0]
    # +1 so index 0 (background/black) is never assigned to a real mask
    color_map = create_color_map(num_masks + 1)
    result = img_tensor.astype(np.float32)
    for mask_idx in range(num_masks):
        binary_mask = masks[mask_idx] > mask_threshold
        if binary_mask.any():
            color = color_map[mask_idx + 1].astype(np.float32)
            result[binary_mask] = result[binary_mask] * 0.5 + color * 0.5
    return Image.fromarray(result.astype(np.uint8))


class YoloESegmentationApp(YoloSegmentationApp):
    """
    YoloSegmentationApp variant for YOLOE (text-prompted) segmentation models.

    Overrides three hook methods on the base class so that
    ``predict_segmentation_from_image`` automatically uses YOLOE-specific
    behaviour:

    * ``filter_predictions`` - keeps only detections whose class index is in
      the set derived from *filter_prompt* / *model_classes*.
    * ``process_and_resize_masks`` - uses a batch-safe path that handles
      zero-detection batches without crashing.
    * ``create_output_images`` - uses threshold-based per-instance colouring
      so every detected object is visible regardless of relative mask
      confidence.

    Parameters
    ----------
    model
        Yolo Segmentation model

        Inputs:
            Tensor of shape (N H W C x float32) with range [0, 1] and RGB channel layout.

        Outputs:
            boxes: torch.Tensor
                Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2)
            scores: torch.Tensor
                Class scores multiplied by confidence: Shape is [batch, num_preds]
            masks: torch.Tensor
                Predicted masks: Shape is [batch, num_preds, 32]
            classes: torch.Tensor
                Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
            protos: torch.Tensor
                Tensor of shape[batch, 32, mask_h, mask_w]
                Multiply masks and protos to generate output masks.
    nms_score_threshold
        Score threshold for non maximum suppression.
    nms_iou_threshold
        Intersection over Union threshold for non maximum suppression.
    filter_prompt
        List of class names to keep in the output (e.g. ["bus"]).
        When None, all detected classes are shown.
    model_classes
        Ordered list of class names the model was exported with
        (e.g. ["bus", "person"]).  Used together with filter_prompt to
        map class names to the integer indices produced by the model.
        When None, no class-based filtering is applied.
    input_spec
        Model input spec. If None, defaults to 640x640.
    """

    def __init__(
        self,
        model: Callable[
            [torch.Tensor],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        nms_score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.7,
        filter_prompt: list[str] | Any | None = None,
        model_classes: list[str] | Any | None = None,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(
            model,
            nms_score_threshold,
            nms_iou_threshold,
            input_spec=input_spec,
        )
        # Build the set of class indices that should be kept.
        if filter_prompt is not None and model_classes is not None:
            filter_set = set(filter_prompt)
            self.filter_indices: set[int] | None = {
                i for i, cls in enumerate(model_classes) if cls in filter_set
            }
        else:
            self.filter_indices = None

    # ------------------------------------------------------------------
    # Overridden hook methods
    # ------------------------------------------------------------------

    def filter_predictions(
        self,
        pred_post_nms_boxes: list[torch.Tensor],
        pred_post_nms_scores: list[torch.Tensor],
        pred_post_nms_class_idx: list[torch.Tensor],
        pred_post_nms_masks: list[torch.Tensor],
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        """
        Keep only detections whose class index is in ``self.filter_indices``.

        When ``self.filter_indices`` is None (no prompt / class list provided)
        all detections are returned unchanged.
        """
        if self.filter_indices is not None:
            for batch_idx in range(len(pred_post_nms_class_idx)):
                if len(pred_post_nms_class_idx[batch_idx]) == 0:
                    continue
                keep = torch.tensor(
                    [
                        cls_idx.item() in self.filter_indices
                        for cls_idx in pred_post_nms_class_idx[batch_idx]
                    ],
                    dtype=torch.bool,
                )
                pred_post_nms_boxes[batch_idx] = pred_post_nms_boxes[batch_idx][keep]
                pred_post_nms_scores[batch_idx] = pred_post_nms_scores[batch_idx][keep]
                pred_post_nms_class_idx[batch_idx] = pred_post_nms_class_idx[batch_idx][
                    keep
                ]
                pred_post_nms_masks[batch_idx] = pred_post_nms_masks[batch_idx][keep]
        return (
            pred_post_nms_boxes,
            pred_post_nms_scores,
            pred_post_nms_class_idx,
            pred_post_nms_masks,
        )

    def process_and_resize_masks(
        self,
        pred_post_nms_masks: list[torch.Tensor],
        pred_post_nms_boxes: list[torch.Tensor],
        proto: torch.Tensor,
        input_h: int,
        input_w: int,
    ) -> list[np.ndarray]:
        """
        Batch-safe mask processing that handles zero-detection batches.

        Delegates to :func:`_process_masks_batch` and
        :func:`_resize_masks_to_input` so that batches with no detections
        produce empty arrays rather than crashing.

        Parameters
        ----------
        pred_post_nms_masks
            Per-batch mask-coefficient tensors, each of shape [num_boxes, 32].
        pred_post_nms_boxes
            Per-batch bounding-box tensors, each of shape [num_boxes, 4].
        proto
            Proto tensor of shape [batch, 32, mask_h, mask_w].
        input_h
            Original image height (resize target).
        input_w
            Original image width (resize target).

        Returns
        -------
        list[np.ndarray]
            Per-batch float32 mask arrays of shape [num_boxes, input_h, input_w].
        """
        processed = _process_masks_batch(
            pred_post_nms_masks,
            pred_post_nms_boxes,
            proto,
            self.input_height,
            self.input_width,
        )
        return _resize_masks_to_input(processed, input_h, input_w)

    def create_output_images(
        self,
        NHWC_int_numpy_frames: list[np.ndarray],
        resized_masks: list[np.ndarray],
    ) -> list[Image.Image]:
        """
        Overlay masks using threshold-based per-instance colouring.

        Unlike the base-class argmax approach, each mask is rendered
        independently so that all detected instances remain visible even when
        their confidence values differ significantly.
        """
        out = []
        for i, img_tensor in enumerate(NHWC_int_numpy_frames):
            out.append(_overlay_masks_on_image(img_tensor, resized_masks[i]))
        return out

    def predict_segmentation_from_image_yoloe(
        self,
        pixel_values_or_image: (
            torch.Tensor | np.ndarray | Image.Image | list[Image.Image]
        ),
        raw_output: bool = False,
    ) -> (
        tuple[
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
            list[torch.Tensor],
        ]
        | list[Image.Image]
    ):
        """
        Backward-compatible alias for ``predict_segmentation_from_image``.

        All YOLOE-specific behaviour (class filtering, batch-safe mask
        processing, threshold-based visualisation) is now provided through
        the overridden hook methods, so calling this method is equivalent to
        calling ``predict_segmentation_from_image`` directly.
        """
        return self.predict_segmentation_from_image(pixel_values_or_image, raw_output)
