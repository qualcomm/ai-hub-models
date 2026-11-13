# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)


def make_grid(
    nx: int,
    ny: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a YOLO grid in ``(1, 1, ny, nx, 2)`` layout.

    Parameters
    ----------
    nx
        Number of grid cells along the x axis.
    ny
        Number of grid cells along the y axis.
    device
        Device on which to create the grid.
    dtype
        Data type of the grid values.

    Returns
    -------
    torch.Tensor
        Grid containing ``(x, y)`` cell coordinates.
    """
    grid_y = torch.arange(ny, device=device, dtype=dtype)
    grid_x = torch.arange(nx, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(1, 1, ny, nx, 2)


def box_transform_xywh2xyxy_split_input(
    xy: torch.Tensor, wh: torch.Tensor
) -> torch.Tensor:
    """
    Convert boxes with (xy, wh) layout to (xyxy)

    Parameters
    ----------
    xy
        Center coordinates (x, y) of boxes.
    wh
        Width and height of boxes.

    Returns
    -------
    converted_boxes : torch.Tensor
        Output box with layout (xyxy)
        i.e. [top_left_x | top_left_y | bot_right_x | bot_right_y]
    """
    wh_half = wh * 0.5
    return torch.cat((xy - wh_half, xy + wh_half), -1)


def detect_postprocess(
    detector_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Post processing to break Yolo(v6,v7) detector output into multiple, consumable tensors (eg. for NMS).
        such as bounding boxes, classes, and confidence.

    Parameters
    ----------
    detector_output
        The output of Yolo Detection model
        Shape is [batch, num_preds, k]
            where, k = # of classes + 5
            k is structured as follows [box_coordinates (4) , conf (1) , # of classes]
            and box_coordinates are [x_center, y_center, w, h]

    Returns
    -------
    boxes : torch.Tensor
        Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
    scores : torch.Tensor
        Class scores multiplied by confidence. Shape is [batch, num_preds].
    class_idx : torch.Tensor
        Predicted class for each bounding box. Shape [batch, num_preds, 1].
    """
    # Break output into parts
    boxes = detector_output[:, :, :4]
    conf = detector_output[:, :, 4:5]
    scores = detector_output[:, :, 5:]

    # Convert boxes to (x1, y1, x2, y2)
    boxes = box_xywh_to_xyxy(boxes)

    # Combine confidence and scores.
    conf_scores = scores * conf

    # Get class ID of most likely score.
    final_scores, class_idx = get_most_likely_score(conf_scores)

    return boxes, final_scores, class_idx.to(torch.uint8)


def detect_postprocess_split_input(
    xy: torch.Tensor,
    wh: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same as `detect_postprocess` with inputs split into separate tensors.

    Parameters
    ----------
    xy
        Center coordinates (x, y) of boxes.
    wh
        Width and height of boxes.
    scores
        Confidence and class scores for each box.

    Returns
    -------
    boxes : torch.Tensor
        Bounding box locations. Shape is [batch, num preds, 4] where 4 == (x1, y1, x2, y2).
    class_idx_scores : torch.Tensor
        Class scores multiplied by confidence. Shape is [batch, num_preds].
    class_idx : torch.Tensor
        Predicted class for each bounding box. Shape [batch, num_preds].
    """
    boxes = box_transform_xywh2xyxy_split_input(xy, wh)

    # For TF Lite, this translates to slice + reshape rather than unpack.
    # This works around a quantization issue where the unpack is left as fp32.
    # TODO(#16306): Move this to regular slice when the quantization issue is fixed.
    conf, per_class_scores = scores.split_with_sizes([1, scores.shape[-1] - 1], dim=-1)

    # Get class ID of most likely score.
    class_idx_scores, class_idx = torch.max(per_class_scores, -1, keepdim=False)

    # Combine confidence and scores.
    # Original repo does this before the max operation, but that is more
    # expensive by a factor of NUM_CLASSES and mathematically equivalent to this.
    class_idx_scores *= conf.squeeze(-1)

    # Cast classes to uint8 for imsdk compatibility
    return boxes, class_idx_scores, class_idx.to(torch.uint8)


def get_most_likely_score(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns most likely score and class id

    Parameters
    ----------
    scores
        final score after post-processing predictions

    Returns
    -------
    scores : torch.Tensor
        Class scores reduced to keep max score per prediction. Shape is [batch, num_preds].
    class_idx : torch.Tensor
        Shape is [batch, num_preds] where the last dim is the index of the most probable class of the prediction.
    """
    scores, class_idx = torch.max(scores, -1, keepdim=False)
    return scores, class_idx
