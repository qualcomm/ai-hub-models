# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import torch

from qai_hub_models.models.templates.corevision.bounding_box_processing import (
    box_xywh_to_xyxy,
)


def decode_preds_from_anchors(
    boxes_and_coordinates: torch.Tensor,
    img_size: tuple[int, int],
    anchors: torch.Tensor,
) -> torch.Tensor:
    """
    Decode predictions using the provided anchors.

    This function can be exported and run inside inference frameworks if desired.

    Parameters
    ----------
    boxes_and_coordinates
        Coordiantes in pixel space. Shape (B, N, K, 2), where N is number of detections, and K is the number of coordinates.
        In the second to last dimension:
        - indices [0-1] are the box coordinates ((x_min, y_min), (width, height))
        - indices [2-K-1] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6), (x7,y7)).
    img_size
        The size of the tensor that was fed to the NETWORK (NOT the original image size). Layout is [H, W]
        H / W is the same order as coordinates.
    anchors
        Box anchors. Range must be [0, 1]. Shape is [Batch, Num Anchors, 2, 2],
        where [2, 2] == [[x_offset, y_offset], [x_scale, y_scale]]

    Returns
    -------
    decoded_coords: torch.Tensor
        The "boxes_and_coordinates" input decoded using the provided anchors.
    """
    assert boxes_and_coordinates.shape[-1] == anchors.shape[-1] == 2
    assert boxes_and_coordinates.shape[-3] == anchors.shape[-3]

    h_size, w_size = img_size

    # Convert offset from normalized [0,1] to image space
    offset = anchors[..., 0:1, :] * torch.tensor(
        [[w_size, h_size]], dtype=anchors.dtype
    )
    scale = anchors[..., 1:2, :]

    # Create mask to zero out offset at index 1 (wh doesn't get offset added)
    K = boxes_and_coordinates.shape[-2]
    mask = (torch.arange(K) != 1).view(K, 1)

    # Scale all coordinates, then add masked offset
    return boxes_and_coordinates * scale + (offset * mask)


def preprocess_hand_x64(
    pts: torch.Tensor, handedness: torch.Tensor, mirror: bool = False
) -> torch.Tensor:
    """
    Normalize hand landmarks, flatten (63), and concatenate handedness (1) → x64.

    pts: (N, 21, 3), handedness: (N, 1) , mirror: (True/False)
    Returns: x64 tensor of shape (N, 64)

    Notes
    -----
    - This preprocessing is performed outside the model due to w8a8 quantization accuracy
    """
    if mirror:
        x_mirror = torch.tensor([-1.0, 1.0, 1.0]).view(1, 1, 3)
        pts = pts * x_mirror
        handedness = 1.0 - handedness
    # Fixed normalization configuration
    center_idx = torch.tensor(
        [0, 1, 5, 9, 13, 17], dtype=torch.long
    )  # stable anatomical anchors
    epsilon = 1e-5  # small constant to avoid divide-by-zero

    # Compute center from selected landmarks
    center = pts[:, center_idx, :].mean(dim=1, keepdim=True)  # (N, 1, 3)

    # Translate points so center is at origin
    normed = pts - center

    # Compute scale based on max range in X or Y
    x = normed[..., 0]
    y = normed[..., 1]
    range_x = x.max(dim=1).values - x.min(dim=1).values
    range_y = y.max(dim=1).values - y.min(dim=1).values
    scale = torch.maximum(range_x, range_y).view(-1, 1, 1) + epsilon

    # Normalize and flatten
    pts_n = normed / scale
    flat = pts_n.reshape(pts_n.shape[0], 63)

    # Append handedness scalar
    return torch.cat([flat, handedness.view(-1, 1).float()], dim=1)


def mediapipe_detector_postprocess(
    coords: torch.Tensor,
    scores: torch.Tensor,
    score_clipping_threshold: float,
    img_size: tuple[int, int],
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Mediapipe detector postprocessing.

    Parameters
    ----------
    coords
        Detected boxes and coordinates in pixel space. Can be either:
        * shape (B, N, K, 2). Where N is number of detections, and K is the number of coordinates.
            In the second to last dimension:
            - indices [0:1] are the box coordinates ((x_min, y_min), (width, height))
            - indices [2:K-1] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6), (x7,y7)).
        * shape (B, N, M). Where N is number of detections, and M is total number of coordinates K * 2.
            In the last dimension:
            - indices [0:3] are the box coordinates ((x_min, y_min), (width, height))
            - indices [4:M] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), ...
    scores
        Raw model output scores of shape (B, N, 1).
    score_clipping_threshold
        Score clipping threshold for postprocessing.
    img_size
        Shape of the network image input (H, W).
    anchors
        Anchors used for decoding bounding box predictions. Shape [N, 2, 2],
        where [2, 2] == [[x_scale, y_scale], [w_offset, h_offset]]

    Returns
    -------
    coords: torch.Tensor
        Detected boxes and coordinates in pixel space.
        Shape (B, N, M), where N is number of detections, and M is total number of coordinates K * 2.
        In the last dimension:
        - indices [0-3] are the box coordinates ((x_min, y_min), (x_max, y_max))
        - indices [4-M] are the K keypoint coordinates ((x1,y1), (x2,y2), (x3,y3), ...).e.

    scores: torch.Tensor
        Clipped and sigmoid activated scores of shape (B, N).
    """
    scores = scores.clamp(
        -score_clipping_threshold,
        score_clipping_threshold,
    )
    scores = scores.sigmoid()

    # Ensure scores are of shape [B, N]
    if scores.dim() == 3 and scores.size(-1) == 1:
        scores = scores.squeeze(dim=-1)

    # Reshape outputs so that they have shape [..., # of coordinates, 2], where 2 == (x, y)
    if coords.dim() == 3:
        coords = coords.view([*list(coords.shape)[:-1], -1, 2])

    # Decode to output coordinates using the model's trained anchors.
    coords = decode_preds_from_anchors(coords, img_size, anchors)

    # flatten coords (remove final [2] dim)
    coords = (
        coords.view([*list(coords.shape)[:-2], -1]) if len(coords.shape) > 3 else coords
    )

    # Convert box coordinates from CWH -> XYXY format.
    xyxy_boxes = box_xywh_to_xyxy(coords[..., :4])
    coords = torch.cat([xyxy_boxes, coords[..., 4:]], dim=-1)

    return coords, scores
