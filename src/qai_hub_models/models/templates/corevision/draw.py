# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.types import Number


def draw_points(
    frame: np.ndarray,
    points: np.ndarray | torch.Tensor,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int | list[int] = 10,
) -> None:
    """
    Draw the given points on the frame.

    Parameters
    ----------
    frame
        np array (H W C x uint8, RGB)
    points
        array (N, 2) where layout is
            [x1, y1] [x2, y2], ...
        or
        array (N * 2,) where layout is
            x1, y1, x2, y2, ...
    color
        Color of drawn points (RGB)
    size
        Size of drawn points

    Returns
    -------
    None
        Modifies frame in place.
    """
    if len(points.shape) == 1:
        points = points.reshape(-1, 2)
    assert isinstance(size, int) or len(size) == len(points)
    cv_keypoints = []
    for i, (x, y) in enumerate(points):
        curr_size = size if isinstance(size, int) else size[i]
        cv_keypoints.append(cv2.KeyPoint(int(x), int(y), curr_size))

    cv2.drawKeypoints(
        frame,
        cv_keypoints,
        outImage=frame,
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def draw_connections(
    frame: np.ndarray,
    points: np.ndarray | torch.Tensor,
    connections: list[tuple[int, int]] | None = None,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 1,
) -> None:
    """
    Draw connecting lines between the given points on the frame.

    Parameters
    ----------
    frame
        np array (H W C x uint8, RGB)
    points
        array (N, 2) where layout is
            [x1, y1] [x2, y2], ...
        or
        array (N * 2,) where layout is
            x1, y1, x2, y2, ...
        or
        array (N, 2, 2) where layout is
            [
              [ # connection 1
                [ x1, y1 ]
                [ x2, y2 ]
              ],
              [ # connection 2
                [ x1, y1 ]
                [ x2, y2 ]
              ],
              ...
            ]
            (in this case, connections is unused and can be None)
    connections
        List of points that should be connected by a line.
        Format is [(src point index, dst point index), ...]

        Unused if points is of shape (N, 2, 2).
    color
        Color of drawn points (RGB)
    size
        Size of drawn connection lines

    Returns
    -------
    None
        Modifies frame in place.
    """
    point_pairs: (
        list[tuple[tuple[int, int], tuple[int, int]]] | torch.Tensor | np.ndarray
    )
    if len(points.shape) == 3:
        point_pairs = points
    else:
        assert connections is not None
        if len(points.shape) == 1:
            points = points.reshape(-1, 2)
        point_pairs = [
            (
                (int(points[i][0]), int(points[i][1])),
                (int(points[j][0]), int(points[j][1])),
            )
            for (i, j) in connections
        ]
    cv2.polylines(
        frame,
        np.asarray(point_pairs, dtype=np.int64),
        isClosed=False,
        color=color,
        thickness=size,  # type: ignore[call-overload]
    )


def draw_box_from_corners(
    frame: np.ndarray,
    corners: np.ndarray | torch.Tensor,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
) -> None:
    """
    Draw a box using the 4 points provided as boundaries.

    Parameters
    ----------
    frame
        np array (H W C x uint8, RGB)
    corners
        array (4, 2) where layout is
            [x1, y1] [x2, y2], ...
        or
        array (8) where layout is
            x1, y1, x2, y2
    color
        Color of drawn points and connection lines (RGB)
    size
        Size of drawn points and connection lines

    Returns
    -------
    None
        Modifies frame in place.
    """
    draw_points(frame, corners, color, size)
    draw_connections(frame, corners, [(0, 1), (0, 2), (1, 3), (2, 3)], color, size)


def draw_box_from_xywh(
    frame: np.ndarray,
    box: np.ndarray | torch.Tensor,
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
) -> None:
    """
    Draw a box using the provided data (center / height / width) to compute the box.

    Parameters
    ----------
    frame
        np array (H W C x uint8, RGB)
    box
        array (4), where layout is
            [xcenter, ycenter, h, w]
    color
        Color of drawn points and connection lines (RGB)
    size
        Size of drawn points and connection lines

    Returns
    -------
    None
        Modifies frame in place.
    """
    xc, yc, h, w = box
    TL = [xc - w // 2, yc - h // 2]
    BR = [xc + w // 2, yc + h // 2]
    cv2.rectangle(frame, TL, BR, color, size)


def draw_box_from_xyxy(
    frame: np.ndarray,
    top_left: np.ndarray | torch.Tensor | tuple[int, int],
    bottom_right: np.ndarray | torch.Tensor | tuple[int, int],
    color: tuple[int, int, int] = (0, 0, 0),
    size: int = 3,
    text: str | None = None,
) -> None:
    """
    Draw a box using the provided top left / bottom right points to compute the box.

    Parameters
    ----------
    frame
        np array (H W C x uint8, RGB)
    top_left
        Top left corner coordinates (x, y)
    bottom_right
        Bottom right corner coordinates (x, y)
    color
        Color of drawn points and connection lines (RGB)
    size
        Size of drawn points and connection lines
    text
        Overlay text at the top of the box.

    Returns
    -------
    None
        Modifies frame in place.
    """
    if not isinstance(top_left, tuple):
        top_left = (int(top_left[0].item()), int(top_left[1].item()))
    if not isinstance(bottom_right, tuple):
        bottom_right = (int(bottom_right[0].item()), int(bottom_right[1].item()))
    cv2.rectangle(frame, top_left, bottom_right, color, size)
    if text is not None:
        cv2.putText(
            frame,
            text,
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            size,
        )


def create_color_map(
    num_classes: Number,
    colors: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Assign a random color to each class in the dataset to produce a segmentation mask for drawing.

    Parameters
    ----------
    num_classes
        Number of colors to produce.
    colors
        Optional dict mapping class index to an (R, G, B) tuple.
        Overrides the randomly generated color for those classes.

    Returns
    -------
    color_map : np.ndarray
        Array of shape (num_classes, 3) containing RGB colors for each class.
    """
    # Use seed for reproducible results
    #
    # We use RandomState instead of np.random.default_rng here because default_rng is a
    # different randomization algorithm from np.random.rand(). RandomState matches the old behavior.
    # Several model tests rely on the older RandomState rng.
    color_map = np.random.RandomState(42).randint(
        0, 256, size=(int(num_classes), 3), dtype=np.uint8
    )
    color_map[0] = [0, 0, 0]  # Background class, usually black
    if colors is not None:
        for idx, rgb in colors.items():
            color_map[idx] = rgb
    return color_map


def draw_obb_on_image(
    image: Image.Image,
    boxes_xywh: torch.Tensor,
    angles: torch.Tensor,
    color: tuple[int, int, int] = (0, 255, 0),
    width: int = 2,
) -> None:
    """
    Draw Oriented Bounding Boxes on a PIL image.
    boxes_xywh: [N, 4]
    angles: [N] (radians)
    """
    draw = ImageDraw.Draw(image)

    for i in range(len(boxes_xywh)):
        x_c, y_c, w, h = boxes_xywh[i].tolist()
        angle = angles[i].item()

        # 1. Define unrotated corners relative to center
        # Top-Left, Top-Right, Bottom-Right, Bottom-Left
        corners = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]

        # 2. Rotate and Translate
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        poly_points = []
        for cx, cy in corners:
            # Rotate
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            # Translate
            poly_points.append((rx + x_c, ry + y_c))

        # 3. Draw Polygon
        draw.polygon(poly_points, outline=color, width=width)
