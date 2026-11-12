# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from qai_hub_models.extern.numba import njit, prange


def compute_iou_3d_rotated(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes the 3D Intersection over Union (IoU) between two rotated bounding boxes.

    This simplified implementation assumes the boxes are axis-aligned and ignores rotation
    for computational efficiency.

    Parameters
    ----------
    box1
        Array of shape (7,) representing a 3D bounding box:
        [x, y, z, l, w, h, theta]
        - x, y, z : Center coordinates
        - l, w, h : Length, width, height
        - theta : Rotation angle (ignored)
    box2
        Array of shape (7,) representing another 3D bounding box in the same format as box1.

    Returns
    -------
    IoU : float
        IoU value between 0.0 and 1.0.
    """
    x1, y1, z1, l1, w1, h1, _ = box1
    x2, y2, z2, l2, w2, h2, _ = box2

    # Compute min and max corners of both boxes
    min1 = [x1 - l1 / 2, y1 - w1 / 2, z1 - h1 / 2]
    max1 = [x1 + l1 / 2, y1 + w1 / 2, z1 + h1 / 2]
    min2 = [x2 - l2 / 2, y2 - w2 / 2, z2 - h2 / 2]
    max2 = [x2 + l2 / 2, y2 + w2 / 2, z2 + h2 / 2]

    # Compute intersection box
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    inter_volume = np.prod(inter_dims)

    # Compute volumes
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    union_volume = vol1 + vol2 - inter_volume
    return inter_volume / union_volume if union_volume > 0 else 0.0


def compute_box_3d(
    dim: np.ndarray, location: np.ndarray, rotation_y: float
) -> np.ndarray:
    """
    Computes the 3D corner coordinates of a bounding box given its dimensions,
    location, and yaw rotation.

    Parameters
    ----------
    dim
        A NumPy array of shape (3,) representing (height, width, length) of the box.
    location
        A NumPy array of shape (3,) representing the (x, y, z) centroid
        location of the box in camera coordinates.
    rotation_y
        A float representing the yaw rotation (around the y-axis) of the box
        in radians.

    Returns
    -------
    corners_3d : np.ndarray
        A NumPy array of shape (8, 3) representing the 8 corners
        of the 3D bounding box in camera coordinates (x, y, z).
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def draw_3d_bbox(
    image: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    classes: dict,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw 3D bounding boxes projected onto 2D image from a camera view.

    Parameters
    ----------
    image
        Numpy array of original image (H W C x uint8) in RGB channel layout.
    coords
        Coordinates of corners in 3D bounding boxes with shape (N, 8, 2).
    labels
        Labels of 3D bounding boxes with shape (N,).
    classes
        Object classes. Class_name as key and color as value in tuple (R, G, B) in range[0-255].
    thickness
        Line thickness, default is 1.

    Returns
    -------
    image_with_boxes : np.ndarray
        Numpy array of image with bboxes (H W C x uint8) in RGB channel layout.
    """
    canvas = image.copy()

    # draw the bboxes
    for index in range(coords.shape[0]):
        name = list(classes.keys())[labels[index]]

        # edges of bboxes
        for start, end in [
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 5),
            (3, 2),
            (3, 7),
            (4, 5),
            (4, 7),
            (2, 6),
            (5, 6),
            (6, 7),
        ]:
            cv2.line(
                canvas,
                coords[index, start].astype(int),
                coords[index, end].astype(int),
                classes[name],
                thickness,
                cv2.LINE_AA,
            )
    return canvas.astype(np.uint8)


def transform_to_matrix(
    translation: list[float],
    rotation: list[float],
    inv: bool = False,
    flat: bool = False,
) -> np.ndarray:
    """
    Converts translation and rotation to a 4x4 transformation matrix.

    Parameters
    ----------
    translation
        Translation vector [x, y, z].
    rotation
        Quaternion rotation [w, x, y, z].
    inv
        Compute the inverse transformation.
    flat
        Use only yaw from rotation for a 2D transformation.

    Returns
    -------
    transformation_matrix : np.ndarray
        4x4 transformation matrix with shape [4, 4].
    """
    from pyquaternion import Quaternion

    if flat:
        yaw = Quaternion(rotation).yaw_pitch_roll[0]
        R = Quaternion(
            scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]
        ).rotation_matrix
    else:
        R = Quaternion(rotation).rotation_matrix

    t = np.array(translation, dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = R if not inv else R.T
    mat[:3, -1] = t if not inv else R.T @ -t
    return mat


def rotation_3d_in_axis(
    points: torch.Tensor,
    angles: torch.Tensor | float,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Rotate points by angles according to axis.

    Parameters
    ----------
    points
        Points of shape (N, M, 3).
    angles
        Vector of angles in shape (N,) or a scalar value.
    axis
        The axis to be rotated. Defaults to 0.
    return_mat
        Whether or not return the rotation matrix (transposed).
        Defaults to False.
    clockwise
        Whether the rotation is clockwise. Defaults to False.

    Returns
    -------
    points_new : torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        If return_mat is False, rotated points in shape (N, M, 3).
        If return_mat is True, also returns the rotation matrix.

    Raises
    ------
    ValueError
        When the axis is not in range [0, 1, 2], it will raise value error.
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, (float, int)):
        angles = torch.full(points.shape[:1], angles)
    elif len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles.item())

    assert (
        len(points.shape) == 3
        and len(angles.shape) == 1
        and points.shape[0] == angles.shape[0]
    ), f"Incorrect shape of points angles: {points.shape}, {angles.shape}"

    assert points.shape[-1] in [
        2,
        3,
    ], f"Points size should be 2 or 3 instead of {points.shape[-1]}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis in {1, -2}:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, zeros, -rot_sin]),
                    torch.stack([zeros, ones, zeros]),
                    torch.stack([rot_sin, zeros, rot_cos]),
                ]
            )
        elif axis in {2, -1}:
            rot_mat_T = torch.stack(
                [
                    torch.stack([rot_cos, rot_sin, zeros]),
                    torch.stack([-rot_sin, rot_cos, zeros]),
                    torch.stack([zeros, zeros, ones]),
                ]
            )
        elif axis in {0, -3}:
            rot_mat_T = torch.stack(
                [
                    torch.stack([ones, zeros, zeros]),
                    torch.stack([zeros, rot_cos, rot_sin]),
                    torch.stack([zeros, -rot_sin, rot_cos]),
                ]
            )
        else:
            raise ValueError(f"axis should in range [-3, -2, -1, 0, 1, 2], got {axis}")
    else:
        rot_mat_T = torch.stack(
            [torch.stack([rot_cos, rot_sin]), torch.stack([-rot_sin, rot_cos])]
        )

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum("jka->ajk", rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    return points_new


def circle_nms(
    dets: np.ndarray, thresh: float, post_max_size: int = 83
) -> torch.LongTensor:
    """Circular NMS.

    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.

    Parameters
    ----------
    dets
        Detection results with the shape of [N, 3].
    thresh
        Value of threshold.
    post_max_size
        Max number of prediction to be kept. Defaults to 83.

    Returns
    -------
    keep_indices : torch.LongTensor
        Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    scores = dets[:, 2]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep: list[int] = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist = (x1[i] - x1[j]) ** 2 + (y1[i] - y1[j]) ** 2

            # ovr = inter / areas[j]
            if dist <= thresh:
                suppressed[j] = 1

    if post_max_size < len(keep):
        return torch.LongTensor(keep[:post_max_size])

    return torch.LongTensor(keep)


def compute_iou_bev(box_a: np.ndarray | list, box_b: np.ndarray | list) -> float:
    """
    Compute the IoU (Intersection over Union) between two rotated 3D bounding boxes in BEV.

    Each box is represented as [x_center, y_center, width, height, rotation].
    This version accurately mimics the CUDA function `boxesioubevLauncher()`.

    Parameters
    ----------
    box_a
        First bounding box in format [x_center, y_center, width, height, rotation].
    box_b
        Second bounding box in format [x_center, y_center, width, height, rotation].

    Returns
    -------
    iou : float
        IoU (Intersection over Union) value between the two boxes.
    """
    x_a, y_a, w_a, h_a, angle_a = box_a
    x_b, y_b, w_b, h_b, angle_b = box_b

    # Convert rotated boxes to corner points
    def get_corners(x: float, y: float, w: float, h: float, angle: float) -> np.ndarray:
        """Compute corner points of the rotated rectangle.

        Parameters
        ----------
        x
            X coordinate of box center.
        y
            Y coordinate of box center.
        w
            Width of the box.
        h
            Height of the box.
        angle
            Rotation angle in radians.

        Returns
        -------
        corners : np.ndarray
            Array of shape (4, 2) containing the four corner points.
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        w_half, h_half = w / 2, h / 2
        return np.array(
            [
                [
                    x - w_half * cos_a + h_half * sin_a,
                    y - w_half * sin_a - h_half * cos_a,
                ],
                [
                    x + w_half * cos_a + h_half * sin_a,
                    y + w_half * sin_a - h_half * cos_a,
                ],
                [
                    x + w_half * cos_a - h_half * sin_a,
                    y + w_half * sin_a + h_half * cos_a,
                ],
                [
                    x - w_half * cos_a - h_half * sin_a,
                    y - w_half * sin_a + h_half * cos_a,
                ],
            ]
        )

    corners_a = get_corners(x_a, y_a, w_a, h_a, angle_a)
    corners_b = get_corners(x_b, y_b, w_b, h_b, angle_b)

    # Compute polygon intersection area using Shapely
    from shapely.geometry import Polygon

    poly_a = Polygon(corners_a)
    poly_b = Polygon(corners_b)

    if not poly_a.is_valid or not poly_b.is_valid:
        return 0.0

    intersection_area = poly_a.intersection(poly_b).area
    union_area = poly_a.area + poly_b.area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def nms_cpu(
    boxes: np.ndarray | torch.Tensor,
    scores: np.ndarray | torch.Tensor,
    thresh: float = 0.4,
    pre_maxsize: int | None = None,
    post_max_size: int | None = None,
) -> list[int]:
    """
    CPU Implementation of 3D NMS (Non-Maximum Suppression) using Rotated IoU.

    Parameters
    ----------
    boxes
        Array of boxes with shape [N, 5] in format (x_center, y_center, width, height, ry).
    scores
        Array of confidence scores with shape [N].
    thresh
        IoU threshold for suppression. Defaults to 0.4.
    pre_maxsize
        Maximum number of boxes before NMS. Defaults to None.
    post_max_size
        Maximum number of boxes after NMS. Defaults to None.

    Returns
    -------
    keep_indices : list[int]
        Indices of selected boxes after NMS.
    """
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()

    # Sort boxes by scores in descending order
    order = np.argsort(-scores)

    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        ious = np.array([compute_iou_bev(boxes[i], boxes[j]) for j in order[1:]])

        # Keep boxes with IoU below the threshold
        remaining = np.where(ious < thresh)[0]
        order = order[remaining + 1]  # Shift indices due to slicing

    keep = np.array(keep, dtype=np.int64).tolist()

    if post_max_size is not None:
        keep = keep[:post_max_size]

    return keep


def xywhr2xyxyr(boxes_xywhr: torch.Tensor) -> torch.Tensor:
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Parameters
    ----------
    boxes_xywhr
        Rotated boxes in XYWHR format.

    Returns
    -------
    boxes_xyxyr : torch.Tensor
        Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def compute_corners(tensor: torch.Tensor) -> torch.Tensor:
    """Initialize 3D boxes and compute corners.

    Parameters
    ----------
    tensor
        Input tensor of shape [N, box_dim] containing box parameters.

    Returns
    -------
    corners : torch.Tensor
        Corners of boxes in shape [N, 8, 3].
    """
    dims = tensor[:, 3:6]  # w, l, h
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    ).to(device=dims.device, dtype=dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    # use relative origin [0.5, 0.5, 0]
    corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    corners = rotate_3d_along_axis(corners, tensor[:, 6], axis=2)
    corners += tensor[:, :3].view(-1, 1, 3)
    return corners


def onnx_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # Create a pi tensor with the same device and data type as y
    pi = torch.tensor(torch.pi, device=y.device, dtype=y.dtype)
    half_pi = pi / 2
    eps = 1e-6

    # Compute the arctangent of y/x
    ans = torch.atan(y / (x + eps))

    # Create boolean tensors representing positive, negative, and zero values of y and x
    y_positive = y > 0
    y_negative = y < 0
    x_negative = x < 0
    x_zero = x == 0

    # Adjust ans based on the positive, negative, and zero values of y and x
    ans += torch.where(
        y_positive & x_negative, pi, torch.zeros_like(ans)
    )  # Quadrants I and II
    ans -= torch.where(
        y_negative & x_negative, pi, torch.zeros_like(ans)
    )  # Quadrants III and IV
    ans = torch.where(y_positive & x_zero, half_pi, ans)  # Positive y-axis
    return torch.where(y_negative & x_zero, -half_pi, ans)  # Negative y-axis


def rotate_3d_along_axis(
    points: torch.Tensor, angles: torch.Tensor, axis: int = 0
) -> torch.Tensor:
    """Rotate points by angles according to axis.

    Parameters
    ----------
    points
        Points of shape (N, M, 3).
    angles
        Vector of angles in shape (N,).
    axis
        The axis to be rotated. Defaults to 0.

    Returns
    -------
    rotated_points : torch.Tensor
        Rotated points in shape (N, M, 3).

    Raises
    ------
    ValueError
        When the axis is not in range [0, 1, 2], it will raise value error.
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis in {2, -1}:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, -rot_sin, zeros]),
                torch.stack([rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                torch.stack([zeros, rot_cos, -rot_sin]),
                torch.stack([zeros, rot_sin, rot_cos]),
                torch.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")

    return torch.einsum("aij,jka->aik", (points, rot_mat_T))


@njit
def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate signed area of triangle formed by points a, b, and c.

    Parameters
    ----------
    a
        2D point with shape (2,).
    b
        2D point with shape (2,).
    c
        2D point with shape (2,).

    Returns
    -------
    area : float
        Signed area of the triangle.
    """
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


@njit
def sort_vertex(pts: np.ndarray, num_pts: int) -> np.ndarray:
    """Sort vertices counterclockwise around their centroid.

    Parameters
    ----------
    pts
        Vertices array of shape (N, 2).
    num_pts
        Number of valid vertices.

    Returns
    -------
    sorted_vertices : np.ndarray
        Sorted vertices of shape (num_pts, 2).
    """
    center = np.zeros(2, dtype=np.float32)
    for i in range(num_pts):
        center += pts[i]
    center /= num_pts

    angles = np.empty(num_pts, dtype=np.float32)
    for i in range(num_pts):
        vec = pts[i] - center
        angles[i] = math.atan2(vec[1], vec[0])

    idx_sorted = np.argsort(angles)
    sorted_pts = np.empty((num_pts, 2), dtype=np.float32)
    for i in range(num_pts):
        sorted_pts[i] = pts[idx_sorted[i]]
    return sorted_pts


@njit
def line_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> tuple[bool, np.ndarray]:
    """Compute intersection point of line segments ab and cd.

    Parameters
    ----------
    a
        Start of first line segment with shape (2,) -- (x, y)
    b
        Endpoint of first line segment with shape (2,) -- (x, y)
    c
        Start of second line segment with shape (2,) -- (x, y)
    d
        Endpoint of second line segment with shape (2,) -- (x, y)

    Returns
    -------
    has_intersection : bool
        True if intersection occurs, False otherwise.
    intersection_point : np.ndarray
        Intersection point (x, y) if intersection occurs.
        Zero vector with shape (2,) otherwise.
    """
    area_abc = triangle_area(a, b, c)
    area_abd = triangle_area(a, b, d)
    if area_abc * area_abd >= 0:
        return False, np.zeros(2)
    area_cda = triangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd
    if area_cda * area_cdb >= 0:
        return False, np.zeros(2)
    t = area_cda / (area_abd - area_abc)
    return True, a + t * (b - a)


@njit
def point_in_quad(pt: np.ndarray, corners: np.ndarray) -> bool:
    """Check if a point lies within a convex quadrilateral.

    Parameters
    ----------
    pt
        Point of shape (2,).
    corners
        4 corners of the quadrilateral, of shape (4, 2).

    Returns
    -------
    is_inside : bool
        True if point is inside, Otherwise False.
    """
    ab, ad = corners[1] - corners[0], corners[3] - corners[0]
    ap = pt - corners[0]
    abab, abap = ab @ ab, ab @ ap
    adad, adap = ad @ ad, ad @ ap
    return 0 <= abap <= abab and 0 <= adap <= adad


@njit
def rbbox_to_corners(box: np.ndarray) -> np.ndarray:
    """Convert rotated bbox to 4 corner points.

    Parameters
    ----------
    box
        Bounding box in format [x, y, w, h, theta], of shape (5,).

    Returns
    -------
    corners : np.ndarray
        4 corner points of the rotated box, of shape (4, 2).
    """
    angle = box[4]
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    w, h = box[2] / 2, box[3] / 2
    dx = np.array([[-w, -h], [-w, h], [w, h], [w, -h]])
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    return dx @ R.T + box[:2]


@njit
def inter(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate intersection area between two rotated boxes.

    Parameters
    ----------
    box1
        Rotated bounding box in format [x, y, w, h, theta], of shape (5,).
    box2
        Rotated bounding box in format [x, y, w, h, theta], of shape (5,).

    Returns
    -------
    intersection_area : float
        Intersection area.
    """
    c1, c2 = rbbox_to_corners(box1), rbbox_to_corners(box2)
    int_pts = np.zeros((24, 2), dtype=np.float32)
    num_pts = 0

    for i in range(4):
        if point_in_quad(c1[i], c2):
            int_pts[num_pts] = c1[i]
            num_pts += 1
        if point_in_quad(c2[i], c1):
            int_pts[num_pts] = c2[i]
            num_pts += 1

    for i in range(4):
        a, b = c1[i], c1[(i + 1) % 4]
        for j in range(4):
            c, d = c2[j], c2[(j + 1) % 4]
            has_inter, pt = line_intersection(a, b, c, d)
            if has_inter:
                int_pts[num_pts] = pt
                num_pts += 1

    if num_pts < 3:
        return 0.0

    sorted_pts = sort_vertex(int_pts, num_pts)
    area = 0.0
    for i in range(num_pts - 2):
        area += abs(triangle_area(sorted_pts[0], sorted_pts[i + 1], sorted_pts[i + 2]))
    return area


@njit(parallel=True)
def get_bev_iou_matrix(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """Calculate bird's-eye-view (BEV) IoU between sets of rotated boxes.

    Parameters
    ----------
    boxes
        Array of N boxes in format [x, y, w, h, theta], of shape (N, 5).
    query_boxes
        Array of K boxes in format [x, y, w, h, theta], of shape (K, 5).

    Returns
    -------
    iou_matrix : np.ndarray
        IoU matrix of shape (N, K).
    """
    N, K = boxes.shape[0], query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    for i in prange(N):
        for j in range(K):
            a1 = boxes[i, 2] * boxes[i, 3]
            a2 = query_boxes[j, 2] * query_boxes[j, 3]
            inter_area = inter(boxes[i], query_boxes[j])
            iou[i, j] = inter_area / (a1 + a2 - inter_area)
    return iou


def get_3d_bbox(scale: np.ndarray, shift: float = 0) -> np.ndarray:
    """
    Compute 8 corners of a 3D bounding box.

    Parameters
    ----------
    scale
        Object size (w, h, d), shape (3,), dtype float32.
    shift
        Optional scalar offset added to all corners.

    Returns
    -------
    np.ndarray
        Array of shape (3, 8) containing the 8 bbox corner coordinates.
    """
    bbox_3d = (
        np.array(
            [
                [scale[0] / 2, scale[1] / 2, scale[2] / 2],
                [scale[0] / 2, scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, scale[1] / 2, -scale[2] / 2],
                [scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
            ],
            dtype=np.float32,
        )
        + shift
    )
    return bbox_3d.T  # (3, 8)
