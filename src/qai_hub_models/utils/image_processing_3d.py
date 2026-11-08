# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import functools

import numpy as np
import torch


def project_to_image(pts_3d: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Projects 3D points from camera coordinates to 2D image coordinates using
    a projection matrix.

    Parameters
    ----------
    pts_3d
        A NumPy array of shape (N, 3) representing N 3D points in camera
        coordinates (x, y, z).
    P
        A NumPy array of shape (3, 4) representing the camera projection
        matrix.

    Returns
    -------
    projected_points : np.ndarray
        A NumPy array of shape (N, 2) representing the projected
        2D points (u, v) on the image plane.
    """
    # Convert 3D points to homogeneous coordinates (x, y, z, 1)
    pts_3d_homo = np.concatenate(
        [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1
    )

    # Project to 2D image plane
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)

    # Normalize by the third coordinate (depth)
    return pts_2d[:, :2] / pts_2d[:, 2:]


def ddd2locrot(
    center: np.ndarray,
    alpha: np.ndarray,
    dim_x: np.ndarray,
    depth: np.ndarray,
    calib: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D detection parameters (center, alpha, dimensions, depth)
    to 3D location and rotation_y in camera coordinates.

    Parameters
    ----------
    center
        2D center (x, y) of the object in the image. Shape (N, 2).
    alpha
        Observation angle of the object. Shape (N, 1).
    dim_x
        Height of the object. Shape (N,).
    depth
        Estimated depth of the object. Shape (N, 1).
    calib
        Camera projection matrix P (3x4).

    Returns
    -------
    locations : np.ndarray
        3D location (x, y, z) of the object's bottom-center. Shape (N, 3).
    rotation_y : np.ndarray
        Rotation around the Y-axis in camera coordinates. Shape (N, 1).
    """
    locations = unproject_2d_to_3d(center, depth, calib)
    # Adjust y-coordinate from center to bottom of the 3D bounding box
    locations[:, 1] += dim_x / 2
    rotation_y = alpha2rot_y(alpha, center[:, 0:1], calib[0, 2], calib[0, 0])
    return locations, rotation_y


def unproject_2d_to_3d(
    pt_2d: np.ndarray, depth: np.ndarray, P: np.ndarray
) -> np.ndarray:
    """
    Unprojects a 2D point with known depth back to 3D camera coordinates.

    Parameters
    ----------
    pt_2d
        2D point (u, v) in image coordinates. Shape (N, 2).
    depth
        The estimated depth (Z-coordinate) of the point
        in camera space. Shape (N, 1).
    P
        Camera projection matrix (P), typically 3x4.

    Returns
    -------
    point_3d : np.ndarray
        The 3D point (X, Y, Z) in camera coordinates. Shape (N, 3).
    """
    z = depth[:, 0] - P[2, 3]
    x = (pt_2d[:, 0] * depth[:, 0] - P[0, 3] - P[0, 2] * z) / P[0, 0]
    y = (pt_2d[:, 1] * depth[:, 0] - P[1, 3] - P[1, 2] * z) / P[1, 1]

    return np.array([x, y, z], dtype=np.float32).transpose(1, 0)


def alpha2rot_y(alpha: np.ndarray, x: np.ndarray, cx: float, fx: float) -> np.ndarray:
    """
    Converts observation angle `alpha` to rotation_y around the Y-axis.

    Parameters
    ----------
    alpha
        Observation angle of the object, ranging [-pi,pi]. Shape (N, 1).
        This is the angle between the ray from camera origin to object
        center and the Z-axis in camera coordinates.
    x
        X-coordinate of the object's center in pixel coordinates. Shape (N, 1).
    cx
        X-coordinate of the principal point (camera center) in pixels.
    fx
        Focal length in X-direction in pixels.

    Returns
    -------
    rotation_y : np.ndarray
        Rotation `rotation_y` around the Y-axis in camera coordinates,
        ranging [-pi,pi]. Shape (N, 1).
    """
    rot_y = alpha + np.arctan2(x - cx, fx)

    # Normalize rot_y to be within [-pi, pi]
    rot_y[rot_y > np.pi] -= 2 * np.pi
    rot_y[rot_y < -np.pi] += 2 * np.pi
    return rot_y


def rotation_matrix_to_euler(rot: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of rotation matrices to Euler angles (pitch-yaw-roll).

    Parameters
    ----------
    rot
        Batch of rotation matrices with shape [N, 3, 3].

    Returns
    -------
    torch.Tensor
        Euler angles in radians with shape [N, 3] as (pitch, yaw, roll).
    """
    batch = rot.shape[0]
    R = rot
    sy = torch.sqrt(R[:, 0, 0] ** 2 + R[:, 1, 0] ** 2).clamp(min=1e-12)
    singular = (sy < 1e-6).float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = torch.zeros(batch, device=rot.device)

    out = torch.zeros(batch, 3, device=rot.device)
    out[:, 0] = x * (1 - singular) + xs * singular
    out[:, 1] = y * (1 - singular) + ys * singular
    out[:, 2] = z * (1 - singular) + zs * singular
    return out


def transform_coordinates_3d(coords: np.ndarray, RT: np.ndarray) -> np.ndarray:
    """
    Apply a 4x4 homogeneous transform to (3, N) coordinates.

    Parameters
    ----------
    coords
        Input coordinates, shape (3, N), dtype float32.
    RT
        Homogeneous transformation matrix, shape (4, 4), dtype float32.

    Returns
    -------
    np.ndarray
        Transformed coordinates, shape (3, N), dtype float32.
    """
    assert coords.shape[0] == 3
    coords_with_ones = np.vstack(
        [coords, np.ones((1, coords.shape[1]), dtype=np.float32)]
    )
    return (RT @ coords_with_ones)[:3, :]


@functools.lru_cache(maxsize=4)
def _get_pixel_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_grid, v_grid) pixel coordinate arrays for a given resolution.

    Results are cached for up to 4 distinct (h, w) sizes.  The returned arrays
    are read-only to prevent callers from silently corrupting the cache.
    """
    xmap = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    ymap = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
    xmap.flags.writeable = False
    ymap.flags.writeable = False
    return xmap, ymap


def depth_to_full_pointcloud(
    depth_filled: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1000.0,
) -> np.ndarray:
    """
    Back-project a full depth image to a 3D point cloud.

    Parameters
    ----------
    depth_filled
        Depth image of shape (H, W) in raw uint16 units.
    fx
        Focal length x in pixels.
    fy
        Focal length y in pixels.
    cx
        Principal point x in pixels.
    cy
        Principal point y in pixels.
    depth_scale
        Factor to convert depth units to metres (default 1000.0 for mm→m).

    Returns
    -------
    np.ndarray
        Point cloud of shape (H, W, 3) in metres, float32.
    """
    h, w = depth_filled.shape
    xmap, ymap = _get_pixel_grids(h, w)

    pts_z = depth_filled / depth_scale
    pts_x = (xmap - cx) * pts_z / fx
    pts_y = (ymap - cy) * pts_z / fy

    return np.stack([pts_x, pts_y, pts_z], axis=2).astype(np.float32)


def build_camera_projection_matrix(
    fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """
    Build a 3x4 camera projection matrix from intrinsic parameters.

    Parameters
    ----------
    fx
        Focal length x in pixels.
    fy
        Focal length y in pixels.
    cx
        Principal point x in pixels.
    cy
        Principal point y in pixels.

    Returns
    -------
    np.ndarray
        Projection matrix of shape (3, 4), dtype float32.
    """
    return np.array(
        [
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
