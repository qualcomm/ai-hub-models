# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import tarfile
from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import torch
from PIL import Image
from qai_hub_models_cli.proto_helpers._common import safe_load_yaml

_NUM_CLASSES = 20

# Lazily populated on first use to avoid importing model.py (and its heavy
# transitive deps such as boto3) at module-load time.
# Using a list avoids a module-level `global` statement (Ruff PLW0603).
_COLOR_LUT: list[np.ndarray] = []


def _build_color_lut() -> np.ndarray:
    """Build the RGB colour look-up table from the darknet53 ``data_cfg.yaml``."""
    # Deferred import: model.py pulls in asset_loaders → aws → boto3, which
    # may not be present in all environments.  We only need it here.
    from qai_hub_models.models.rangenet_plus_plus.model import (
        DARKNET53_MODEL_ASSET,
    )

    tar_path = DARKNET53_MODEL_ASSET.fetch()
    model_dir = tar_path.parent / "darknet53"
    if not model_dir.exists():
        with tarfile.open(tar_path) as tar:
            tar.extractall(tar_path.parent)

    with open(model_dir / "data_cfg.yaml") as f:
        cfg = safe_load_yaml(f)

    learning_map_inv = {int(k): int(v) for k, v in cfg["learning_map_inv"].items()}
    color_map_bgr = {int(k): tuple(v) for k, v in cfg["color_map"].items()}

    # color_map is stored as BGR in the YAML → reverse to RGB for PIL/numpy
    return np.array(
        [
            tuple(reversed(color_map_bgr[learning_map_inv[i]]))
            for i in range(_NUM_CLASSES)
        ],
        dtype=np.uint8,
    )


def _color_lut() -> np.ndarray:
    """Return the cached RGB colour look-up table, building it on first call."""
    if not _COLOR_LUT:
        _COLOR_LUT.append(_build_color_lut())
    return _COLOR_LUT[0]


def project_points_to_range_image(
    points: np.ndarray,
    H: int = 64,
    W: int = 2048,
    fov_up: float = 3.0,
    fov_down: float = -25.0,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project a (N, 4) point cloud [x, y, z, intensity] into a (1, 5, H, W) range image.

    Parameters
    ----------
    points
        (N, 4) float32 array of LiDAR points [x, y, z, intensity].
    H
        Number of rows in the range image (vertical resolution).
    W
        Number of columns in the range image (horizontal resolution).
    fov_up
        Upper vertical field-of-view boundary in degrees.
    fov_down
        Lower vertical field-of-view boundary in degrees.
    means
        Per-channel mean for normalisation; defaults to SemanticKITTI values.
    stds
        Per-channel standard deviation for normalisation; defaults to SemanticKITTI values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(range_image, u_idx, v_idx)`` where ``range_image`` is a
        (1, 5, H, W) float32 normalised tensor, ``u_idx`` is the (N,) column
        index and ``v_idx`` is the (N,) row index of each point.
    """
    # Per-channel mean and std for [depth, x, y, z, intensity] from the
    # SemanticKITTI dataset statistics in lidar-bonnetal sensor.yaml.
    if means is None:
        means = np.array([12.12, 10.88, 0.23, -1.04, 0.21], dtype=np.float32)
    if stds is None:
        stds = np.array([12.32, 11.47, 6.91, 0.86, 0.16], dtype=np.float32)

    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    fov_up_r = fov_up / 180.0 * np.pi
    fov_down_r = fov_down / 180.0 * np.pi
    fov = abs(fov_down_r) + abs(fov_up_r)

    depth = np.sqrt(x**2 + y**2 + z**2)
    yaw = -np.arctan2(y, x)
    # Clamp depth to 1e-5 to avoid division by zero in arcsin
    pitch = np.arcsin(z / np.clip(depth, 1e-5, None))

    # Map yaw in [-pi, pi] -> column index in [0, W-1]
    u = (0.5 * (yaw / np.pi + 1.0) * W).astype(int).clip(0, W - 1)
    # Map pitch relative to fov_down -> row index in [0, H-1]
    v = ((1.0 - (pitch + abs(fov_down_r)) / fov) * H).astype(int).clip(0, H - 1)

    img = np.zeros((5, H, W), dtype=np.float32)
    img[0, v, u] = depth
    img[1, v, u] = x
    img[2, v, u] = y
    img[3, v, u] = z
    img[4, v, u] = intensity
    img = (img - means[:, None, None]) / stds[:, None, None]
    return img[np.newaxis], u, v


class RangeNetApp:
    """
    End-to-end application for RangeNet++ LiDAR semantic segmentation.

    Supports range image segmentation and 3D bird's-eye-view (BEV) rendering.
    """

    def __init__(self, model: Callable[..., torch.Tensor]) -> None:
        self.model = model

    def predict(self, *args: Any, **kwargs: Any) -> Image.Image | np.ndarray:
        return self.segment_range_image(*args, **kwargs)

    @overload
    def segment_range_image(
        self,
        range_image: np.ndarray | torch.Tensor,
        raw_output: Literal[True],
    ) -> np.ndarray: ...

    @overload
    def segment_range_image(
        self,
        range_image: np.ndarray | torch.Tensor,
        raw_output: Literal[False] = ...,
    ) -> Image.Image: ...

    def segment_range_image(
        self,
        range_image: np.ndarray | torch.Tensor,
        raw_output: bool = False,
    ) -> Image.Image | np.ndarray:
        """
        Run inference on a pre-projected range image.

        Parameters
        ----------
        range_image
            (1, 5, H, W) or (5, H, W) float32 tensor.
        raw_output
            If True, return the raw class-index mask array of shape [1, H, W].

        Returns
        -------
        Image.Image | np.ndarray
            Colour-coded PIL Image, or class-index mask array when
            ``raw_output=True``.
        """
        if isinstance(range_image, np.ndarray):
            range_image = torch.from_numpy(range_image)
        if range_image.dim() == 3:
            range_image = range_image.unsqueeze(0)

        mask = self.model(range_image)

        if raw_output:
            return mask.cpu().numpy()

        pred = torch.argmax(mask[0], dim=0).cpu().numpy().clip(0, _NUM_CLASSES - 1)
        colored = _color_lut()[pred]
        img = Image.fromarray(colored)
        return img.resize((img.width, img.height * 8), Image.NEAREST)

    def reproject_to_3d(
        self,
        points: np.ndarray,
        H: int = 64,
        W: int = 2048,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project point cloud to range image, run inference, and re-project labels back.

        Parameters
        ----------
        points
            (N, 4) float32 LiDAR points [x, y, z, intensity].
        H
            Number of rows in the range image (vertical resolution).
        W
            Number of columns in the range image (horizontal resolution).
        fov_up
            Upper vertical field-of-view boundary in degrees.
        fov_down
            Lower vertical field-of-view boundary in degrees.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(xyz, labels, colors)`` — (N, 3) 3-D coordinates, (N,) compact
            class id per point, and (N, 3) RGB colour per point.
        """
        range_image, u_idx, v_idx = project_points_to_range_image(
            points, H=H, W=W, fov_up=fov_up, fov_down=fov_down
        )
        mask = self.model(torch.from_numpy(range_image))

        pred_2d = torch.argmax(mask[0], dim=0).cpu().numpy().clip(0, _NUM_CLASSES - 1)
        point_labels = pred_2d[v_idx, u_idx]
        lut = _color_lut()
        return points[:, :3], point_labels, lut[point_labels]

    def bird_eye_view(
        self,
        points: np.ndarray,
        resolution: float = 0.1,
        x_range: tuple[float, float] = (-50, 50),
        y_range: tuple[float, float] = (-50, 50),
        **kwargs: Any,
    ) -> Image.Image:
        """
        Render a top-down bird's-eye-view (BEV) image coloured by semantic class.

        Parameters
        ----------
        points
            (N, 4) float32 LiDAR points [x, y, z, intensity].
        resolution
            Metres per pixel (default 0.1 m/px).
        x_range
            Forward/back range in metres.
        y_range
            Left/right range in metres.
        **kwargs
            Additional keyword arguments forwarded to ``reproject_to_3d``
            (e.g. ``H``, ``W``, ``fov_up``, ``fov_down``).

        Returns
        -------
        Image.Image
            Top-down BEV PIL Image with semantic colours.
        """
        xyz, _, colors = self.reproject_to_3d(points, **kwargs)
        return self._render_bev(xyz, colors, resolution, x_range, y_range)

    def segment_and_bev(
        self,
        points: np.ndarray,
        range_image: np.ndarray | None = None,
        resolution: float = 0.1,
        x_range: tuple[float, float] = (-50, 50),
        y_range: tuple[float, float] = (-50, 50),
        H: int = 64,
        W: int = 2048,
        fov_up: float = 3.0,
        fov_down: float = -25.0,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Run inference once and return both the segmentation image and BEV image.

        Parameters
        ----------
        points
            (N, 4) float32 LiDAR points [x, y, z, intensity].
        range_image
            Pre-projected (1, 5, H, W) float32 array. Re-projected from
            points if not provided.
        resolution
            Metres per pixel for the BEV canvas.
        x_range
            Forward/back range in metres.
        y_range
            Left/right range in metres.
        H
            Number of rows in the range image (ignored when range_image is provided).
        W
            Number of columns in the range image (ignored when range_image is provided).
        fov_up
            Upper vertical field-of-view boundary in degrees (ignored when range_image is provided).
        fov_down
            Lower vertical field-of-view boundary in degrees (ignored when range_image is provided).

        Returns
        -------
        tuple[Image.Image, Image.Image]
            (seg_image, bev_image) — colour-coded segmentation image and top-down BEV image.
        """
        ri_arr, u_idx, v_idx = project_points_to_range_image(
            points, H=H, W=W, fov_up=fov_up, fov_down=fov_down
        )
        if range_image is not None:
            # Use the caller-supplied range image for inference but keep the
            # projection indices so we can map predictions back to points.
            ri_arr = range_image

        ri_tensor = (
            torch.from_numpy(ri_arr) if isinstance(ri_arr, np.ndarray) else ri_arr
        )
        if ri_tensor.dim() == 3:
            ri_tensor = ri_tensor.unsqueeze(0)

        mask = self.model(ri_tensor)

        lut = _color_lut()
        pred_2d = torch.argmax(mask[0], dim=0).cpu().numpy().clip(0, _NUM_CLASSES - 1)

        # Segmentation image from the 2-D prediction grid
        seg_img = Image.fromarray(lut[pred_2d])
        seg_img = seg_img.resize((seg_img.width, seg_img.height * 8), Image.NEAREST)

        # BEV image — reuse per-point labels from the same pred_2d
        point_labels = pred_2d[v_idx, u_idx]
        bev_img = self._render_bev(
            points[:, :3], lut[point_labels], resolution, x_range, y_range
        )

        return seg_img, bev_img

    def _render_bev(
        self,
        xyz: np.ndarray,
        colors: np.ndarray,
        resolution: float = 0.1,
        x_range: tuple[float, float] = (-50, 50),
        y_range: tuple[float, float] = (-50, 50),
    ) -> Image.Image:
        """
        Render a top-down bird's-eye-view canvas from pre-computed coordinates and colours.

        Parameters
        ----------
        xyz
            (N, 3) float32 array of 3-D point coordinates [x, y, z].
        colors
            (N, 3) uint8 array of RGB colours, one per point.
        resolution
            Metres per pixel (default 0.1 m/px).
        x_range
            (min, max) forward/back extent in metres defining the canvas height.
        y_range
            (min, max) left/right extent in metres defining the canvas width.

        Returns
        -------
        Image.Image
            Top-down BEV PIL Image (RGB) of size
        """
        bev_W = int((y_range[1] - y_range[0]) / resolution)
        bev_H = int((x_range[1] - x_range[0]) / resolution)
        canvas = np.zeros((bev_H, bev_W, 3), dtype=np.uint8)
        col = ((xyz[:, 1] - y_range[0]) / resolution).astype(int)
        row = ((x_range[1] - xyz[:, 0]) / resolution).astype(int)
        mask = (col >= 0) & (col < bev_W) & (row >= 0) & (row < bev_H)
        canvas[row[mask], col[mask]] = colors[mask]
        return Image.fromarray(canvas)
