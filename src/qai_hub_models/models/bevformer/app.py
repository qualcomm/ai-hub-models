# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import io
from collections.abc import Callable

import matplotlib
import numpy as np
import torch
from PIL import Image as PILImage
from PIL.Image import Image as PilImage

# Select the non-interactive Agg backend for headless rendering. This MUST run
# before importing matplotlib.pyplot below; do not let an imports-at-top cleanup
# reorder it or headless rendering breaks.
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from qai_hub_models.models.bevformer.model import (
    BEV_H,
    BEV_W,
    EMBED_DIMS,
)
from qai_hub_models.models.templates.corevision.bounding_box_processing_3d import (
    compute_corners,
    draw_3d_bbox,
)
from qai_hub_models.models.templates.corevision.image_processing_3d import (
    project_to_image,
)

# point_cloud_range from the BEVFormer-tiny model config (line 21):
# https://github.com/fundamentalvision/BEVFormer/blob/master/projects/configs/bevformer/bevformer_tiny.py
# Format: [xmin, ymin, zmin, xmax, ymax, zmax] in metres.
PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
POST_CENTER_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]

# nuScenes 10-class detection task. Maps class name -> (R, G, B) color in
# range [0, 255], the format expected by draw_3d_bbox.
OBJECT_CLASSES = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}
CLASS_NAMES = list(OBJECT_CLASSES.keys())

IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMG_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

SCORE_THRESHOLD = 0.5

CAM_NAMES = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]


def _class_color(cls_idx: int) -> tuple[float, float, float]:
    """Return the RGB color for a class index as matplotlib 0-1 floats."""
    r, g, b = OBJECT_CLASSES[CLASS_NAMES[cls_idx]]
    return (r / 255.0, g / 255.0, b / 255.0)


class BEVFormerApp:
    def __init__(
        self,
        model: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
        score_threshold: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        model
            BEVFormer model -> returns (bev_embed, all_cls_scores, all_bbox_preds).
        score_threshold
            Minimum class probability for a detection to be kept.
        """
        self.model = model
        self.score_threshold = score_threshold

    def predict_3d_boxes(
        self,
        image: torch.Tensor,
        can_bus: torch.Tensor,
        lidar2img: torch.Tensor,
        prev_bev: torch.Tensor | None = None,
        use_prev_bev: torch.Tensor | None = None,
        raw_output: bool = False,
    ) -> dict | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Parameters
        ----------
        image
            (num_cam, 3, H, W) float32 normalized multi-camera images.
        can_bus
            (18,) float32 ego-motion / CAN-bus vector.
        lidar2img
            (1, num_cam, 4, 4) float32 lidar-to-image projection matrices.
        prev_bev
            (1, BEV_H, BEV_W, EMBED_DIMS) float32 previous-frame BEV. Defaults to
            zeros (scene-start frame).
        use_prev_bev
            (1,) float32 flag: 0 = ignore prev_bev, 1 = use it. Defaults to 0.
        raw_output
            If True return raw model outputs as numpy arrays (for test.py golden
            comparison). If False return decoded detections dict (for demo.py).

        Returns
        -------
        dict | tuple[np.ndarray, np.ndarray, np.ndarray] | None
            raw_output=True: (bev_embed, all_cls_scores, all_bbox_preds) numpy arrays.
            raw_output=False: decoded detections dict, or None if nothing passes
            the score threshold.
        """
        if prev_bev is None:
            prev_bev = torch.zeros(1, BEV_H, BEV_W, EMBED_DIMS)
        if use_prev_bev is None:
            use_prev_bev = torch.zeros(1)

        with torch.no_grad():
            bev_embed, cls_scores, bbox_preds = self.model(
                image, use_prev_bev, prev_bev, can_bus, lidar2img
            )

        if raw_output:
            return (
                bev_embed.detach().cpu().numpy(),
                cls_scores.detach().cpu().numpy(),
                bbox_preds.detach().cpu().numpy(),
            )

        return self.decode_detections(cls_scores, bbox_preds)

    def decode_detections(
        self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor
    ) -> dict | None:
        """Decode raw model outputs into a filtered detections dict.

        Applies sigmoid to class logits, keeps detections above
        ``score_threshold``, converts log-encoded box dimensions, computes
        heading from sin/cos components, and filters by ``POST_CENTER_RANGE``.

        Parameters
        ----------
        cls_scores
            (num_dec, 1, num_query, num_classes) float32. Raw class logits
            from all decoder layers; only the last layer [-1] is used.
        bbox_preds
            (num_dec, 1, num_query, 10) float32. Raw box predictions from all
            decoder layers; only the last layer [-1] is used. Column layout:
            [cx, cy, log_w, log_l, cz, log_h, sin_rot, cos_rot, vx, vy].

        Returns
        -------
        dict | None
            Detection dict with keys x, y, z, l, w, h, rot, vx, vy, scores,
            labels (all torch.Tensor, N detections), or None if no detections
            pass the score threshold and center-range filter.
        """
        cls = cls_scores[-1, 0]  # (num_query, num_classes)
        bbox = bbox_preds[-1, 0]  # (num_query, 10)

        probs = cls.sigmoid()
        scores, labels = probs.max(dim=1)

        keep = scores > self.score_threshold
        scores, labels, bbox = scores[keep], labels[keep], bbox[keep]
        if scores.numel() == 0:
            return None

        x = bbox[:, 0]
        y = bbox[:, 1]
        z = bbox[:, 4]
        w = bbox[:, 2].exp()
        l = bbox[:, 3].exp()
        h = bbox[:, 5].exp()
        rot = torch.atan2(bbox[:, 6], bbox[:, 7])
        vx, vy = bbox[:, 8], bbox[:, 9]

        pcr = POST_CENTER_RANGE
        mask = (
            (x >= pcr[0])
            & (x <= pcr[3])
            & (y >= pcr[1])
            & (y <= pcr[4])
            & (z >= pcr[2])
            & (z <= pcr[5])
        )

        return dict(
            x=x[mask],
            y=y[mask],
            z=z[mask],
            l=l[mask],
            w=w[mask],
            h=h[mask],
            rot=rot[mask],
            vx=vx[mask],
            vy=vy[mask],
            scores=scores[mask],
            labels=labels[mask],
        )


def visualize_bev(detections: dict) -> PilImage:
    """Render a top-down bird's-eye-view plot of the detected boxes.

    X = forward (plotted up), Y = left. Each detection is drawn as a rotated
    rectangle colored by class, with a heading arrow and score label.

    Parameters
    ----------
    detections
        Decoded detections dict (from ``decode_detections``) with keys
        x, y, l, w, rot, scores, labels - each a 1-D tensor of length N
        (metres for x/y/l/w, radians for rot).

    Returns
    -------
    PilImage
        Rendered BEV plot as a PIL Image (use display_or_save_image to save/display).
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    # pc_range boundary
    rect = plt.Rectangle(
        (PC_RANGE[1], PC_RANGE[0]),
        PC_RANGE[4] - PC_RANGE[1],
        PC_RANGE[3] - PC_RANGE[0],
        linewidth=1.5,
        edgecolor="white",
        facecolor="none",
        linestyle="--",
        alpha=0.5,
    )
    ax.add_patch(rect)

    ax.plot(0, 0, marker="*", color="white", markersize=14, zorder=10)
    ax.text(
        0, 1.5, "EGO", color="white", fontsize=8, ha="center", va="bottom", zorder=10
    )

    x = detections["x"].numpy()
    y = detections["y"].numpy()
    l = detections["l"].numpy()
    w = detections["w"].numpy()
    rot = detections["rot"].numpy()
    scores = detections["scores"].numpy()
    labels = detections["labels"].numpy()

    legend_patches: dict[str, mpatches.Patch] = {}
    for i in range(len(x)):
        cx, cy = y[i], x[i]
        bw, bl = l[i], w[i]
        cls_idx = int(labels[i])
        color = _class_color(cls_idx)
        name = CLASS_NAMES[cls_idx]

        # World rot is CCW from world-X (forward). The BEV plot swaps axes:
        # plot-X = world-Y, plot-Y = world-X. A world angle of rot corresponds
        # to a plot angle of π/2 - rot, so swap sin/cos to compensate.
        cos_a, sin_a = np.sin(rot[i]), np.cos(rot[i])
        corners_x = np.array([-bw / 2, bw / 2, bw / 2, -bw / 2, -bw / 2])
        corners_y = np.array([-bl / 2, -bl / 2, bl / 2, bl / 2, -bl / 2])
        rx = corners_x * cos_a - corners_y * sin_a + cx
        ry = corners_x * sin_a + corners_y * cos_a + cy
        ax.plot(rx, ry, color=color, linewidth=1.2, alpha=0.9)

        ax.annotate(
            "",
            xy=(cx + cos_a * bw / 2, cy + sin_a * bw / 2),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        )
        ax.text(
            cx,
            cy,
            f"{scores[i]:.2f}",
            color=color,
            fontsize=8,
            ha="center",
            va="center",
        )
        if name not in legend_patches:
            legend_patches[name] = mpatches.Patch(color=color, label=name)

    ax.set_xlim(PC_RANGE[1] - 5, PC_RANGE[4] + 5)
    ax.set_ylim(PC_RANGE[0] - 5, PC_RANGE[3] + 5)
    ax.set_xlabel("Y — Left/Right (m)", color="white")
    ax.set_ylabel("X — Forward (m)", color="white")
    ax.set_title("BEVFormer — Bird's Eye View", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("white")
    ax.legend(
        handles=list(legend_patches.values()),
        loc="upper right",
        fontsize=7,
        facecolor="#1a1a2e",
        labelcolor="white",
    )
    ax.set_aspect("equal")
    ax.grid(True, color="white", alpha=0.1)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    buf.seek(0)
    return PILImage.open(buf).copy()


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization and convert to displayable uint8 images.

    Parameters
    ----------
    img_tensor
        (num_cam, 3, H, W) float32 tensor normalized with ImageNet
        mean/std (channel-first).

    Returns
    -------
    np.ndarray
        (num_cam, H, W, 3) uint8 array in range [0, 255] (channel-last).
    """
    img = img_tensor.numpy().transpose(0, 2, 3, 1)
    img = img * IMG_STD[None, None, None, :] + IMG_MEAN[None, None, None, :]
    return np.clip(img, 0, 255).astype(np.uint8)


def visualize_on_cameras(
    img_tensor: torch.Tensor,
    lidar2img_tensor: torch.Tensor,
    detections: dict,
) -> PilImage:
    """Draw 3D detection wireframe boxes on all six camera images.

    Projects each detection's 3D box onto every camera (skipping boxes
    behind or fully outside a given frame) and draws the 12 cube edges
    colored by class with a class/score label.

    Parameters
    ----------
    img_tensor
        (num_cam, 3, H, W) float32 normalized multi-camera images.
    lidar2img_tensor
        (1, num_cam, 4, 4) float32 lidar-to-image projection matrices.
    detections
        Decoded detections dict with keys x, y, z, l, w, h, rot, scores,
        labels, each a 1-D tensor of length N.

    Returns
    -------
    PilImage
        Camera-view composite as a PIL Image (use display_or_save_image to save/display).
    """
    imgs = denormalize_image(img_tensor)  # (6,H,W,3)
    l2i = lidar2img_tensor[0].numpy()  # (6,4,4)

    scores = detections["scores"].numpy()
    labels = detections["labels"].numpy()

    # Assemble [N, 7] boxes (x, y, z, w, l, h, rot) and build the 8 corners with
    # the shared compute_corners helper (returns [N, 8, 3] in the ego frame).
    boxes = torch.stack(
        [
            detections["x"],
            detections["y"],
            detections["z"],
            detections["w"],
            detections["l"],
            detections["h"],
            detections["rot"],
        ],
        dim=1,
    )
    corners_3d = compute_corners(boxes).numpy()  # (N, 8, 3), ego frame
    # compute_corners uses a bottom-origin ([0.5, 0.5, 0]) so z spans [cz, cz+h];
    # The decoded z is the box center, so shift down by h/2 to recenter
    corners_3d[..., 2] -= detections["h"].numpy()[:, None] / 2
    n_boxes = corners_3d.shape[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    for cam_idx in range(6):
        ax = axes[cam_idx]
        img = imgs[cam_idx].copy()
        H, W = img.shape[:2]
        P = l2i[cam_idx]  # (4,4) ego -> image-homogeneous

        box_corners = []
        box_labels = []
        text_anchors = []
        for i in range(n_boxes):
            pts3d = corners_3d[i]  # (8,3)
            # depth (camera z) before the perspective divide.
            depth = pts3d @ P[2, :3] + P[2, 3]
            if np.any(depth <= 1e-3):
                continue  # skip if any corner is at/behind the camera plane
                # (near-zero/negative depth projects to garbage edges)
            corners2d = project_to_image(pts3d, P[:3])  # (8,2)
            u, v = corners2d[:, 0], corners2d[:, 1]
            if np.all(u < 0) or np.all(u > W) or np.all(v < 0) or np.all(v > H):
                continue  # box fully outside this camera's frame
            box_corners.append(corners2d)
            box_labels.append(int(labels[i]))
            text_anchors.append((u.mean(), v.mean(), int(labels[i]), scores[i]))

        if box_corners:
            img = draw_3d_bbox(
                img,
                np.stack(box_corners),
                np.array(box_labels),
                OBJECT_CLASSES,
            )

        ax.imshow(img)
        ax.set_title(CAM_NAMES[cam_idx], fontsize=8, color="white", pad=3)
        ax.axis("off")

        for cx_text, cy_text, cls_idx, score in text_anchors:
            ax.text(
                cx_text,
                cy_text,
                f"{CLASS_NAMES[cls_idx]}\n{score:.2f}",
                color=_class_color(cls_idx),
                fontsize=5,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.4),
            )

    fig.patch.set_facecolor("#111111")
    plt.suptitle(
        "BEVFormer - 3D Detections Projected on Cameras", color="white", fontsize=11
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(
        buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    plt.close()
    buf.seek(0)
    return PILImage.open(buf).copy()
