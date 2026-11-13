# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL.Image import Image

from qai_hub_models.models.templates.corevision.display import (
    is_headless,
    is_running_in_notebook,
)


def copy_configs(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    required_subdir: str,
) -> None:
    """
    Copy a SAM-family configuration directory into ``dst`` under the
    expected ``<required_subdir>/configs/<src name>`` layout.

    If ``dst`` is not already inside ``required_subdir``, the required
    subdirectory is appended so downstream hydra imports resolve.

    Parameters
    ----------
    src
        Path to the source configuration directory (e.g.
        ``<sam lib>/configs/sam2.1``).
    dst
        Base path where the configuration directory should be copied.
    required_subdir
        Package-relative subdirectory the configs must live under
        (e.g. ``"qai_hub_models/models/sam2"``).
    """
    src_path = Path(src)
    dst_path = Path(dst) / "configs" / src_path.name
    if required_subdir not in str(dst_path):
        dst_path = Path(dst) / required_subdir / "configs" / src_path.name
    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)


## Helper routines
def show_image(
    image: Image,
    masks: np.ndarray | torch.Tensor | None = None,
    input_coords: list[list[int]] | None = None,
    output_dir: str | None = None,
) -> None:
    """Show input image with mask applied"""
    if masks is None or input_coords is None:
        print("No masks or points detected, so no images to display.")
        return

    should_display_in_notebook = is_running_in_notebook()
    should_save_to_disk = not should_display_in_notebook and (
        is_headless() or output_dir is not None
    )

    if should_display_in_notebook:
        print(
            "In python notebooks, make sure inline plotting is enabled via running '%matplotlib inline'"
        )

    if should_save_to_disk:
        if output_dir is None:
            output_dir = str(Path.cwd() / "build")
        os.makedirs(output_dir, exist_ok=True)
        print("Saving images to disk: ")

    for mask_idx, mask in enumerate(np.split(masks, masks.shape[1], 1)):
        plt.figure(num=mask_idx, figsize=(10, 10))
        plt.title(f"Predicted Mask {mask_idx}")
        plt.imshow(image)
        x_coords_points = [p[0] for p in input_coords]
        y_coords_points = [p[1] for p in input_coords]
        plt.scatter(
            x_coords_points,
            y_coords_points,
            color="red",  # Color of the points
            marker="o",  # Marker style (e.g., 'o' for circle, 'x', '+', '*')
            s=100,  # Size of the markers
            label="Points",
        )
        _draw_mask(mask)
        plt.axis("off")
        if should_save_to_disk:
            assert output_dir is not None
            outpath = os.path.join(output_dir, f"demo_output_mask{mask_idx}.jpg")
            plt.savefig(
                outpath,
                bbox_inches="tight",
            )
            print(f"   {outpath}")

    if not should_save_to_disk:
        plt.show()


def _draw_mask(mask: np.ndarray) -> None:
    """Helper routine to add mask over existing plot"""
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    plt.imshow(mask_image)
