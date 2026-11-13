# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.datasets.kitti_lidar_place_recognition.kitti_lidar_place_recognition import (
    lidar_scan_to_range_image,
)
from qai_hub_models.datasets.nclt_lidar_place_recognition.nclt_lidar_place_recognition import (
    NCLT_LPR_FOV_DOWN_DEG,
    NCLT_LPR_FOV_UP_DEG,
    fill_empty_rows,
)
from qai_hub_models.models.overlap_transformer.model import INPUT_HEIGHT, INPUT_WIDTH
from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.input_spec import InputSpec


class OverlapTransformerApp:
    """
    Lightweight application for OverlapTransformer LiDAR place recognition.

    Projects raw point clouds into spherical range images and generates
    L2-normalized global descriptors, suitable for retrieval-based loop
    closure / revisit detection across LiDAR scans.
    """

    def __init__(
        self,
        model: ExecutableModelProtocol[torch.Tensor],
        input_spec: InputSpec | None = None,
        fov_up_deg: float = NCLT_LPR_FOV_UP_DEG,
        fov_down_deg: float = NCLT_LPR_FOV_DOWN_DEG,
        fill_rows: bool = True,
    ) -> None:
        self.model = model
        if input_spec is not None:
            _, _, input_height, input_width = input_spec["range_image"][0]
        else:
            input_height, input_width = INPUT_HEIGHT, INPUT_WIDTH
        self.input_height: int = input_height
        self.input_width: int = input_width
        self.fov_up_deg = fov_up_deg
        self.fov_down_deg = fov_down_deg
        # NCLT's 32 beams need row-fill to match the density KITTI's 64 beams give
        # the checkpoint natively; pass fill_rows=False alongside KITTI's FOV.
        self.fill_rows = fill_rows

    def predict(self, *scans: np.ndarray) -> np.ndarray:
        """
        Generate place-recognition descriptors for one or more LiDAR scans.

        Parameters
        ----------
        *scans
            One or more raw point clouds, each a numpy array of shape
            [N, 4+] (x, y, z, ...), projected using this app's fov_up_deg
            / fov_down_deg (defaults match NCLT velodyne_sync scans).

        Returns
        -------
        descriptors : np.ndarray
            Array of shape [N, 256] containing one L2-normalized global
            descriptor per input scan.
        """
        # One scan per forward call: the model is compiled at batch_size=1,
        # so batching multiple scans together breaks on-device inference.
        descriptors = []
        for scan in scans:
            range_image = lidar_scan_to_range_image(
                scan,
                self.input_height,
                self.input_width,
                fov_up_deg=self.fov_up_deg,
                fov_down_deg=self.fov_down_deg,
            )
            if self.fill_rows:
                range_image = fill_empty_rows(range_image)
            batch = torch.from_numpy(range_image).unsqueeze(0).unsqueeze(0)
            descriptors.append(self.model(batch.contiguous()))
        return np.asarray(torch.cat(descriptors, dim=0).detach().cpu())
