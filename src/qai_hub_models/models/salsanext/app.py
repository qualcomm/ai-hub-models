# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from qai_hub_models_cli.proto_helpers._common import safe_load_yaml

from qai_hub_models.models.salsanext.external_repos.salsanext.train.common.laserscan import (
    SemLaserScan,
)
from qai_hub_models.models.salsanext.external_repos.salsanext.train.tasks.semantic.postproc.KNN import (
    KNN,
)
from qai_hub_models.models.salsanext.model import (
    ARCH_ADDRESS,
    DATA_ADDRESS,
)


class SalsaNextApp:
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self._model = model
        with open(ARCH_ADDRESS) as f:
            self.arch = safe_load_yaml(f)
        with open(DATA_ADDRESS) as f:
            self.data = safe_load_yaml(f)
        self.nclasses = len(self.data["learning_map_inv"])

    def preprocess_lidar(self, lidar_input: str) -> tuple[torch.Tensor, Any]:
        color_map = self.data["color_map"]
        self.learning_map: Mapping[int, int] = self.data["learning_map"]

        sensor = self.arch["dataset"]["sensor"]
        img_H = sensor["img_prop"]["height"]
        img_W = sensor["img_prop"]["width"]
        img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        fov_up = sensor["fov_up"]
        fov_down = sensor["fov_down"]

        scan = SemLaserScan(
            color_map, project=True, H=img_H, W=img_W, fov_up=fov_up, fov_down=fov_down
        )
        scan.open_scan(lidar_input)

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)

        proj = torch.cat(
            [
                proj_range.unsqueeze(0),
                proj_xyz.permute(2, 0, 1),
                proj_remission.unsqueeze(0),
            ]
        )
        proj = (proj - img_means[:, None, None]) / img_stds[:, None, None]
        proj *= proj_mask.float()
        return proj.unsqueeze(0), scan

    def detect(self, lidar_input: str) -> torch.Tensor:
        proj, scan = self.preprocess_lidar(lidar_input)

        post = None
        if self.arch["post"]["KNN"]["use"]:
            post = KNN(self.arch["post"]["KNN"]["params"], self.nclasses)

        output = self._model(proj)
        proj_argmax = output[0].argmax(dim=0)

        npoints = scan.points.shape[0]
        unproj_range = torch.from_numpy(scan.unproj_range).clone()[:npoints]
        p_x = torch.from_numpy(scan.proj_x)[:npoints]
        p_y = torch.from_numpy(scan.proj_y)[:npoints]

        if post:
            unproj_argmax = post(
                torch.from_numpy(scan.proj_range), unproj_range, proj_argmax, p_x, p_y
            )
        else:
            unproj_argmax = proj_argmax[p_y, p_x]

        return unproj_argmax.to(torch.int32)

    def load_lidar_gt(self, input_path: str) -> torch.Tensor:
        self.gt_np = (
            np.fromfile(input_path, dtype=np.int32).reshape(-1) & 0xFFFF
        ).astype(np.int32)
        if hasattr(self.learning_map, "get"):
            self.gt_np = np.vectorize(self.learning_map.get)(self.gt_np).astype(
                np.int32
            )
        else:
            self.gt_np = np.array(
                [self.learning_map[x] for x in self.gt_np], dtype=np.int32
            )
        return torch.from_numpy(self.gt_np)
