# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from qai_hub_models.datasets.coco.coco_keypoints import CocoKeypointsDataset
from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.models.templates.pose.topdown import letterbox
from qai_hub_models.utils.base_dataset import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class CocoHRNetPoseDataset(CocoKeypointsDataset):
    """Full COCO scenes letterboxed for the RTMDet component."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
    ) -> None:
        super().__init__(split, input_spec)
        # Keep negative scenes so false-positive person detections are scored too.
        self._samples = [(image_id, 1) for image_id in self._id_to_filename]

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        image_id, category_id = self._samples[index]
        image = Image.open(self.image_dir / self._id_to_filename[image_id]).convert(
            "RGB"
        )
        padded, scale = letterbox(
            image=np.asarray(image), target_hw=(self.target_h, self.target_w)
        )
        return numpy_image_to_torch(padded).squeeze(0), (
            torch.tensor(image_id, dtype=torch.int64),
            torch.tensor(category_id, dtype=torch.int64),
            torch.tensor(scale, dtype=torch.float32),
            torch.zeros(2, dtype=torch.int64),
        )

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 split - full scene images",
        )
