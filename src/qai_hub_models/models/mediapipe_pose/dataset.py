# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch

from qai_hub_models.datasets.coco.coco import CocoDatasetBase
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.base_dataset import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


class CocoPosesCalibrationDataset(CocoDatasetBase):
    """
    Calibration-only dataset for MediaPipe Pose, sourced from MS COCO (CC-BY 4.0).

    Only images containing a person are kept, matching the pose input
    distribution so the landmark component receives valid detector ROIs.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        if split != DatasetSplit.TRAIN:
            raise ValueError(
                "CocoPosesCalibrationDataset only supports the TRAIN split."
            )

        # No train download; calibrate from val2017 (auto-downloadable, no
        # license wall) to keep `qai-hub-models export` self-service.
        super().__init__(
            split=DatasetSplit.VAL,
            input_spec=input_spec,
        )
        spec = input_spec or {}
        self.input_height = spec["image"][0][2] if "image" in spec else 128
        self.input_width = spec["image"][0][3] if "image" in spec else 128

        person_cat_ids = self.coco.getCatIds(catNms=["person"])
        person_image_ids = set(self.coco.getImgIds(catIds=person_cat_ids))
        self.ids = sorted(set(self.ids) & person_image_ids)
        if not self.ids:
            raise ValueError("No COCO images with person annotations found.")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self._load_image(self.ids[index])
        image = image.resize((self.input_width, self.input_height))
        image_tensor = app_to_net_image_inputs(image)[1].squeeze(0)
        return image_tensor, 0

    def __len__(self) -> int:
        return len(self.ids)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="val2017 person subset",
        )
