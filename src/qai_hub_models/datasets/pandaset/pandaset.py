# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

PANDASET_ID = "pandaset"


class PandaSetDatasetBase(BaseDataset):
    """Asset loading and frame indexing shared by the PandaSet LiDAR and BEV datasets."""

    _frames: Sequence[object]

    def __init__(
        self,
        asset: CachedWebDatasetAsset,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self._asset = asset
        self.data_root = asset.extracted_path
        super().__init__(self.data_root, split=split, input_spec=input_spec)

    def _download_data(self) -> None:
        self._asset.fetch(extract=True)

    @abstractmethod
    def _build_frame_list(self) -> None:
        """Index the frames available for this sensor format."""

    def _validate_data(self) -> bool:
        self._build_frame_list()
        return bool(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    @staticmethod
    def default_samples_per_job() -> int:
        return 50
