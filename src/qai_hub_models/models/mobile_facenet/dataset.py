# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

import torch

from qai_hub_models.models.mobile_facenet.external_repos.pytorch_mobile_facenet.facenet_utils import (
    align_face,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

LFW_FOLDER_NAME = "lfw"
LFW_VERSION = 1
LFW_ASSET = CachedWebDatasetAsset.from_asset_store(
    LFW_FOLDER_NAME,
    LFW_VERSION,
    "lfw_ds.tar.gz",
)
LFW_JSON = CachedWebDatasetAsset.from_asset_store(
    LFW_FOLDER_NAME,
    LFW_VERSION,
    "lfw_funneled.json",
)


def _load_aligned_tensor(
    parent_dir: Path, samples: list[dict], token: str
) -> torch.Tensor:
    """Align and transform one face given a partial path token from the pair file."""
    matched = [s for s in samples if token in s["full_path"].replace("\\", "/")]
    if not matched:
        raise KeyError(f"No sample found for token: {token}")
    sample = matched[0]

    full_path = parent_dir / sample["full_path"]

    img_bgr = align_face(full_path, sample["landmarks"])  # 112x112 BGR
    img_rgb = img_bgr[..., ::-1].copy()
    return numpy_image_to_torch(img_rgb)[0]


class LFWDataset(BaseDataset):
    """LFW (Labeled Faces in the Wild) verification dataset."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        input_spec: InputSpec | None = None,
    ) -> None:
        # LFW only ships a single test split; split is accepted for API
        # compatibility but ignored — calibration jobs will reuse test data.
        BaseDataset.__init__(self, LFW_ASSET.extracted_path, split, input_spec)

        self.json_path = LFW_JSON.path
        self.pairs_path = LFW_ASSET.extracted_path / "lfw_test_pair.txt"

        with open(self.json_path) as f:
            data = json.load(f)
        self._samples: list[dict] = data["samples"]

        with open(self.pairs_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # Each line: "<path1> <path2> <label>"
        self._pairs: list[tuple[str, str, int]] = []
        for line in lines:
            tokens = line.split()
            self._pairs.append((tokens[0], tokens[1], int(tokens[2])))

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[tuple[torch.Tensor, torch.Tensor], int]:
        token1, token2, label = self._pairs[idx]
        img1 = _load_aligned_tensor(LFW_ASSET.extracted_path, self._samples, token1)
        img2 = _load_aligned_tensor(LFW_ASSET.extracted_path, self._samples, token2)
        return (img1, img2), label

    def _download_data(self) -> None:
        LFW_JSON.fetch()
        LFW_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 500

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="http://vis-www.cs.umass.edu/lfw/",
            split_description="LFW standard test split (6000 pairs)",
        )
