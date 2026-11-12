# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    numpy_image_to_torch,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

HPATCHES_FOLDER_NAME = "hpatches"
HPATCHES_VERSION = 1
HPATCHES_ASSET = CachedWebDatasetAsset(
    "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/"
    "hpatches-sequences-release.zip",
    HPATCHES_FOLDER_NAME,
    HPATCHES_VERSION,
    "hpatches-sequences-release.zip",
)

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640


def _find_sequences_root(base: Path) -> Path:
    """The HPatches zip nests its sequence folders one level deep; locate them."""
    if any(base.glob("i_*")) or any(base.glob("v_*")):
        return base
    for child in sorted(base.iterdir()):
        if child.is_dir() and (any(child.glob("i_*")) or any(child.glob("v_*"))):
            return child
    raise FileNotFoundError(f"Could not locate HPatches sequence folders under {base}")


DEFAULT_SAMPLE_SEQUENCE = "v_adam"


def get_sample_image_pair_paths(
    sequence: str = DEFAULT_SAMPLE_SEQUENCE,
) -> tuple[str, str]:
    """
    A real HPatches viewpoint-change pair (image 1 and image 3 of ``sequence``
    -- a planar poster shot head-on and at an oblique angle, which gives an
    unambiguous visual check of correct matching). Used as the default demo
    input and as the model's real sample input for export/PSNR checks, since
    many other HPatches sequences contain repetitive textures (tiled art,
    brick walls) where correct matches still connect distant, visually
    similar patches.
    """
    HPATCHES_ASSET.fetch(extract=True)
    seq_root = _find_sequences_root(HPATCHES_ASSET.extracted_path)
    seq_dir = seq_root / sequence
    if not seq_dir.is_dir():
        seq_dir = sorted(
            d for d in seq_root.iterdir() if d.is_dir() and d.name.startswith("v_")
        )[0]
    return str(seq_dir / "1.ppm"), str(seq_dir / "3.ppm")


class HPatchesDataset(BaseDataset):
    """
    HPatches sequences benchmark: pairs of images related by a known homography,
    used to evaluate keypoint detector/descriptor repeatability and matching score.

    Each sequence directory contains 6 images (``1.ppm`` .. ``6.ppm``) and 5
    homography files (``H_1_2`` .. ``H_1_6``) mapping image 1's original pixel
    coordinates to each other image's original pixel coordinates. Sequences
    prefixed ``i_`` capture illumination changes; ``v_`` capture viewpoint changes.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TEST,
        input_spec: InputSpec | None = None,
        alteration: str = "all",
    ) -> None:
        # HPatches only ships a single benchmark split; split is accepted for
        # API compatibility but ignored.
        BaseDataset.__init__(self, HPATCHES_ASSET.extracted_path, split, input_spec)

        if input_spec is not None:
            _, _, self.height, self.width = input_spec["image_a"][0]
        else:
            self.height, self.width = DEFAULT_HEIGHT, DEFAULT_WIDTH

        assert alteration in ("all", "i", "v")
        self.alteration = alteration

        seq_root = _find_sequences_root(HPATCHES_ASSET.extracted_path)
        prefixes = ["i_", "v_"] if alteration == "all" else [f"{alteration}_"]
        seq_dirs = sorted(
            d
            for d in seq_root.iterdir()
            if d.is_dir() and any(d.name.startswith(p) for p in prefixes)
        )

        self._pairs: list[tuple[Path, int]] = [
            (seq_dir, k) for seq_dir in seq_dirs for k in range(2, 7)
        ]

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self, idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        seq_dir, k = self._pairs[idx]

        image_a_pil = Image.open(seq_dir / "1.ppm").convert("RGB")
        image_b_pil = Image.open(seq_dir / f"{k}.ppm").convert("RGB")
        homography_orig = torch.from_numpy(np.loadtxt(seq_dir / f"H_1_{k}")).float()

        w1, h1 = image_a_pil.size
        wk, hk = image_b_pil.size

        image_a = image_a_pil.resize((self.width, self.height), Image.BILINEAR)
        image_b = image_b_pil.resize((self.width, self.height), Image.BILINEAR)
        image_a_t = numpy_image_to_torch(np.array(image_a))[0]
        image_b_t = numpy_image_to_torch(np.array(image_b))[0]

        # The homography is defined in the images' original pixel coordinates.
        # Rescale it so it's valid in the resized (model input) coordinate frame:
        # H' = Sb @ H @ inv(Sa), where Sa/Sb map original -> resized coordinates.
        scale_a = torch.tensor(
            [[self.width / w1, 0.0, 0.0], [0.0, self.height / h1, 0.0], [0.0, 0.0, 1.0]]
        )
        scale_b = torch.tensor(
            [[self.width / wk, 0.0, 0.0], [0.0, self.height / hk, 0.0], [0.0, 0.0, 1.0]]
        )
        homography = scale_b @ homography_orig @ torch.inverse(scale_a)

        return (image_a_t, image_b_t), homography

    def _download_data(self) -> None:
        HPATCHES_ASSET.fetch(extract=True)

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @classmethod
    def dataset_name(cls) -> str:
        return "hpatches"

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/hpatches/hpatches-dataset",
            split_description="Full HPatches sequences benchmark (116 sequences x 5 pairs)",
        )
