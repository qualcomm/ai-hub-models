# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

# Default LR input spatial dimensions (must match model.py DEFAULT_HEIGHT/DEFAULT_WIDTH)
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64

# REDS4 test set (clips 000, 011, 015, 020 from REDS training set)
# Hosted on the QAIHM public asset store; original source: https://huggingface.co/datasets/Iceclear/REDS4
REDS4_FOLDER_NAME = "reds4"
REDS4_VERSION = 1
REDS4_ASSET = CachedWebDatasetAsset.from_asset_store(
    REDS4_FOLDER_NAME,
    REDS4_VERSION,
    "REDS.zip",
)

# The 4 test clips defined in the BasicVSR++ paper (from REDS training set)
REDS4_TEST_CLIPS = ["000", "011", "015", "020"]
FRAMES_PER_CLIP = 100
# Each sample is a consecutive 2-frame pair; last frame of each clip has no successor.
NUM_SAMPLES_DEFAULT = len(REDS4_TEST_CLIPS) * (FRAMES_PER_CLIP - 1)  # 396
SCALING_FACTOR = 4


class REDS4Dataset(BaseDataset):
    """REDS4 test dataset for video super-resolution evaluation.

    Uses clips 000, 011, 015, 020 from the REDS training set, as defined in
    the BasicVSR++ paper. Each sample is a consecutive 2-frame LR clip with
    the HR ground truth for the second frame (center-cropped to model input size):
      - LR: train_sharp_bicubic/X4/<clip>/<frame>.png, float32 [0, 1]
      - HR: train_sharp/<clip>/<frame>.png, float32 [0, 1]

    Source: QAIHM public asset store (original: https://huggingface.co/datasets/Iceclear/REDS4)

    Metric note: this dataset reports luma-weighted PSNR on 64x64 center crops
    of frame t+1 only.  The BasicVSR++ paper reports RGB PSNR on full 480x270
    frames over all 100 frames per clip; the numbers are not directly comparable.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        scaling_factor: int = SCALING_FACTOR,
    ) -> None:
        self.reds4_path = REDS4_ASSET.extracted_path
        self.scaling_factor = scaling_factor

        input_spec = input_spec or {
            "frame": TensorSpec(shape=(1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH))
        }
        key = next(iter(input_spec))
        shape = input_spec[key][0]
        self.input_height: int = shape[2]
        self.input_width: int = shape[3]

        BaseDataset.__init__(self, self.reds4_path, split, input_spec)

        # REDS4 has no train/val partition; both splits draw from the same 4 clips.
        self._samples: list[tuple[Path, Path, Path]] = []
        for clip in REDS4_TEST_CLIPS:
            lr_base = (
                self.reds4_path
                / "train_sharp_bicubic"
                / f"X{self.scaling_factor}"
                / clip
            )
            hr_base = self.reds4_path / "train_sharp" / clip
            for frame_idx in range(FRAMES_PER_CLIP - 1):
                self._samples.append(
                    (
                        lr_base / f"{frame_idx:08d}.png",
                        lr_base / f"{frame_idx + 1:08d}.png",
                        hr_base / f"{frame_idx + 1:08d}.png",
                    )
                )

    def _validate_data(self) -> bool:
        if not self.reds4_path.exists():
            return False
        for clip in REDS4_TEST_CLIPS:
            lr_base = (
                self.reds4_path
                / "train_sharp_bicubic"
                / f"X{self.scaling_factor}"
                / clip
            )
            hr_base = self.reds4_path / "train_sharp" / clip
            if not (lr_base / "00000000.png").exists():
                return False
            if not (lr_base / "00000099.png").exists():
                return False
            if not (hr_base / "00000099.png").exists():
                return False
        return True

    def _download_data(self) -> None:
        REDS4_ASSET.fetch(extract=True)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a 2-frame LR clip and the HR ground truth for the second frame.

        Parameters
        ----------
        item
            Sample index.

        Returns
        -------
        lr_clip : torch.Tensor
            ``(2, 3, H, W)`` float32, frames [t, t+1].
        hr_frame : torch.Tensor
            ``(3, 4H, 4W)`` float32, ground truth for frame t+1.
        """
        lr_path_t, lr_path_t1, hr_path = self._samples[item]

        def _load_lr(path: Path) -> torch.Tensor:
            pil = Image.open(path).convert("RGB")
            pw, ph = pil.size  # native LR width, height
            # Center-crop to model input size; stays on-distribution vs bicubic resize.
            cy = (ph - self.input_height) // 2
            cx = (pw - self.input_width) // 2
            pil = pil.crop((cx, cy, cx + self.input_width, cy + self.input_height))
            return torch.from_numpy(
                (np.asarray(pil, dtype=np.float32) / 255.0).transpose(2, 0, 1)
            ).contiguous()

        hr_pil = Image.open(hr_path).convert("RGB")
        hw, hh = hr_pil.size  # native HR width, height
        s = self.scaling_factor
        # Crop HR at the same spatial position scaled by s.
        cy_hr = (hh - self.input_height * s) // 2
        cx_hr = (hw - self.input_width * s) // 2
        hr_pil = hr_pil.crop(
            (cx_hr, cy_hr, cx_hr + self.input_width * s, cy_hr + self.input_height * s)
        )
        hr_tensor = torch.from_numpy(
            (np.asarray(hr_pil, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        ).contiguous()

        lr_clip = torch.stack([_load_lr(lr_path_t), _load_lr(lr_path_t1)], dim=0)
        return lr_clip, hr_tensor

    def get_dataloader(
        self, num_samples: int, samples_per_job: int | None = None
    ) -> DataLoader:
        count = min(num_samples, len(self))
        indices = list(np.linspace(0, len(self) - 1, count, dtype=int))
        return DataLoader(
            Subset(self, indices),
            batch_size=1,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def default_samples_per_job() -> int:
        # 200 pairs gives a representative PSNR sample across all 4 REDS4 clips.
        return 200

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/Iceclear/REDS4",
            split_description="REDS4 test clips 000/011/015/020 (396 consecutive 2-frame pairs, 4x bicubic LR)",
        )
