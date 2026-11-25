# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from qai_hub_models.models.igev_plusplus.constants import DEFAULT_HEIGHT, DEFAULT_WIDTH
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

SCENEFLOW_FOLDER_NAME = "sceneflow"
SCENEFLOW_VERSION = 1

# 600-pair FlyingThings3D TEST subset (100 calibration + 500 evaluation).
# Stored in private S3 due to SceneFlow research-only license.
SCENEFLOW_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{SCENEFLOW_FOLDER_NAME}/v{SCENEFLOW_VERSION}/ft3d_test_600.tar.gz",
    SCENEFLOW_FOLDER_NAME,
    SCENEFLOW_VERSION,
    "ft3d_test_600.tar.gz",
    installation_steps=[
        "Download the FlyingThings3D subset of SceneFlow (frames_finalpass + disparity) from "
        "https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html. "
        "This dataset requires a pre-packaged 600-pair archive (ft3d_test_600.tar.gz) with "
        "the frames_finalpass RGB images and disparity PFM files. "
        "Contact your team admin for access to the archive, then run: "
        "qai-hub-models configure-dataset SceneFlowFT3DDataset ft3d_test_600.tar.gz",
    ],
    configure_files=["ft3d_test_600.tar.gz"],
)

# Non-overlapping split sizes within the 600-pair archive.
_NUM_CAL_PAIRS = 100
_NUM_EVAL_PAIRS = 500


def _read_pfm(path: str | Path) -> np.ndarray:
    """Read a .pfm disparity file and return a float32 H x W numpy array."""
    with open(path, "rb") as f:
        header = f.readline().decode("utf-8").rstrip()
        assert header in ("Pf", "PF"), f"Unsupported PFM header: {header}"
        w, h = map(int, f.readline().decode("utf-8").rstrip().split())
        scale = float(f.readline().decode("utf-8").rstrip())
        little_endian = scale < 0
        data = np.frombuffer(f.read(), dtype="<f4" if little_endian else ">f4")
    data = data.reshape(h, w)
    return np.flipud(data).astype(np.float32)


class SceneFlowFT3DDataset(BaseDataset):
    """600-pair FlyingThings3D TEST subset of SceneFlow for stereo disparity.

    Source: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html
    License: Research use only. Commercial use prohibited. Cite Mayer et al. CVPR 2016.

    Split behaviour:
        DatasetSplit.TRAIN -> first 100 pairs  (calibration)
        DatasetSplit.VAL   -> remaining 500 pairs (float / quantized evaluation)

    Each item returns:
        inputs : (left_image, right_image) float32 tensors [3, H, W] in [0, 1]
        gt     : disparity tensor [H, W] float32 in pixels
    """

    @classmethod
    def dataset_name(cls) -> str:
        return "sceneflow_ft3d"

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        input_data_tar: str | None = None,
    ) -> None:
        self.data_path = SCENEFLOW_ASSET.extracted_path
        self.input_data_tar = input_data_tar
        BaseDataset.__init__(self, self.data_path, split, input_spec)

        # Deterministic sorted list of all 600 left images in the archive
        all_left = sorted((self.data_path / "frames_finalpass").rglob("*/left/*.png"))

        # Non-overlapping split — sorted order keeps first 100 pairs
        # disjoint from the next 500; do not change the sort.
        if split == DatasetSplit.TRAIN:
            selected = all_left[:_NUM_CAL_PAIRS]
        else:
            # VAL or TEST -> evaluation subset
            selected = all_left[_NUM_CAL_PAIRS : _NUM_CAL_PAIRS + _NUM_EVAL_PAIRS]

        self.left_files = selected
        self.right_files = [p.parent.parent / "right" / p.name for p in self.left_files]
        self.disp_files = [
            self.data_path
            / "disparity"
            / p.relative_to(self.data_path / "frames_finalpass").with_suffix(".pfm")
            for p in self.left_files
        ]

    def __len__(self) -> int:
        return len(self.left_files)

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        index
            Index of the sample to load.

        Returns
        -------
        tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            - inputs: (left_image, right_image) float32 tensors [3, H, W] in [0, 1]
            - gt: disparity tensor [H, W] float32 in pixels;
                  invalid pixels are marked -1.
        """
        H = (
            self.input_spec["left_image"][0][2]
            if self.input_spec is not None
            else DEFAULT_HEIGHT
        )
        W = (
            self.input_spec["left_image"][0][3]
            if self.input_spec is not None
            else DEFAULT_WIDTH
        )

        # Use cv2.resize on uint8 to match app.py preprocessing path.
        left_np = cv2.resize(
            np.array(Image.open(self.left_files[index]).convert("RGB")),
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )
        right_np = cv2.resize(
            np.array(Image.open(self.right_files[index]).convert("RGB")),
            (W, H),
            interpolation=cv2.INTER_LINEAR,
        )
        left = torch.from_numpy(left_np.astype(np.float32) / 255.0).permute(2, 0, 1)
        right = torch.from_numpy(right_np.astype(np.float32) / 255.0).permute(2, 0, 1)

        disp_np = _read_pfm(self.disp_files[index])
        orig_w = disp_np.shape[1]
        # FlyingThings3D disparities are positive (min > 1.0 empirically);
        # pixels with disp <= 0 or >= 192 are marked -1 as invalid so the
        # evaluator (valid = gt >= 0) can mask them correctly.
        valid_mask = (disp_np > 0) & (disp_np < 192.0)
        disp_np = np.where(valid_mask, disp_np, -1.0).astype(np.float32)
        disp = torch.from_numpy(disp_np)
        disp = (
            F.interpolate(
                disp.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="nearest",
                align_corners=None,
            )
            .squeeze(0)
            .squeeze(0)
        )
        # Scale disparity values to match new pixel coordinate space.
        # Only scale valid (>= 0) pixels; invalid (-1) remain -1.
        valid = disp >= 0
        disp[valid] = disp[valid] * (W / orig_w)

        return (left, right), disp

    def _download_data(self) -> None:
        SCENEFLOW_ASSET.fetch(extract=True, local_path=self.input_data_tar)

    def _validate_data(self) -> bool:
        if not (
            self.data_path.exists()
            and (self.data_path / "frames_finalpass").exists()
            and (self.data_path / "disparity").exists()
        ):
            return False
        left_files = sorted((self.data_path / "frames_finalpass").rglob("*/left/*.png"))
        if len(left_files) < 600:
            return False
        for p in left_files[:600]:
            right = p.parent.parent / "right" / p.name
            disp = (
                self.data_path
                / "disparity"
                / p.relative_to(self.data_path / "frames_finalpass").with_suffix(".pfm")
            )
            if not right.exists() or not disp.exists():
                return False
        return True

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        cls(input_data_tar=str(files[0]))

    @staticmethod
    def default_samples_per_job() -> int:
        # Each stereo pair (left + right) at DEFAULT_HEIGHT x DEFAULT_WIDTH
        # float32 = ~4.5 MB; 500 samples uses the full VAL split for
        # consistent results matching manifest numerics_benchmark (EPE=0.5546px).
        return 500

    @staticmethod
    def default_num_calibration_samples() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html",
            split_description="FlyingThings3D TEST 600-pair subset of SceneFlow (frames_finalpass). First 100 pairs used for calibration, remaining 500 for evaluation.",
        )
