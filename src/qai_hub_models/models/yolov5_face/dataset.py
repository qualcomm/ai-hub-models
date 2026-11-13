# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import warnings
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    resize_pad,
)
from qai_hub_models.models.yolov5_face.app import PAD_VALUE
from qai_hub_models.models.yolov5_face.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec

# WIDER FACE is loaded directly from its archives rather than through the
# HuggingFace `datasets` library: the `CUHK-CSE/wider_face` repo relies on a
# `wider_face.py` loading script, and `datasets>=4.0` (pinned by the base
# package) removed dataset-script support entirely. Downloading and parsing the
# raw archives keeps this model on the shared `datasets` version and avoids a
# per-model dependency pin.
_HF_DATA_URL = "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data"
WIDERFACE_FOLDER_NAME = "widerface"
WIDERFACE_VERSION = 1

# Annotation archive (plain-text bbox ground truth for train/val).
WIDERFACE_SPLIT_ASSET = CachedWebDatasetAsset(
    f"{_HF_DATA_URL}/wider_face_split.zip",
    WIDERFACE_FOLDER_NAME,
    WIDERFACE_VERSION,
    "wider_face_split.zip",
)

# Image archives, keyed by split. WIDER FACE has no public test-set ground
# truth, so only train/val are supported for evaluation.
_IMAGE_ASSETS = {
    DatasetSplit.TRAIN: CachedWebDatasetAsset(
        f"{_HF_DATA_URL}/WIDER_train.zip",
        WIDERFACE_FOLDER_NAME,
        WIDERFACE_VERSION,
        "WIDER_train.zip",
    ),
    DatasetSplit.VAL: CachedWebDatasetAsset(
        f"{_HF_DATA_URL}/WIDER_val.zip",
        WIDERFACE_FOLDER_NAME,
        WIDERFACE_VERSION,
        "WIDER_val.zip",
    ),
}

# Ground-truth annotation filename within the split archive, keyed by split.
_GT_FILENAME = {
    DatasetSplit.TRAIN: "wider_face_train_bbx_gt.txt",
    DatasetSplit.VAL: "wider_face_val_bbx_gt.txt",
}

# Maximum number of faces per image (for fixed-tensor padding).
# The densest WIDER FACE images have ~1500 faces; 500 covers 99.9 % of samples.
DEFAULT_MAX_FACES = 500


def _make_ignore(num_faces: int, keep_index: np.ndarray) -> np.ndarray:
    """Build a per-face ignore array from a 1-based keep_index (from mat file)."""
    ig = np.zeros(num_faces, dtype=np.float32)
    if len(keep_index) > 0:
        ig[keep_index.ravel() - 1] = 1.0
    return ig


class WiderFaceDataset(BaseDataset):
    """
    WIDER FACE dataset loaded directly from the official archives.

    For the VAL split, ground-truth boxes and per-difficulty ignore arrays are
    derived from the four official ``.mat`` files bundled in the external repo
    (``widerface_evaluate/ground_truth/``).  This allows the evaluator to
    compute Easy / Medium / Hard AP without any file I/O at evaluation time.

    For the TRAIN split (calibration only), boxes come from the text GT file
    and all-ones ignore arrays are returned (no difficulty split on train).

    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        max_faces: int = DEFAULT_MAX_FACES,
    ) -> None:
        if split not in _IMAGE_ASSETS:
            raise ValueError(
                f"WIDER FACE supports TRAIN and VAL splits only, got {split.name}."
            )
        self.max_faces = max_faces
        self._image_asset = _IMAGE_ASSETS[split]
        input_spec = input_spec or {"image": TensorSpec(shape=(1, 3, 640, 640))}
        self.input_height = int(input_spec["image"][0][2])
        self.input_width = int(input_spec["image"][0][3])

        BaseDataset.__init__(
            self,
            self._image_asset.extracted_path.parent,
            split=split,
            input_spec=input_spec,
        )

    @property
    def _images_root(self) -> Path:
        """Directory containing the WIDER FACE ``images/<event>/<img>.jpg`` tree."""
        return self._image_asset.extracted_path / "images"

    def _download_data(self) -> None:
        self._image_asset.fetch(extract=True)
        WIDERFACE_SPLIT_ASSET.fetch(extract=True)

    def _validate_data(self) -> bool:
        gt_file = WIDERFACE_SPLIT_ASSET.extracted_path / _GT_FILENAME[self.split]
        if not (self._images_root.is_dir() and gt_file.exists()):
            return False
        if self.split == DatasetSplit.VAL:
            gt_dir = (
                EXTERNAL_REPO_PATHS["yolov5_face"]
                / "widerface_evaluate"
                / "ground_truth"
            )
            if not gt_dir.exists():
                raise FileNotFoundError(
                    f"WIDER FACE evaluation mat files not found at {gt_dir}. "
                    "Ensure the yolov5_face external repo is set up "
                    "(run setup_external_repos or check EXTERNAL_REPO_PATHS)."
                )
        return True

    @cached_property
    def _samples(self) -> list[tuple[str, list[list[int]]]]:
        """Parse the text GT file into ``(image_relpath, boxes)`` records."""
        gt_file = WIDERFACE_SPLIT_ASSET.extracted_path / _GT_FILENAME[self.split]
        samples: list[tuple[str, list[list[int]]]] = []
        with open(gt_file) as f:
            lines = [line.strip() for line in f]

        i = 0
        n = len(lines)
        while i < n:
            image_relpath = lines[i]
            i += 1
            if not image_relpath:
                continue
            num_boxes = int(lines[i])
            i += 1
            boxes: list[list[int]] = []
            if num_boxes == 0:
                # A zero-face image still has one all-zero filler line.
                i += 1
            else:
                for _ in range(num_boxes):
                    parts = lines[i].split()
                    i += 1
                    # Each line: xmin ymin w h blur expression illumination invalid occlusion pose
                    # Column 7 is the 'invalid' flag — skip faces the benchmark ignores.
                    if int(parts[7]) == 1:
                        continue
                    boxes.append([int(v) for v in parts[:4]])
            samples.append((image_relpath, boxes))
        return samples

    @cached_property
    def _mat_index(
        self,
    ) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Build a lookup from (event_name, image_stem) to mat-derived GT.

        Only valid for ``DatasetSplit.VAL``; raises ``FileNotFoundError`` (via
        ``_validate_data``) if the external repo mat files are absent.

        """
        gt_dir = (
            EXTERNAL_REPO_PATHS["yolov5_face"] / "widerface_evaluate" / "ground_truth"
        )
        face_mat = loadmat(str(gt_dir / "wider_face_val.mat"))
        easy_mat = loadmat(str(gt_dir / "wider_easy_val.mat"))
        medium_mat = loadmat(str(gt_dir / "wider_medium_val.mat"))
        hard_mat = loadmat(str(gt_dir / "wider_hard_val.mat"))

        event_list = face_mat["event_list"]
        file_list = face_mat["file_list"]
        facebox_list = face_mat["face_bbx_list"]
        easy_gt = easy_mat["gt_list"]
        medium_gt = medium_mat["gt_list"]
        hard_gt = hard_mat["gt_list"]

        index: dict[
            tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        for i in range(len(event_list)):
            event_name = str(event_list[i][0][0])
            img_list = file_list[i][0]
            gt_bbx_list = facebox_list[i][0]
            easy_list = easy_gt[i][0]
            medium_list = medium_gt[i][0]
            hard_list = hard_gt[i][0]

            for j in range(len(img_list)):
                stem = str(img_list[j][0][0])
                boxes_xywh = gt_bbx_list[j][0].astype(np.float32)
                num_faces = boxes_xywh.shape[0]

                index[(event_name, stem)] = (
                    boxes_xywh,
                    _make_ignore(num_faces, easy_list[j][0]),
                    _make_ignore(num_faces, medium_list[j][0]),
                    _make_ignore(num_faces, hard_list[j][0]),
                )
        return index

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, index: int
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        image : torch.Tensor
            Shape (3, H, W), float32 [0, 1], RGB.

        ground_truth : tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            scale
                Scalar float32 — letterbox scale factor.
            padding
                Shape (2,) int32 — (pad_left, pad_top).
            gt_boxes_xywh
                Shape (max_faces, 4) float32 — GT boxes in original pixel
                coordinates, format [x, y, w, h].  Rows beyond ``num_faces``
                are zero-padded.
            ignore_easy
                Shape (max_faces,) float32 — 1 = face counts for Easy subset,
                0 = ignore.  Zero-padded beyond ``num_faces``.
            ignore_medium
                Shape (max_faces,) float32 — same for Medium subset.
            ignore_hard
                Shape (max_faces,) float32 — same for Hard subset.
            num_faces
                Scalar int32 — number of valid (non-padded) entries.
        """
        image_relpath, _text_boxes = self._samples[index]

        path_parts = image_relpath.split("/")
        event_name = path_parts[0]
        image_stem = Path(path_parts[1]).stem

        pil_image = Image.open(self._images_root / image_relpath).convert("RGB")

        image_tensor = app_to_net_image_inputs(pil_image)[1]
        net_input, scale, (pad_left, pad_top) = resize_pad(
            image_tensor,
            dst_size=(self.input_height, self.input_width),
            pad_value=PAD_VALUE,
        )
        net_input = net_input.squeeze(0)

        gt_boxes = torch.zeros(self.max_faces, 4, dtype=torch.float32)
        ign_easy = torch.zeros(self.max_faces, dtype=torch.float32)
        ign_med = torch.zeros(self.max_faces, dtype=torch.float32)
        ign_hard = torch.zeros(self.max_faces, dtype=torch.float32)

        if self.split == DatasetSplit.VAL:
            boxes_xywh, ie, im, ih = self._mat_index[(event_name, image_stem)]
            if len(boxes_xywh) > self.max_faces:
                warnings.warn(
                    f"{image_relpath} has {len(boxes_xywh)} faces, exceeding "
                    f"max_faces={self.max_faces}. Excess faces are silently dropped, "
                    "which will undercount `count_face` and inflate AP. "
                    "Increase max_faces if this matters.",
                    RuntimeWarning,
                    stacklevel=1,
                )
            num_faces = min(len(boxes_xywh), self.max_faces)
            gt_boxes[:num_faces] = torch.from_numpy(boxes_xywh[:num_faces])
            ign_easy[:num_faces] = torch.from_numpy(ie[:num_faces])
            ign_med[:num_faces] = torch.from_numpy(im[:num_faces])
            ign_hard[:num_faces] = torch.from_numpy(ih[:num_faces])
        else:
            # TRAIN is calibration-only; difficulty splits are undefined.
            # All-ones ignore means every face counts — do not use with the evaluator.
            num_faces = min(len(_text_boxes), self.max_faces)
            if num_faces > 0:
                arr = torch.tensor(_text_boxes[:num_faces], dtype=torch.float32)
                gt_boxes[:num_faces] = arr
            ign_easy[:num_faces] = 1.0
            ign_med[:num_faces] = 1.0
            ign_hard[:num_faces] = 1.0

        return net_input, (
            torch.tensor(scale, dtype=torch.float32),
            torch.tensor([pad_left, pad_top], dtype=torch.int32),
            gt_boxes,
            ign_easy,
            ign_med,
            ign_hard,
            torch.tensor(num_faces, dtype=torch.int32),
        )

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/CUHK-CSE/wider_face",
            split_description="validation split",
        )
