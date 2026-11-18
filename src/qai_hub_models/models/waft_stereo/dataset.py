# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import default_collate
from torchvision.datasets.coco import CocoDetection

from qai_hub_models.datasets.coco.coco import CocoDatasetBase
from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
    resize_pad,
)
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec, TensorSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

# The encoder normalizes after padding, so padding with these values (rather than black)
# to keep letterbox borders close to 0 post-normalization, matching upstream's Padder
_IMAGENET_MEAN = (0.485, 0.456, 0.406)


def resize_pad_rgb(
    image: torch.Tensor, dst_size: tuple[int, int]
) -> tuple[torch.Tensor, float, tuple[int, int]]:
    channels = []
    for c, mean in enumerate(_IMAGENET_MEAN):
        padded, scale, padding = resize_pad(
            image[:, c : c + 1], dst_size, pad_value=mean
        )
        channels.append(padded)
    return torch.cat(channels, dim=1), scale, padding


KITTI2015_FOLDER_NAME = "kitti_2015_scene_flow"
KITTI2015_VERSION = 1
_NUM_VAL_PAIRS = 40
_NUM_TOTAL_PAIRS = 200

KITTI2015_INSTALLATION_STEPS = [
    "Download the scene flow stereo benchmark from https://www.cvlibs.net/download.php?file=data_scene_flow.zip",
]
KITTI2015_CONFIGURE_FILES = ["data_scene_flow.zip"]

# KITTI is CC BY-NC-SA 3.0 (non-commercial), so it is used for accuracy eval
# only, never calibration - sourced from Qualcomm's private bucket rather than redistributed alongside public release
KITTI2015_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    f"qai-hub-models/datasets/{KITTI2015_FOLDER_NAME}/v{KITTI2015_VERSION}/data_scene_flow.zip",
    f"{KITTI2015_FOLDER_NAME}_private",
    KITTI2015_VERSION,
    "data_scene_flow.zip",
    installation_steps=KITTI2015_INSTALLATION_STEPS,
    configure_files=KITTI2015_CONFIGURE_FILES,
)


class Kitti2015Dataset(BaseDataset):
    """
    KITTI 2015 Scene Flow stereo benchmark (eval only):
    https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

    KITTI is CC BY-NC-SA 3.0 (non-commercial), so it may only be used to
    measure accuracy, never for quantization calibration - calibration uses
    CocoStereoCalibrationDataset (CC BY 4.0) instead.

    Only the 200 "training" pairs publish ground truth disparity
    (`disp_occ_0`, KITTI's "all" setting - occluded pixels included); the
    test split's GT is withheld by the benchmark server, so only VAL is
    supported. The last `_NUM_VAL_PAIRS` of those 200 pairs are used for eval.
    """

    pair_ids: list[str]

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        input_data_zip: str | None = None,
    ) -> None:
        if split != DatasetSplit.VAL:
            raise ValueError(
                "KITTI 2015 is non-commercially licensed and is used for eval "
                "only; only VAL is supported."
            )
        self.input_data_zip = input_data_zip
        self.kitti_path = KITTI2015_PRIVATE_ASSET.extracted_path
        BaseDataset.__init__(self, self.kitti_path, split, input_spec)
        self.pair_ids = self.pair_ids[-_NUM_VAL_PAIRS:]
        self.input_height: int | None
        self.input_width: int | None
        if input_spec is not None:
            self.input_height = input_spec["img1"][0][2]
            self.input_width = input_spec["img1"][0][3]
        else:
            self.input_height = None
            self.input_width = None

    def _validate_data(self) -> bool:
        image_dir = self.kitti_path / "training" / "image_2"
        if not image_dir.exists():
            return False
        self.pair_ids = sorted(
            p.stem
            for p in image_dir.glob("*_10.png")
            if (self.kitti_path / "training" / "image_3" / p.name).exists()
            and (self.kitti_path / "training" / "disp_occ_0" / p.name).exists()
        )
        # A partial extraction can leave fewer pairs, silently shifting which
        # pairs land in the last `_NUM_VAL_PAIRS` used for eval.
        return len(self.pair_ids) == _NUM_TOTAL_PAIRS

    def _download_data(self) -> None:
        KITTI2015_PRIVATE_ASSET.fetch(extract=True, local_path=self.input_data_zip)

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        if len(files) != 1:
            raise ValueError(
                f"{cls.__name__}.configure expects 1 file(s), got {len(files)}."
            )
        cls(split=DatasetSplit.VAL, input_data_zip=str(files[0]))

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, float, int, int],
    ]:
        """
        Get a stereo image pair and its ground-truth disparity map.

        If `input_spec` was provided at construction, both images are
        resized (preserving aspect ratio) and letterbox-padded (ImageNet
        mean) to (input_height, input_width). The ground-truth disparity
        map is always returned at native resolution, since KITTI's D1-all
        threshold is calibrated for native resolution - along with the
        `resize_pad` scale/padding applied to the images, so a prediction
        made at network resolution can be restored to native resolution
        (via `undo_resize_pad`) before being compared against it.

        Parameters
        ----------
        index
            Index of the pair to retrieve.

        Returns
        -------
        img_pair : tuple[torch.Tensor, torch.Tensor]
            img1_tensor
                Left image, shape [3, H, W], float32, RGB range [0, 1].
            img2_tensor
                Right image, shape [3, H, W], float32, RGB range [0, 1].
        ground_truth : tuple[torch.Tensor, float, int, int]
            disp_tensor
                Ground-truth disparity map at native resolution, shape
                [H, W], float32, in pixels. Invalid pixels are 0, which the
                evaluator's valid mask excludes via `gt <= 0`.
            scale, pad_left, pad_top
                The `resize_pad` params applied to img1/img2 (identity
                values 1.0/0/0 when `input_spec` is None).
        """
        pair_id = self.pair_ids[index]
        training_dir = self.kitti_path / "training"
        img1 = np.asarray(
            Image.open(training_dir / "image_2" / f"{pair_id}.png").convert("RGB"),
            dtype=np.float32,
        )
        img2 = np.asarray(
            Image.open(training_dir / "image_3" / f"{pair_id}.png").convert("RGB"),
            dtype=np.float32,
        )
        # KITTI disparity is a 16-bit PNG scaled by 256; 0 denotes an invalid pixel.
        disp_gt = (
            np.asarray(
                Image.open(training_dir / "disp_occ_0" / f"{pair_id}.png")
            ).astype(np.float32)
            / 256.0
        )
        disp_tensor = torch.from_numpy(disp_gt).contiguous().float()

        if self.input_height is not None and self.input_width is not None:
            img1_t, scale, (pad_left, pad_top) = resize_pad_rgb(
                torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0) / 255.0,
                (self.input_height, self.input_width),
            )
            img2_t, _, _ = resize_pad_rgb(
                torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0) / 255.0,
                (self.input_height, self.input_width),
            )
            img1_tensor = img1_t.squeeze(0).contiguous()
            img2_tensor = img2_t.squeeze(0).contiguous()
        else:
            img1_tensor = (
                torch.from_numpy(img1.transpose(2, 0, 1)).contiguous().float() / 255.0
            )
            img2_tensor = (
                torch.from_numpy(img2.transpose(2, 0, 1)).contiguous().float() / 255.0
            )
            scale, pad_left, pad_top = 1.0, 0, 0
        return (img1_tensor, img2_tensor), (disp_tensor, scale, pad_left, pad_top)

    @classmethod
    def dataset_name(cls) -> str:
        return "kitti_2015_scene_flow"

    @staticmethod
    def collate_fn(
        batch: list[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, float, int, int],
            ]
        ],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # disp_native has variable native resolution per pair, so it's kept
        # as a list instead of stacked; WaftStereoD1Evaluator.add_batch already
        # indexes it per-sample. img1/img2 are fixed-size and collate normally.
        img_pairs, gts = zip(*batch, strict=False)
        disp_native, scale, pad_left, pad_top = zip(*gts, strict=False)
        return default_collate(list(img_pairs)), (
            list(disp_native),
            default_collate(list(scale)),
            default_collate(list(pad_left)),
            default_collate(list(pad_top)),
        )

    @staticmethod
    def default_samples_per_job() -> int:
        return _NUM_VAL_PAIRS

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo",
            split_description=(
                "scene flow 2015 training pairs (only split with public GT); "
                "last 40 used for eval"
            ),
        )


class CocoStereoCalibrationDataset(CocoDatasetBase):
    """COCO (CC BY 4.0) images repurposed as WAFT-Stereo calibration pairs.

    KITTI's non-commercial license rules it out for calibration (see
    Kitti2015Dataset), but calibration needs no ground truth and no true
    stereo correspondence - WaftStereoEncoder is a Siamese network that runs
    img1/img2 independently, so two unrelated COCO images stand in fine for a
    stereo pair. The classification head sees these uncorrelated feature maps
    too (unlike real stereo pairs), but this is validated end to end via the
    KITTI D1-all numbers rather than checked in isolation.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        if split != DatasetSplit.TRAIN:
            raise ValueError("CocoStereoCalibrationDataset only supports TRAIN.")
        input_spec = input_spec or {"img1": TensorSpec(shape=(1, 3, 480, 640))}
        height, width = input_spec["img1"][0][2], input_spec["img1"][0][3]
        super().__init__(
            split=split,
            input_spec={"image": TensorSpec(shape=(1, 3, height, width))},
        )

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[()]]:
        """
        Get a calibration pair of two independently-sampled COCO images.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        img_pair : tuple[torch.Tensor, torch.Tensor]
            img1_tensor, img2_tensor
                Each shape [3, H, W], float32, RGB range [0, 1], resized
                (preserving aspect ratio) and letterbox-padded (ImageNet
                mean) to (target_h, target_w).
        ground_truth : tuple[()]
            Empty - these samples are calibration-only, not used for
            accuracy eval.
        """
        image1, _ = CocoDetection.__getitem__(self, index)
        image2, _ = CocoDetection.__getitem__(self, (index + 1) % len(self))
        img1_tensor = resize_pad_rgb(
            preprocess_PIL_image(image1.convert("RGB")), (self.target_h, self.target_w)
        )[0].squeeze(0)
        img2_tensor = resize_pad_rgb(
            preprocess_PIL_image(image2.convert("RGB")), (self.target_h, self.target_w)
        )[0].squeeze(0)
        return (img1_tensor, img2_tensor), ()

    @staticmethod
    def default_samples_per_job() -> int:
        return 300

    @staticmethod
    def default_num_calibration_samples() -> int:
        return 16

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://cocodataset.org/",
            split_description="train2017 split, repurposed as calibration pairs (no ground truth needed)",
        )
