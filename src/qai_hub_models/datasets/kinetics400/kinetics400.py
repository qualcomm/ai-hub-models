# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from qai_hub_models.datasets.kinetics400.video_utils import (
    DEFAULT_NUM_CLIPS,
    DEFAULT_NUM_CROPS,
    VIDEOMAE_FRAME_SAMPLE_RATE,
    VIDEOMAE_NUM_CLIPS,
    VIDEOMAE_NUM_CROPS,
    multi_crop,
    preprocess_video_224,
    preprocess_video_kinetics_400,
    read_video_at_fps,
    read_video_per_second,
    sample_clips,
    sample_video,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.labels import get_class_names

KINETICS400_FOLDER_NAME = "kinetics400"
KINETICS400_VERSION = 2

# Some of the video files in the training data downloaded from the Internet
# have corrupted video files that can't be opened. If we try to load these samples
# at runtime, it throws an error and kills the process. We decided it best to remove
# these files right after downloading to avoid this error.
CORRUPTED_TRAIN_VIDEOS = [
    "-I4Ggi6-QOE_000054_000064.mp4",
    "-K_uevUt2V8_000003_000013.mp4",
    "-6wkYqjFei0_000015_000025.mp4",
    "-IT7W5_Y3Gc_000003_000013.mp4",
    "-4c4r9YeS6s_000098_000108.mp4",
    "--PyMoD3_eg_000020_000030.mp4",
    "-3pp5xan1Hw_000006_000016.mp4",
]


def _get_labeled_data(
    videos_folder: Path, labels_csv_path: Path
) -> tuple[list[str], list[int]]:
    """
    Given the folder with a subset of videos and the appropriate labels file,
    returns a list of filenames and a list of label indices for the subset that match.
    """
    video_metadata_rows = []
    for filename in os.listdir(videos_folder):
        filename_split = filename[: -len(".mp4")].split("_")
        youtube_id = "_".join(filename_split[:-2])
        start = int(filename_split[-2])
        end = int(filename_split[-1])
        video_metadata_rows.append((youtube_id, start, end))
    video_metadata_df = pd.DataFrame(
        video_metadata_rows, columns=["youtube_id", "time_start", "time_end"]
    )
    labels_df = pd.read_csv(labels_csv_path)
    join_df = labels_df.merge(
        video_metadata_df, on=["youtube_id", "time_start", "time_end"], how="inner"
    )

    # Sort to ensure deterministic ordering regardless of filesystem
    join_df = join_df.sort_values(by=["youtube_id", "time_start"])
    video_paths: list[str] = []
    label_indices: list[int] = []
    label_index_map = {
        label: i for (i, label) in enumerate(get_class_names("kinetics400"))
    }
    for _, row in join_df.iterrows():
        assert isinstance(row, pd.Series)
        video_paths.append(
            f"{row.youtube_id}_{row.time_start:06d}_{row.time_end:06d}.mp4"
        )
        label_indices.append(label_index_map[row.label])
    return video_paths, label_indices


class PreprocessProtocol(Enum):
    """
    Per-clip decode + resize + crop recipe, inferred once at ``__init__`` from
    the model's input spatial size.

    KINETICS_112_TORCHVISION
        torchvision R3D / R2+1D / MC3: read at 15 fps, resize to (128, 171),
        center-crop to 112x112.
    KINETICS_224_VIDEOMAE
        VideoMAE: read at native fps, short-side resize to 256; center-crop to
        224x224 only when ``multi_crop`` won't run after.
    """

    KINETICS_112_TORCHVISION = "kinetics_112_torchvision"
    KINETICS_224_VIDEOMAE = "kinetics_224_videomae"


class Kinetics400Dataset(BaseDataset):
    """
    Class for using the Kinetics400 dataset for video classification:
        https://github.com/cvdfoundation/kinetics-dataset
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_frames: int = 16,
        input_spec: InputSpec | None = None,
        num_clips: int = DEFAULT_NUM_CLIPS,
        num_crops: int = DEFAULT_NUM_CROPS,
    ) -> None:
        self.num_frames = num_frames
        self.split_str = split.name.lower()
        self.videos_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/{self.split_str}/part_0.tar.gz",
            KINETICS400_FOLDER_NAME,
            KINETICS400_VERSION,
            os.path.join(self.split_str, "part_0.tar.gz"),
        )
        self.csv_asset = CachedWebDatasetAsset(
            f"https://s3.amazonaws.com/kinetics/400/annotations/{self.split_str}.csv",
            KINETICS400_FOLDER_NAME,
            KINETICS400_VERSION,
            os.path.join("annotations", f"{self.split_str}.csv"),
        )
        self.videos_folder = self.videos_asset.extracted_path
        self.video_dim = input_spec["video"][0][-1] if input_spec else 112
        assert self.video_dim in [112, 224], "Video dimension must be 112 or 224."
        # torchvision uses the __init__ defaults (5 clips x 1 crop, stride 1);
        # VideoMAE uses 5 clips x 1 crop, stride 2 (reduced from its published
        # 10x3 for the 2 GB job limit — see video_utils.py).
        if self.video_dim == 112:
            self.protocol = PreprocessProtocol.KINETICS_112_TORCHVISION
            self.frame_sample_rate = 1
            self.num_clips, self.num_crops = num_clips, num_crops
        else:
            self.protocol = PreprocessProtocol.KINETICS_224_VIDEOMAE
            self.frame_sample_rate = VIDEOMAE_FRAME_SAMPLE_RATE
            self.num_clips, self.num_crops = VIDEOMAE_NUM_CLIPS, VIDEOMAE_NUM_CROPS
        self.num_views = self.num_clips * self.num_crops
        # Cache the decoded views of the most-recently accessed video so the
        # num_views consecutive __getitem__ calls for one video decode it once.
        self._cache_video_idx: int | None = None
        self._cache_views: list[torch.Tensor] = []
        BaseDataset.__init__(
            self, str(self.videos_folder), split=split, input_spec=input_spec
        )

    @property
    def num_videos(self) -> int:
        # Number of non-corrupted videos in each split
        return 993 if self.split == DatasetSplit.TRAIN else 1000

    def __len__(self) -> int:
        return self.num_videos * self.num_views

    def _validate_data(self) -> bool:
        if not self.csv_asset.path.exists():
            return False

        videos_folder = self.videos_asset.path.parent
        if not videos_folder.exists():
            return False

        self.mp4_files, self.label_indices = _get_labeled_data(
            self.videos_folder, self.csv_asset.path
        )

        if len(self.mp4_files) != self.num_videos:
            return False

        return len(self.label_indices) == self.num_videos

    def preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the per-clip preprocessing for ``self.protocol``."""
        if self.protocol is PreprocessProtocol.KINETICS_112_TORCHVISION:
            return preprocess_video_kinetics_400(tensor)
        # Skip center-crop when multi_crop handles the spatial views instead.
        return preprocess_video_224(tensor, center_crop=self.num_crops == 1)

    def _build_views(self, video_idx: int) -> list[torch.Tensor]:
        """
        Decode video ``video_idx`` and build all ``num_clips * num_crops``
        preprocessed views, each of shape ``[3, T, video_dim, video_dim]``.
        """
        video_path = str(self.videos_folder / self.mp4_files[video_idx])
        if self.protocol is PreprocessProtocol.KINETICS_112_TORCHVISION:
            raw_video = read_video_at_fps(video_path, target_fps=15)
        else:
            raw_video = read_video_per_second(video_path)

        # VideoMAE strides the video before windowing; torchvision uses stride 1.
        if self.frame_sample_rate > 1:
            raw_video = raw_video[:: self.frame_sample_rate]

        if self.num_clips > 1:
            clips = sample_clips(raw_video, self.num_frames, self.num_clips)
        else:
            clips = [sample_video(raw_video, self.num_frames)]

        views: list[torch.Tensor] = []
        for clip in clips:
            preprocessed = self.preprocess_tensor(clip)
            if self.num_crops > 1:
                views.extend(multi_crop(preprocessed, self.video_dim, self.num_crops))
            else:
                views.append(preprocessed)
        return views

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Return a single view of a video.

        Parameters
        ----------
        index
            Flattened index in ``[0, num_videos * num_views)``. Views of the
            same video are contiguous: ``video_idx = index // num_views``.

        Returns
        -------
        view : torch.Tensor
            One preprocessed clip of shape ``[3, T, video_dim, video_dim]``.
        gt : tuple[torch.Tensor, torch.Tensor]
            ``(label, video_id)`` as scalar ``int64`` tensors. ``video_id``
            lets the evaluator aggregate scores across a video's views.
        """
        video_idx = index // self.num_views
        view_idx = index % self.num_views

        if self._cache_video_idx != video_idx:
            self._cache_views = self._build_views(video_idx)
            self._cache_video_idx = video_idx

        view = self._cache_views[view_idx]
        label = torch.tensor(self.label_indices[video_idx], dtype=torch.int64)
        video_id = torch.tensor(video_idx, dtype=torch.int64)
        return view, (label, video_id)

    def get_dataloader(
        self, num_samples: int, samples_per_job: int | None = None
    ) -> DataLoader:
        """Return a DataLoader over whole videos (all views kept together).

        ``num_samples`` and ``samples_per_job`` count **views**. Values that
        are not already a multiple of ``num_views`` are rounded up so a
        video's views are never split across jobs. ``num_samples=-1``
        (full eval) maps to all videos.
        """

        def _round_up(n: int) -> int:
            return ((n + self.num_views - 1) // self.num_views) * self.num_views

        views_per_job = _round_up(samples_per_job or self.default_samples_per_job())
        if num_samples != -1:
            num_samples = _round_up(num_samples)
        num_videos_wanted = min(max(1, num_samples // self.num_views), self.num_videos)
        stride = self.num_videos // num_videos_wanted
        selected: list[int] = []
        for videos_selected, video_idx in enumerate(range(0, self.num_videos, stride)):
            base = video_idx * self.num_views
            selected.extend(range(base, base + self.num_views))
            if videos_selected + 1 >= num_videos_wanted:
                break
        return DataLoader(
            Subset(self, selected),
            batch_size=min(len(selected), views_per_job),
            collate_fn=self.collate_fn,
        )

    def _download_data(self) -> None:
        self.videos_asset.fetch(extract=True)
        self.csv_asset.fetch()

        if self.split == DatasetSplit.TRAIN:
            for video in CORRUPTED_TRAIN_VIDEOS:
                os.remove(self.videos_folder / video)

    @staticmethod
    def default_samples_per_job() -> int:
        """The default value for how many samples to run in each inference job."""
        return 200

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://github.com/cvdfoundation/kinetics-dataset",
            split_description="part 0 of the validation split",
        )
