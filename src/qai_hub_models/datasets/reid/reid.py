# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
from PIL import Image

from qai_hub_models.models.templates.corevision.image_processing import (
    preprocess_PIL_image,
)
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

# ── Shared sample type ────────────────────────────────────────────────────────


class ReidSample(NamedTuple):
    """A single image entry in a ReID dataset."""

    image_path: Path
    person_id: int
    camera_id: int
    is_query: bool


# ── Shared sampling helper ────────────────────────────────────────────────────


def limit_reid_eval_samples(
    samples: list[ReidSample], max_eval_samples: int
) -> list[ReidSample]:
    """Return at most *max_eval_samples* items from *samples*, preserving a
    valid query/gallery split for ReID evaluation.

    The budget is split in a 1:3 query-to-gallery ratio.  Gallery images are
    sampled round-robin across person IDs so that every identity is represented
    even in a truncated evaluation.  Taking the first N images sorted by
    filename would concentrate the gallery in the lowest-numbered PIDs (because
    filenames sort by PID first), leaving only a small fraction of identities
    in the gallery and collapsing the number of matchable queries.

    Only query images whose person ID appears in the selected gallery are
    admitted, so every query has at least one valid match.
    """
    if len(samples) <= max_eval_samples:
        return samples

    queries = [s for s in samples if s.is_query]
    galleries = [s for s in samples if not s.is_query]
    if not queries or not galleries:
        return samples[:max_eval_samples]

    target_queries = max(1, max_eval_samples // 4)
    target_galleries = max_eval_samples - target_queries

    gallery_by_pid: dict[int, list[ReidSample]] = defaultdict(list)
    for s in galleries:
        gallery_by_pid[s.person_id].append(s)
    pid_order = sorted(gallery_by_pid)
    selected_galleries: list[ReidSample] = []
    round_idx = 0
    while len(selected_galleries) < target_galleries:
        added_any = False
        for pid in pid_order:
            if round_idx < len(gallery_by_pid[pid]):
                selected_galleries.append(gallery_by_pid[pid][round_idx])
                added_any = True
                if len(selected_galleries) >= target_galleries:
                    break
        if not added_any:
            break
        round_idx += 1

    gallery_pids = {s.person_id for s in selected_galleries}
    matchable_queries = [s for s in queries if s.person_id in gallery_pids]
    selected_queries = matchable_queries[: min(len(matchable_queries), target_queries)]

    selected = selected_queries + selected_galleries
    selected.sort(key=lambda s: (not s.is_query, s.image_path.name))
    return selected


# ── Base class ────────────────────────────────────────────────────────────────


class BaseReidDataset(BaseDataset):
    """Abstract base for person re-identification datasets.

    Subclasses must supply:
      - ``FILENAME_PATTERN`` class attribute (compiled regex with ``pid`` and
        ``camid`` named groups)
      - ``get_private_asset`` static method returning a
        :class:`CachedPrivateDatasetAsset`
      - ``dataset_name``, ``default_samples_per_job``, ``get_dataset_metadata``
      - ``_get_data_root`` — returns the directory containing
        ``bounding_box_test/`` and ``query/``
      - ``_build_samples`` — parses images into a flat list of
        :class:`ReidSample`

    Everything else (``__init__``, ``__len__``, ``__getitem__``, ``configure``,
    ``_download_data``, ``_validate_data``) is
    implemented here and shared across all subclasses.
    """

    @staticmethod
    @abstractmethod
    def get_private_asset() -> CachedPrivateDatasetAsset:
        """Return the :class:`CachedPrivateDatasetAsset` for this dataset."""

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        max_eval_samples: int = 1000,
        input_data_path: str | None = None,
    ) -> None:
        self.max_eval_samples = max_eval_samples
        self.input_height = 256
        self.input_width = 128
        if input_spec and "image" in input_spec:
            self.input_height = input_spec["image"][0][2]
            self.input_width = input_spec["image"][0][3]
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(
                f"{self.__class__.__name__} input spatial dimensions must be positive."
            )

        self.samples: list[ReidSample] = []
        self.input_data_path = input_data_path
        BaseDataset.__init__(
            self, self.get_private_asset().extracted_path, split, input_spec
        )
        self.samples = self._build_samples(split)

        if split in (DatasetSplit.VAL, DatasetSplit.TEST) and max_eval_samples > 0:
            if max_eval_samples < 4:
                raise ValueError(
                    f"max_eval_samples={max_eval_samples} is too small for the 1:3 "
                    "query-to-gallery ratio; set it to at least 4 (or 0 to use all samples)."
                )
            self.samples = limit_reid_eval_samples(self.samples, max_eval_samples)
            if not any(s.is_query for s in self.samples) or not any(
                not s.is_query for s in self.samples
            ):
                raise ValueError(
                    f"{self.__class__.__name__} has no valid query/gallery split "
                    f"after limiting to {max_eval_samples} samples."
                )

        if len(self.samples) == 0:
            raise ValueError(f"{self.__class__.__name__} yielded zero valid samples.")
        if split in (DatasetSplit.VAL, DatasetSplit.TEST) and (
            not any(s.is_query for s in self.samples)
            or not any(not s.is_query for s in self.samples)
        ):
            raise ValueError(
                f"{self.__class__.__name__} eval split requires both query "
                "and gallery samples."
            )

    # ── abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _build_samples(self, split: DatasetSplit) -> list[ReidSample]:
        """Parse the on-disk layout into a flat list of :class:`ReidSample`."""

    @abstractmethod
    def _get_data_root(self) -> Path:
        """Return the directory that contains ``bounding_box_test/`` and ``query/``."""

    # ── shared implementations ────────────────────────────────────────────────

    def _download_data(self) -> None:
        if self.input_data_path is None:
            self.get_private_asset().fetch(extract=True)
        else:
            self.get_private_asset().fetch(
                extract=True,
                local_path=Path(self.input_data_path).expanduser().resolve(),
            )

    def _validate_data(self) -> bool:
        data_root = self._get_data_root()
        gallery_root = data_root / "bounding_box_test"
        query_root = data_root / "query"
        return (
            gallery_root.exists()
            and query_root.exists()
            and self._contains_images(gallery_root)
            and self._contains_images(query_root)
        )

    @staticmethod
    def _contains_images(root: Path) -> bool:
        """Return True if *root* contains at least one jpg or png image."""
        return any(root.rglob("*.jpg")) or any(root.rglob("*.png"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Return (image_tensor, (person_id, camera_id, is_query)) for the given index."""
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = torch.nn.functional.interpolate(
            preprocess_PIL_image(image),
            size=(self.input_height, self.input_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return image_tensor, (
            sample.person_id,
            sample.camera_id,
            1 if sample.is_query else 0,
        )

    @classmethod
    def configure(cls, files: list[str | os.PathLike]) -> None:
        """Configure the dataset from a local zip archive.

        Parameters
        ----------
        files:
            A single-element list containing the path to the dataset zip file.
        """
        if len(files) != 1:
            raise ValueError(
                f"configure() expects exactly one file: the path to the dataset zip. "
                f"Got {len(files)} file(s)."
            )
        cls(input_data_path=str(files[0]))
