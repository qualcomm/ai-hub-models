# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.dataset.lerobot_episode_loader import (
    LeRobotEpisodeLoader,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.dataset.sharded_single_step_dataset import (
    extract_step_data,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.types import (
    EmbodimentTag,
    ModalityConfig,
    VLAStepData,
)
from qai_hub_models.models.grootn17.model import (
    DEFAULT_DATASET_ASSET,
    GrootCollection,
    GrootEvalContext,
)
from qai_hub_models.models.grootn17.utils import concatenate_action_modalities
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec


def _action_offsets(
    modality_configs: dict[str, ModalityConfig], action_horizon: int
) -> list[int]:
    offsets = list(modality_configs["action"].delta_indices)
    if action_horizon <= 0 or len(offsets) < action_horizon:
        raise ValueError(
            f"Action modality provides {len(offsets)} offsets for action horizon {action_horizon}."
        )
    return offsets[:action_horizon]


def step_to_tensors(
    step: VLAStepData, modality_configs: dict[str, ModalityConfig]
) -> tuple[torch.Tensor, ...]:
    """Transport raw video/state/text only; actions are separate evaluator labels."""
    if not isinstance(step.text, str):
        raise TypeError(
            "Evaluation requires a language string; use an empty string for an empty instruction."
        )
    fields = [
        torch.from_numpy(
            np.stack([np.asarray(frame) for frame in step.images[key]]).astype(np.uint8)
        )
        for key in modality_configs["video"].modality_keys
    ]
    fields.extend(
        torch.from_numpy(np.array(step.states[key], dtype=np.float32, copy=True))
        for key in modality_configs["state"].modality_keys
    )
    fields.append(
        torch.from_numpy(
            np.frombuffer(step.text.encode("utf-8"), dtype=np.uint8).copy()
        )
    )
    return tuple(fields)


def step_from_tensors(
    fields: Sequence[torch.Tensor],
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
) -> VLAStepData:
    """Reconstruct one observation from a loader batch, preserving complete UTF-8."""
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    if len(fields) != len(video_keys) + len(state_keys) + 1:
        raise ValueError(
            "Evaluation fields must contain configured videos, states, and one language tensor."
        )
    arrays = []
    for tensor in fields:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim < 1
            or tensor.shape[0] != 1
        ):
            raise ValueError("GR00T N1.7 evaluation fields must have batch size 1.")
        arrays.append(tensor[0].detach().cpu().numpy())
    videos = arrays[: len(video_keys)]
    states = arrays[len(video_keys) : -1]
    language = arrays[-1]
    if any(
        video.ndim != 4 or video.dtype != np.uint8 or video.shape[-1] != 3
        for video in videos
    ):
        raise ValueError(
            "Evaluation videos must have unbatched shape [T, H, W, 3] and dtype uint8."
        )
    if any(state.ndim != 2 or state.dtype != np.float32 for state in states):
        raise ValueError(
            "Evaluation states must have unbatched shape [T, D] and dtype float32."
        )
    if language.ndim != 1 or language.dtype != np.uint8:
        raise ValueError(
            "Evaluation language must be a one-dimensional uint8 UTF-8 array."
        )
    return VLAStepData(
        images={
            key: list(video) for key, video in zip(video_keys, videos, strict=True)
        },
        states=dict(zip(state_keys, states, strict=True)),
        actions={},
        text=language.tobytes().decode("utf-8"),
        embodiment=embodiment_tag,
    )


class GrootDataset(BaseDataset):
    """Complete DROID cadence windows, with one observation per inference job."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        eval_context: GrootEvalContext | None = None,
    ) -> None:
        if split != DatasetSplit.VAL:
            raise ValueError("GR00T N1.7 supports the validation split only.")
        self.eval_context = (
            eval_context
            if eval_context is not None
            else GrootCollection.get_eval_context()
        )
        self.config = self.eval_context.app_config
        # Validation raises on invalid local data, so BaseDataset never deletes it.
        self.dataset_path = Path(
            dataset_path
            if dataset_path is not None
            else DEFAULT_DATASET_ASSET.fetch(extract=True)
        )
        self._validate_data()
        super().__init__(self.dataset_path, split, input_spec)
        self._episode_loader = LeRobotEpisodeLoader(
            dataset_path=str(self.dataset_path),
            modality_configs=self.config.modality_configs,
            video_backend="ffmpeg",
        )
        offsets = _action_offsets(
            self.config.modality_configs, self.config.action_horizon
        )
        self._cadence_points = [
            (episode_index, step_index)
            for episode_index, metadata in enumerate(
                self._episode_loader.episodes_metadata
            )
            for step_index in range(0, metadata["length"], self.config.action_horizon)
            if all(0 <= step_index + offset < metadata["length"] for offset in offsets)
        ]
        if not self._cadence_points:
            raise ValueError(
                f"No complete {self.config.action_horizon}-step evaluation windows in {self.dataset_path}."
            )
        self._cached_episode_index: int | None = None
        self._cached_episode: pd.DataFrame | None = None

    @property
    def sample_ids(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._cadence_points)

    def __len__(self) -> int:
        return len(self._cadence_points)

    def __getitem__(self, index: int) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        episode_index, step_index = self._cadence_points[index]
        if self._cached_episode_index != episode_index:
            episode = self._episode_loader[episode_index]
            expected_length = self._episode_loader.episodes_metadata[episode_index][
                "length"
            ]
            if len(episode) != expected_length:
                raise ValueError(
                    f"Episode {episode_index} metadata says {expected_length} steps, but loaded {len(episode)}."
                )
            self._cached_episode = episode
            self._cached_episode_index = episode_index
        assert self._cached_episode is not None
        step = extract_step_data(
            self._cached_episode,
            step_index=step_index,
            modality_configs=self.config.modality_configs,
            embodiment_tag=self.config.embodiment_tag,
            allow_padding=False,
        )
        gt_chunk = concatenate_action_modalities(
            step.actions, self.config.action_layout, has_batch_axis=False
        )[0]
        return step_to_tensors(step, self.config.modality_configs), torch.from_numpy(
            gt_chunk
        )

    def _validate_data(self) -> bool:
        required = (
            "info.json",
            "episodes.jsonl",
            "modality.json",
            "stats.json",
            "tasks.jsonl",
        )
        if not self.dataset_path.is_dir() or any(
            not (self.dataset_path / "meta" / name).is_file() for name in required
        ):
            raise ValueError(
                f"Expected a local LeRobot DROID dataset with meta/{', '.join(required)} at {self.dataset_path}. No data was modified."
            )
        return True

    def _download_data(self) -> None:
        DEFAULT_DATASET_ASSET.fetch(extract=True)

    @staticmethod
    def collate_fn(batch: list[Any]) -> Any:
        if len(batch) != 1:
            raise ValueError("GR00T N1.7 evaluation requires samples_per_job=1.")
        return BaseDataset.collate_fn(batch)

    @staticmethod
    def default_samples_per_job() -> int:
        return 1

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://droid-dataset.github.io/",
            split_description="Complete DROID action windows at action-horizon cadence",
        )
