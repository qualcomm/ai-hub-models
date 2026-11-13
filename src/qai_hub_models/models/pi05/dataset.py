# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi05 import make_pi05_pre_post_processors

from qai_hub_models.models.templates.corevision.image_processing import (
    resize_and_normalize,
)
from qai_hub_models.utils.base_dataset import (
    BaseDataset,
    DatasetMetadata,
    DatasetSplit,
)
from qai_hub_models.utils.input_spec import InputSpec

DATASET_REPO_ID = "HuggingFaceVLA/libero"


class LiberoDataset(BaseDataset):
    """
    LIBERO dataset (via LeRobot) preprocessed for Pi05 calibration.

    Each sample is the official PI05 preprocessor output reduced to the
    model-ready tensors Pi05 components consume: resized/normalized camera
    images plus tokenized language. Downstream component inputs (backbone
    hidden states, action-expert KV caches) are built on the fly by
    Pi05App.get_calibration_data by running the upstream components.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        # Function-scoped to break the import cycle: model.py imports
        # LiberoDataset at module level, so this module defers its model import
        # to here (runs once per instantiation, not a hot path).
        from qai_hub_models.models.pi05.model import (
            DEFAULT_CHECKPOINT,
            load_checkpoint,
        )

        # Instantiating LeRobotDataset downloads/loads the data from the HF
        # cache; use its root as the dataset path so download_data() is a
        # no-op validated by the path existing.
        self._lerobot = LeRobotDataset(DATASET_REPO_ID)
        BaseDataset.__init__(self, self._lerobot.root, split, input_spec)

        # PI05 config/preprocessor used to normalize and tokenize the raw
        # batch. load_checkpoint is lru_cached, so the policy is shared with
        # Pi05App.get_calibration_data.
        cfg = load_checkpoint(DEFAULT_CHECKPOINT).config
        self.preprocessor, _ = make_pi05_pre_post_processors(
            config=cfg,
            dataset_stats=self._lerobot.meta.stats,
        )
        self.image_keys = sorted(
            k
            for k, v in cfg.input_features.items()
            if v.type == FeatureType.VISUAL and "empty" not in k
        )

    def _download_data(self) -> None:
        # Data is fetched by LeRobotDataset during __init__.
        pass

    def _validate_data(self) -> bool:
        # LeRobotDataset (loaded in __init__) is the source of truth; a non-empty
        # frame count means the HF cache is present and usable.
        return self._lerobot.num_frames > 0

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], int]:
        raw_sample = self._lerobot[index]
        raw_batch: dict = {}
        for k, v in raw_sample.items():
            if isinstance(v, torch.Tensor):
                raw_batch[k] = v.unsqueeze(0)
            elif isinstance(v, str):
                raw_batch[k] = [v]
            else:
                raw_batch[k] = v

        batch = self.preprocessor(raw_batch)

        sample: dict[str, torch.Tensor] = {}
        for key in self.image_keys:
            img = batch[key]
            if img.ndim != 4:
                continue
            sample[key] = resize_and_normalize(img).squeeze(0)
        sample["observation.language.tokens"] = batch[
            "observation.language.tokens"
        ].squeeze(0)
        sample["observation.language.attention_mask"] = (
            batch["observation.language.attention_mask"].squeeze(0).to(torch.float32)
        )
        # Label is unused for calibration; return 0 (an int collates cleanly and
        # keeps the (dict, int) sample type simple), matching other datasets.
        return sample, 0

    def __len__(self) -> int:
        return self._lerobot.num_frames

    @staticmethod
    def default_samples_per_job() -> int:
        return 100

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://huggingface.co/datasets/HuggingFaceVLA/libero",
            split_description="LIBERO training split (via LeRobot)",
        )
