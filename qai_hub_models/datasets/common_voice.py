# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
import random
from pathlib import Path

import pandas as pd

from qai_hub_models.datasets.common import (
    BaseDataset,
    DatasetSplit,
)
from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.input_spec import InputSpec

CommonVoice_FOLDER_NAME = "common_voice"
CommonVoice_VERSION = 1

CommonVoice_ASSET = CachedWebDatasetAsset.from_asset_store(
    CommonVoice_FOLDER_NAME,
    CommonVoice_VERSION,
    "common_voice.zip",
)


class CommonVoiceText(BaseDataset):
    def __init__(
        self,
        lang: str | Path,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        self.common_voice_text = CachedWebDatasetAsset.from_asset_store(
            CommonVoice_FOLDER_NAME,
            CommonVoice_VERSION,
            f"train_{lang}.tsv",
        )
        super().__init__(self.common_voice_text.path(), split, input_spec)

        df = pd.read_csv(self.dataset_path, sep="\t")
        texts_list = df["sentence"].head(n=self.default_samples_per_job()).tolist()
        examples = []
        random_sample = 4
        random.seed(42)
        for sentence in texts_list:
            num_repeats = random.randint(1, random_sample)
            example = " ".join([sentence] * num_repeats)
            examples.append(example)
        self.texts_list = examples

    def __len__(self) -> int:
        return len(self.texts_list)

    def __getitem__(self, index: int) -> tuple[str, list[None]]:
        """
        Returns a tuple of wav feature and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        wav_file: str
            the path of a single wav file
        ground_truth: list[None]
            Empty list, no ground truth data.
        """
        return self.texts_list[index], []

    def _download_data(self) -> None:
        self.common_voice_text.fetch()

    @classmethod
    def dataset_name(cls) -> str:
        return "common_voice_text"

    @staticmethod
    def default_samples_per_job() -> int:
        return 5


class CommonVoiceDataset(BaseDataset):
    def __init__(
        self,
        input_tar: str | None = None,
        split: DatasetSplit = DatasetSplit.TRAIN,
        input_spec: InputSpec | None = None,
    ) -> None:
        common_voice_path = (
            CommonVoice_ASSET.path(extracted=True) / CommonVoice_FOLDER_NAME
        )
        BaseDataset.__init__(self, common_voice_path, split)

        self.wav_list = []
        for root, _dirs, files in os.walk(self.dataset_path):
            for name in files:
                if name.endswith(".wav"):
                    self.wav_list.append(os.path.join(root, name))

    def __len__(self) -> int:
        return len(self.wav_list)

    def __getitem__(self, index: int) -> tuple[str, list[None]]:
        """
        Returns a tuple of wav feature and label data.

        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        wav_file: str
            the path of a single wav file
        ground_truth: list[None]
            Empty list, no ground truth data.
        """
        return self.wav_list[index], []

    def _download_data(self) -> None:
        CommonVoice_ASSET.fetch(extract=True)

    @classmethod
    def dataset_name(cls) -> str:
        return "common_voice"

    @staticmethod
    def default_samples_per_job() -> int:
        return 200
