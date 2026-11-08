# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn import functional as F

from qai_hub_models.utils.asset_loaders import CachedWebDatasetAsset
from qai_hub_models.utils.base_dataset import BaseDataset, DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec

LIBRISPEECH_FOLDER_NAME = "librispeech"
LIBRISPEECH_VERSION = 2
# LibriSpeech test-clean dataset from: www.openslr.org/12 (test-clean.tar.gz)
LIBRISPEECH_CLEAN_ASSET = CachedWebDatasetAsset.from_asset_store(
    LIBRISPEECH_FOLDER_NAME,
    LIBRISPEECH_VERSION,
    "test-clean.tar.gz",
)
# LibriSpeech train-clean-100 used exclusively for quantization calibration.
LIBRISPEECH_TRAIN_ASSET = CachedWebDatasetAsset.from_asset_store(
    LIBRISPEECH_FOLDER_NAME,
    LIBRISPEECH_VERSION,
    "train-clean-100.tar.gz",
)

TARGET_SAMPLE_RATE = 16000
# 35 seconds at 16kHz — covers max LibriSpeech test-clean duration
DEFAULT_SEQUENCE_LENGTH = 560000
DEFAULT_MAX_TEXT_LENGTH = (
    600  # covers longest transcription in LibriSpeech test-clean (576 chars)
)


class LibriSpeechDataset(BaseDataset):
    """LibriSpeech dataset for Parakeet TDT calibration and ASR evaluation.

    Returns raw float32 audio tensor and ASCII-encoded ground truth tensor.
    """

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        max_sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        max_text_length: int = DEFAULT_MAX_TEXT_LENGTH,
        input_spec: InputSpec | None = None,
    ) -> None:
        # Calibration and evaluation use separate LibriSpeech assets.
        self.base_path = (
            LIBRISPEECH_TRAIN_ASSET.extracted_path
            if split == DatasetSplit.TRAIN
            else LIBRISPEECH_CLEAN_ASSET.extracted_path
        )
        BaseDataset.__init__(self, self.base_path, split)
        self.target_sample_rate = target_sample_rate
        self.max_sequence_length = max_sequence_length
        self.max_text_length = max_text_length

    def __getitem__(
        self, index: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Parameters
        ----------
        index
            Index of the sample to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Padded audio tensor [max_sequence_length] and valid length scalar.
        torch.Tensor
            Tensor of shape [max_text_length] containing ASCII character
            codes for the transcription, padded with zeros.
        """
        audio_path = self.audio_files[index]
        audio_np, audio_sr = sf.read(str(audio_path), dtype="float32")

        # Convert to mono
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)

        # Resample to 16kHz if needed
        if audio_sr != self.target_sample_rate:
            audio_np = resample_poly(
                audio_np,
                self.target_sample_rate,
                audio_sr,
            ).astype(np.float32)

        # Truncate then pad to fixed length
        audio_len = min(len(audio_np), self.max_sequence_length)
        audio_np = audio_np[:audio_len]
        if audio_len < self.max_sequence_length:
            audio_np = np.pad(audio_np, (0, self.max_sequence_length - audio_len))
        audio = torch.from_numpy(audio_np)
        audio_length = torch.tensor(audio_len, dtype=torch.int32)

        # Transcription → ASCII tensor
        text = str(self.transcriptions[index])[: self.max_text_length]
        gt_tensor = torch.tensor([ord(c) for c in text], dtype=torch.int32)
        if len(gt_tensor) < self.max_text_length:
            gt_tensor = F.pad(
                gt_tensor, (0, self.max_text_length - len(gt_tensor)), value=0
            )

        return (audio, audio_length), gt_tensor

    def _validate_data(self) -> bool:
        self.audio_files = []
        self.transcriptions = []
        dataset_subdir = (
            "train-clean-100" if self.split == DatasetSplit.TRAIN else "test-clean"
        )
        dataset_path = self.base_path / dataset_subdir

        if not dataset_path.exists():
            return False

        for speaker_dir in sorted(dataset_path.glob("*")):
            if not speaker_dir.is_dir():
                continue
            for chapter_dir in sorted(speaker_dir.glob("*")):
                if not chapter_dir.is_dir():
                    continue
                trans_file = (
                    chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
                )
                if not trans_file.exists():
                    continue
                trans_dict = {}
                with open(trans_file, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            trans_dict[parts[0].strip()] = parts[1].strip().lower()
                for audio_path in sorted(chapter_dir.glob("*.flac")):
                    audio_id = audio_path.stem
                    if audio_id in trans_dict:
                        self.audio_files.append(audio_path)
                        self.transcriptions.append(trans_dict[audio_id])

        if not self.audio_files or len(self.audio_files) != len(self.transcriptions):
            raise ValueError("No audio files or transcriptions found.")

        return True

    def __len__(self) -> int:
        return len(self.audio_files)

    def _download_data(self) -> None:
        if self.split == DatasetSplit.TRAIN:
            LIBRISPEECH_TRAIN_ASSET.fetch(extract=True)
        else:
            LIBRISPEECH_CLEAN_ASSET.fetch(extract=True)

    @classmethod
    def dataset_name(cls) -> str:
        return "libri_speech"

    @staticmethod
    def default_samples_per_job() -> int:
        return 200

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://www.openslr.org/12",
            split_description="train-clean-100 for calibration; test-clean for evaluation",
        )
