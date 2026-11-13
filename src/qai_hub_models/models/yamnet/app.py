# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch

from qai_hub_models.protocols import ExecutableModelProtocol
from qai_hub_models.utils.labels import get_class_names

SAMPLE_RATE = 16000
CHUNK_LENGTH = 0.98


def preprocessing_yamnet_from_source(
    waveform_for_torch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    waveform_for_torch
        Tensor of audio of dimension (..., time)

    Returns
    -------
    patches : torch.Tensor
        batched torch tsr of shape [N, C, T]
    spectrogram : torch.Tensor
        Mel frequency spectrogram of size (..., ``n_mels``, time)
    """
    from torch_audioset.data.torch_input_processing import (
        WaveformToInput as TorchTransform,
    )

    #  This is a _log_ mel-spectrogram transform that adheres to the transform
    #  used by Google's vggish model input processing pipeline
    patches, spectrogram = TorchTransform().wavform_to_log_mel(
        waveform_for_torch, SAMPLE_RATE
    )

    return patches, spectrogram


def chunk_and_resample_audio(
    audio: np.ndarray,
    audio_sample_rate: int,
    model_sample_rate: int = SAMPLE_RATE,
    model_chunk_seconds: float = CHUNK_LENGTH,
) -> list[np.ndarray]:
    """
    Parameters
    ----------
    audio
        Raw audio numpy array of shape [# of samples]
    audio_sample_rate
        Sample rate of audio array, in samples / sec.
    model_sample_rate
        Sample rate (samples / sec) required to run Yamnet. The audio file
        will be resampled to use this rate.
    model_chunk_seconds
        Split the audio in to N sequences of this many seconds.
        The final split may be shorter than this many seconds.

    Returns
    -------
    audio_chunks : list[np.ndarray]
        List of audio arrays, chunked into N arrays of model_chunk_seconds seconds.
    """
    if audio_sample_rate != model_sample_rate:
        audio = resampy.resample(audio, audio_sample_rate, model_sample_rate)
        audio_sample_rate = model_sample_rate
    number_of_full_length_audio_chunks = int(
        audio.shape[1] // audio_sample_rate // model_chunk_seconds
    )
    last_sample_in_full_length_audio_chunks = int(
        audio_sample_rate * number_of_full_length_audio_chunks * model_chunk_seconds
    )
    if number_of_full_length_audio_chunks == 0:
        return [audio]

    return [
        *np.array_split(
            audio[:, :last_sample_in_full_length_audio_chunks],
            number_of_full_length_audio_chunks,
            axis=1,
        ),
    ]


def load_audiofile(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Decode the WAV file.

    Parameters
    ----------
    path
        Path of the input audio.

    Returns
    -------
    x : np.ndarray
        Reads audio sample from path and converts to torch tensor.
    sr : int
        sampling rate of audio samples
    """
    x, sr = sf.read(path, dtype="int16", always_2d=True)
    x = x / 2**15
    x = x.T.astype(np.float32)
    # Convert to mono and the sample rate expected by YAMNet.
    if x.shape[0] > 1:
        x = np.mean(x, axis=0, keepdims=True)
    return x, sr


class YamNetApp:
    """
    This class consists of light-weight "app code" that is required to
    perform end to end inference with an YamNetApp.

    For a given audio input, the app will:
        * Pre-process the audio
        * Run audio Classification
        * Return the class index of audio CHUNK_LENGTH of 0.98 seconds.
    """

    def __init__(
        self,
        model: ExecutableModelProtocol[torch.Tensor],
    ) -> None:
        """
        Parameters
        ----------
        model
            Executable model for inference.
        """
        self.model = model

    def predict(
        self, path: str | Path, audio_sample_rate: int | None = None
    ) -> list[str]:
        """
        Predict audio classes for given sample.

        Parameters
        ----------
        path
            Path to audio file if a string.
        audio_sample_rate
            The sample rate of the provided audio, in samples / second.
            If audio is a numpy array, this must be provided.
            If audio is a file and audio_sample_rate is None, this is ignored and the sample rate will be derived from the audio file.

        Returns
        -------
        predicted_classes : list[str]
            List of class ids from AudioSet-YouTube corpus is returned.
        """
        audio, audio_sample_rate = load_audiofile(path)

        assert audio_sample_rate is not None
        assert isinstance(audio, np.ndarray)
        acculist = []
        for x in chunk_and_resample_audio(audio, audio_sample_rate):
            pred = self.classify(x)
            acculist.append(pred)
        accu = np.stack(acculist)
        # Average them along time to get an overall classifier output for the clip.
        mean_scores = np.mean(accu, axis=0)
        mean_scores = mean_scores[0]
        top_N = 5
        # Report the highest-scoring classes.
        top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
        # Label each audio one-by-one, for all the chunks,
        actions = get_class_names("yamnet")
        return [actions[prediction] for prediction in top_class_indices]

    def classify(self, segment: np.ndarray) -> np.ndarray:
        """
        From the provided audio samples,calculate scores(matrix of
        time_frames, num_classes classifier scores).

        Parameters
        ----------
        segment
            chunked audio samples

        Returns
        -------
        class_probabilities : np.ndarray
            raw_prediction, class_probs for each chunk of audio samples
        """
        patches, _ = preprocessing_yamnet_from_source(torch.tensor(segment))
        # Inference using mdoel
        raw_prediction = self.model(patches)
        return raw_prediction.numpy()
