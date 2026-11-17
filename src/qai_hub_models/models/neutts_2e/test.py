# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qai_hub_models.models.neutts_2e.app import NeuTTS2EApp
from qai_hub_models.models.neutts_2e.demo import main as demo_main
from qai_hub_models.models.neutts_2e.model import SAMPLE_RATE, NeuTTS2E
from qai_hub_models.models.templates.voiceai_tts.test_utils import (
    assert_transcription_matches,
)
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo

SAMPLE_TEXT = "Hello there, nice to meet you."


def _assert_speech(wav: np.ndarray) -> None:
    assert np.isfinite(wav).all()
    assert len(wav) > SAMPLE_RATE // 2
    assert np.abs(wav).max() > 1e-2


def test_task(tmp_path: Path) -> None:
    app = NeuTTS2EApp(NeuTTS2E.from_pretrained(), seed=0)
    wav = app.synthesize(SAMPLE_TEXT, speaker="emily")
    assert app.sample_rate == SAMPLE_RATE
    _assert_speech(wav)

    wav_path = tmp_path / "out.wav"
    sf.write(wav_path, wav, SAMPLE_RATE)
    whisper = WhisperLargeV3Turbo.from_pretrained()
    assert_transcription_matches(
        wav_path,
        SAMPLE_TEXT,
        whisper.encoder.float(),
        whisper.decoder.float(),
        WhisperLargeV3Turbo.get_hf_whisper_version(),
    )


def test_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    demo_main(is_test=True)

    wav, sample_rate = sf.read(tmp_path / "neutts_2e_output.wav", dtype="float32")
    assert sample_rate == SAMPLE_RATE
    _assert_speech(wav)
