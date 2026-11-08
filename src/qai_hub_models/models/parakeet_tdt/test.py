# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import re
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile
import torch

import qai_hub_models.models.parakeet_tdt.app as app_module
from qai_hub_models.models.parakeet_tdt.app import ParakeetTDTApp
from qai_hub_models.models.parakeet_tdt.demo import load_demo_audio
from qai_hub_models.models.parakeet_tdt.demo import main as demo_main
from qai_hub_models.models.parakeet_tdt.model import (
    HF_MODEL_ID,
    JOINT_HIDDEN_SIZE,
    MAX_TIME_STEPS,
    NUM_MEL_BINS,
    ParakeetTDT,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def test_task() -> None:
    """Validate the complete autoregressive Parakeet transcription flow."""
    model = ParakeetTDT.from_pretrained(HF_MODEL_ID)
    app = ParakeetTDTApp(
        encoder=model.encoder,
        decoder=model.decoder,
        joint=model.joint,
        hf_model_id=HF_MODEL_ID,
    )

    audio, sample_rate = soundfile.read(load_demo_audio())
    transcription = app.transcribe(
        np.asarray(audio, dtype=np.float32),
        sample_rate,
    )

    reference_asset = CachedWebModelAsset.from_asset_store(
        "parakeet_tdt",
        "1",
        "common_voice_en_19653650.txt",
    )
    reference_asset.fetch()
    expected = Path(reference_asset.path).read_text(encoding="utf-8")

    def normalize(text: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", "", text.casefold()).split())

    assert normalize(transcription) == normalize(expected), (
        f"Transcription mismatch.\n"
        f"Expected: {normalize(expected)!r}\n"
        f"Actual:   {normalize(transcription)!r}"
    )


def test_long_audio_on_device_encoder_chunks() -> None:
    """Validate static encoder inputs for long-audio on-device chunking."""

    class FakeAsyncResult:
        def __init__(self, output: torch.Tensor) -> None:
            self.output = output

        def wait(self) -> torch.Tensor:
            return self.output

    class FakeOnDeviceModel:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def async_model(
            self,
            features: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> FakeAsyncResult:
            self.calls.append((features, attention_mask))
            output = torch.zeros(
                1,
                MAX_TIME_STEPS // 8,
                JOINT_HIDDEN_SIZE,
                dtype=torch.float32,
            )
            return FakeAsyncResult(output)

    app = ParakeetTDTApp.__new__(ParakeetTDTApp)
    fake_encoder = FakeOnDeviceModel()
    app.encoder = fake_encoder  # type: ignore[assignment]

    # Force more than one audio chunk.
    audio = np.zeros(
        app_module.CHUNK_SAMPLES * 2 + 1,
        dtype=np.float32,
    )
    chunks, has_left_overlap = app._chunk_audio(audio)

    assert len(chunks) > 1
    assert has_left_overlap[0] is False
    assert all(has_left_overlap[1:])
    assert all(len(chunk) <= app_module.MAX_AUDIO_SAMPLES for chunk in chunks)

    chunk_inputs: list[dict[str, torch.Tensor]] = []
    for index, chunk in enumerate(chunks):
        time_steps = max(1, len(chunk) // 160)

        # Make one input deliberately oversized so the defensive truncation
        # branch in _run_encoder_chunks is exercised.
        if index == 1:
            time_steps = MAX_TIME_STEPS + 4

        chunk_inputs.append(
            {
                "input_features": torch.zeros(
                    1,
                    time_steps,
                    NUM_MEL_BINS,
                    dtype=torch.float32,
                ),
                "attention_mask": torch.ones(
                    1,
                    time_steps,
                    dtype=torch.int32,
                ),
            }
        )

    with patch.object(app_module, "OnDeviceModel", FakeOnDeviceModel):
        outputs, valid_lengths = app._run_encoder_chunks(chunk_inputs)

    assert len(outputs) == len(chunks)
    assert len(valid_lengths) == len(chunks)
    assert all(length <= MAX_TIME_STEPS // 8 for length in valid_lengths)

    assert len(fake_encoder.calls) == len(chunks)
    for features, attention_mask in fake_encoder.calls:
        assert features.shape == (1, MAX_TIME_STEPS, NUM_MEL_BINS)
        assert attention_mask.shape == (1, MAX_TIME_STEPS)
        assert attention_mask.dtype == torch.int32


def test_demo() -> None:
    demo_main(is_test=True)
