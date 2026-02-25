# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import re

import soundfile

from qai_hub_models.models._shared.hf_whisper.app import HfWhisperApp
from qai_hub_models.models._shared.melotts.app import DEFAULT_TEXTS, MeloTTSApp
from qai_hub_models.models.melotts_en.demo import main as demo_main
from qai_hub_models.models.melotts_en.model import MeloTTS_EN
from qai_hub_models.models.whisper_large_v3_turbo.model import WhisperLargeV3Turbo


def test_synthesized_audio() -> None:
    model = MeloTTS_EN.from_pretrained()
    out_audio_path = MeloTTSApp(
        model.encoder, model.flow, model.decoder, model.tts_object, model.language()
    ).predict(DEFAULT_TEXTS[model.language()])

    wav, audio_sample_rate = soundfile.read(out_audio_path)

    model_sr = WhisperLargeV3Turbo.from_pretrained()
    app_sr = HfWhisperApp(
        model_sr.encoder, model_sr.decoder, WhisperLargeV3Turbo.get_hf_whisper_version()
    )
    transcription = app_sr.transcribe(wav, audio_sample_rate)
    trans = "".join(re.findall(r"\b\w+\b", transcription))

    original_text = DEFAULT_TEXTS[model.language()]
    original_text = "".join(re.findall(r"\b\w+\b", original_text))
    print(
        "\nOriginal_text: ",
        DEFAULT_TEXTS[model.language()],
        "\n",
        "Transcription: ",
        transcription,
        sep="",
    )
    assert trans.lower() == original_text.lower()


def test_demo() -> None:
    demo_main(is_test=True)
