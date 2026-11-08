# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import soundfile

from qai_hub_models.models.parakeet_tdt.app import ParakeetTDTApp
from qai_hub_models.models.parakeet_tdt.model import (
    HF_MODEL_ID,
    ParakeetTDT,
)
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset


def load_demo_audio() -> str:
    TEST_AUDIO_PATH = CachedWebModelAsset.from_asset_store(
        "hf_whisper_asr_shared",
        "1",
        "audio/common_voice_en_19653650.wav",
    )
    TEST_AUDIO_PATH.fetch()
    return TEST_AUDIO_PATH.path


def main(is_test: bool = False) -> None:
    """Run Parakeet TDT demo for speech recognition."""
    parser = get_model_cli_parser(ParakeetTDT)
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for transcription (.wav or .flac format).",
    )
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default=HF_MODEL_ID,
        help="HuggingFace model ID or local path for processor/tokenizer.",
    )
    args = parser.parse_args([] if is_test else None)

    model = model_from_cli_args(ParakeetTDT, args)
    app = ParakeetTDTApp(
        encoder=model.encoder,
        decoder=model.decoder,
        joint=model.joint,
        hf_model_id=args.hf_model_id,
    )

    wav_file = args.audio_file or load_demo_audio()
    audio, sample_rate = soundfile.read(wav_file)

    transcription = app.transcribe(audio, sample_rate)
    if not is_test:
        print("Transcription:", transcription)


if __name__ == "__main__":
    main()
