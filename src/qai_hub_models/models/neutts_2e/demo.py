# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from qai_hub_models.models.neutts_2e.app import EMOTIONS, SPEAKERS, NeuTTS2EApp
from qai_hub_models.models.neutts_2e.model import NeuTTS2E
from qai_hub_models.utils.args import (
    add_output_dir_arg,
    get_model_cli_parser,
    model_from_cli_args,
)

DEFAULT_INPUT_TEXT = "I can't believe it's finally here!"
DEFAULT_SPEAKER = "emily"
DEFAULT_EMOTION = "happy"
DEFAULT_SEED = 0


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(NeuTTS2E)
    parser.add_argument("--input-text", type=str, default=DEFAULT_INPUT_TEXT)
    parser.add_argument(
        "--speaker",
        type=str,
        default=DEFAULT_SPEAKER,
        choices=SPEAKERS,
        help="Which of the four fixed speakers to synthesize as. NeuTTS-2E does "
        "not clone voices; each speaker ships as pre-encoded reference codes.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default=DEFAULT_EMOTION,
        choices=EMOTIONS,
        help="Emotion to condition on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Sampling seed; generation is reproducible for a fixed seed.",
    )
    add_output_dir_arg(parser)
    args = parser.parse_args([] if is_test else None)

    model = model_from_cli_args(NeuTTS2E, args)
    app = NeuTTS2EApp(model, seed=args.seed)

    wav = app.synthesize(args.input_text, speaker=args.speaker, emotion=args.emotion)

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "neutts_2e_output.wav"
    sf.write(output_path, wav, app.sample_rate)
    print(
        f"Synthesized {len(wav) / app.sample_rate:.2f}s as {args.speaker} "
        f"({args.emotion}) and saved to {output_path}"
    )


if __name__ == "__main__":
    main()
