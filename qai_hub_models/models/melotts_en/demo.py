# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.melotts.app import DEFAULT_TEXTS, MeloTTSApp
from qai_hub_models.models.melotts_en.model import MeloTTS_EN
from qai_hub_models.utils.args import get_model_cli_parser, model_from_cli_args


def main(is_test: bool = False) -> None:
    parser = get_model_cli_parser(MeloTTS_EN)

    args = parser.parse_args([] if is_test else None)
    model = model_from_cli_args(MeloTTS_EN, args)
    app = MeloTTSApp(
        model.encoder, model.flow, model.decoder, model.tts_object, model.language()
    )

    audio_path = app.predict(DEFAULT_TEXTS[model.language()])
    print(f"ENGLISH Audio generated and saved to {audio_path}")


if __name__ == "__main__":
    main()
