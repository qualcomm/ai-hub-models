# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.pipertts.model import (
    PiperTTS,
)

MODEL_ID = __name__.split(".")[-2]


class PiperTTS_EN_AMY(PiperTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ENGLISH

    @classmethod
    def get_voice_id(cls) -> str:
        return "amy"

    @classmethod
    def get_metadata_name_suffix(cls) -> str:
        return "-amy"
