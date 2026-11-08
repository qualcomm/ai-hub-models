# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import mecab_ko
from g2pkk.g2pkk import G2p
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.melotts.model import (
    SDP,
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    T5Decoder,
    T5Encoder,
    get_tts_object,
)

MODEL_ID = __name__.split(".")[-2]


class _MeCabKo:
    def __init__(self) -> None:
        self._tagger = mecab_ko.Tagger()

    def pos(self, sentence: str) -> list[tuple[str, str]]:
        tokens = []
        node = self._tagger.parseToNode(sentence)
        while node:
            if node.surface:
                tokens.append((node.surface, node.feature.split(",")[0]))
            node = node.next
        return tokens


# g2pkk loads `mecab` (python-mecab-ko, no wheels past cp312) and pip-installs it at
# runtime if missing. Serve its tagger from mecab-ko instead, which bundles libmecab.
G2p.check_mecab = lambda self: None
G2p.get_mecab = lambda self: _MeCabKo()


class Encoder_KO(Encoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.KOREAN), speed_adjustment=1.1)


class Sdp_KO(SDP):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.KOREAN), speed_adjustment=1.1)


class Flow_KO(Flow):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.KOREAN))


class Decoder_KO(Decoder):
    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_tts_object(TTSLanguage.KOREAN))

    def component_precision(self) -> Precision:
        return Precision.float


class BertWrapper_KO(BertWrapper):
    language = TTSLanguage.KOREAN

    def component_precision(self) -> Precision:
        return Precision.w8a16


class MeloTTS_KO(MeloTTS):
    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.KOREAN

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(
            Encoder_KO.from_pretrained(),
            Sdp_KO.from_pretrained(),
            Flow_KO.from_pretrained(),
            Decoder_KO.from_pretrained(),
            bert_wrapper=BertWrapper_KO.from_pretrained(),
            charsiu_encoder=T5Encoder.from_pretrained(),
            charsiu_decoder=T5Decoder.from_pretrained(),
        )
