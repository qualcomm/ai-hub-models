# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import copy
import os

import nltk

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.models._shared.melotts.model import (
    MAX_NUM_INPUT_IDS,
    BertWrapper,
    Decoder,
    Encoder,
    Flow,
    T5Decoder,
    T5Encoder,
    get_bert_model,
    get_t5model,
    get_tts_object,
)
from qai_hub_models.models._shared.melotts.utils import (
    write_melotts_supplementary_files,
)
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]


class Encoder_EN(Encoder):
    @classmethod
    def from_pretrained(cls) -> "Encoder_EN":
        return cls(get_tts_object("ENGLISH"), speed_adjustment=0.85)


class Flow_EN(Flow):
    @classmethod
    def from_pretrained(cls) -> "Flow_EN":
        return cls(get_tts_object("ENGLISH"))


class Decoder_EN(Decoder):
    @classmethod
    def from_pretrained(cls) -> "Decoder_EN":
        return cls(get_tts_object("ENGLISH"))


class BertWrapper_EN(BertWrapper):
    @classmethod
    def from_pretrained(cls) -> "BertWrapper_EN":
        return cls(get_bert_model("ENGLISH"))


class T5Encoder_EN(T5Encoder):
    @classmethod
    def from_pretrained(cls) -> "T5Encoder_EN":
        return cls(get_t5model())


class T5Decoder_EN(T5Decoder):
    @classmethod
    def from_pretrained(cls) -> "T5Decoder_EN":
        # here the t5model is passed to T5Decoder by reference
        # use deepcopy to prevent cached t5model being modified, so the cache can be reused
        return cls(copy.deepcopy(get_t5model()), MAX_NUM_INPUT_IDS)


@CollectionModel.add_component(Encoder_EN)
@CollectionModel.add_component(Flow_EN)
@CollectionModel.add_component(Decoder_EN)
@CollectionModel.add_component(BertWrapper_EN)
@CollectionModel.add_component(T5Encoder_EN)
@CollectionModel.add_component(T5Decoder_EN)
class MeloTTS_EN(CollectionModel):
    def __init__(
        self,
        encoder: Encoder,
        flow: Flow,
        decoder: Decoder,
        bert_model: BertWrapper,
        charsiu_encoder: T5Encoder,
        charsiu_decoder: T5Decoder,
    ) -> None:
        super().__init__(
            encoder, flow, decoder, bert_model, charsiu_encoder, charsiu_decoder
        )
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.bert_model = bert_model
        self.charsiu_encoder = charsiu_encoder
        self.charsiu_decoder = charsiu_decoder
        self.speaker_id = encoder.speaker_id
        self.tts_object = get_tts_object("ENGLISH")

    @classmethod
    def language(cls) -> str:
        return "ENGLISH"

    @classmethod
    def from_pretrained(cls) -> "MeloTTS_EN":
        nltk.download("averaged_perceptron_tagger_eng")
        return cls(
            Encoder_EN.from_pretrained(),
            Flow_EN.from_pretrained(),
            Decoder_EN.from_pretrained(),
            BertWrapper_EN.from_pretrained(),
            T5Encoder_EN.from_pretrained(),
            T5Decoder_EN.from_pretrained(),
        )

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_melotts_supplementary_files("ENGLISH", output_dir, metadata)
