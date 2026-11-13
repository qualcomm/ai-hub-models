# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import torch
from qai_hub.client import DatasetEntries
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.melotts.model import (
    DEC_SEQ_OVERLAP,
    MAX_BERT_TOKENS,
    MAX_DEC_SEQ_LEN,
    MAX_SEQ_LEN,
    SDP,
    UPSAMPLE_FACTOR,
    UPSAMPLED_MAX_SEQ_LEN,
    Decoder,
    Encoder,
    Flow,
    MeloTTS,
    get_tts_object,
)
from qai_hub_models.models.templates.melotts.utils import download_unidic
from qai_hub_models.models.templates.voiceai_tts.app_utils import (
    ByT5Tokenizer,
    calibrate_charsiu_decoder,
    calibrate_charsiu_encoder,
    generate_path,
)
from qai_hub_models.models.templates.voiceai_tts.language import BERT_MODEL_IDS
from qai_hub_models.utils.base_app import CollectionAppQuantizeProtocol
from qai_hub_models.utils.evaluate.helpers import sample_dataset
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

if TYPE_CHECKING:
    from melo.api import TTS

DEFAULT_TEXTS = {
    TTSLanguage.ENGLISH: "Text to speech helps people communicate.",
    TTSLanguage.SPANISH: "La conversión de texto a voz ayuda a las personas a comunicarse.",
    TTSLanguage.CHINESE: "文本转语音技术有助于人们进行交流。",
    TTSLanguage.KOREAN: "텍스트 음성 변환 기술은 사람들이 의사소통하는 데 도움을 줍니다.",
    TTSLanguage.JAPANESE: "富士山の標高は3776メートルで、前回の大規模な噴火は1707年に発生し、それから300年以上が経過しています。",
}
CommonVoice_FOLDER_NAME = "common_voice"
CommonVoice_VERSION = 1


def get_text_for_tts_infer(
    *args: Any, **kwargs: Any
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    download_unidic()
    from melo.utils import get_text_for_tts_infer

    return get_text_for_tts_infer(*args, **kwargs)


class MeloTTSApp(CollectionAppQuantizeProtocol):
    def __init__(
        self,
        encoder: Encoder,
        sdp: SDP,
        flow: Flow,
        decoder: Decoder,
        tts_object: "TTS",
        language: TTSLanguage,
    ) -> None:
        super().__init__()
        self.language = language
        self.encoder = encoder
        self.sdp = sdp
        self.flow = flow
        self.decoder = decoder
        self.tts_object = tts_object
        # different speaker_id has different accent
        # for example, myshell-ai/MeloTTS-English ckpt has 5 accents
        """
        "spk2id": {
          "EN-US": 0,
          "EN-BR": 1,
          "EN_INDIA": 2,
          "EN-AU": 3,
          "EN-Default": 4
        }
        """
        self.speaker_id = next(
            iter(tts_object.hps.data.spk2id.values())
        )  # csim speaker_id is 0 by default

    def predict(self, text: str) -> str:
        """
        Parameters
        ----------
        text
            the text needed to synthesized into audio

        Returns
        -------
        output_path : str
            Synthesized audio path.
        """
        output_path = f"synthesized-audio_{self.language}.wav"
        self.tts_to_file(
            text,
            self.speaker_id,
            output_path,
        )
        return output_path

    @classmethod
    def preprocess_text(
        cls, tts_object: "TTS", text: str
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """
        The helper function to convert text to phones and tone.

        Parameters
        ----------
        tts_object
             the instance of Melo TTS class
        text
             the text that need to be synthesized into audio.

        Returns
        -------
        phones : Tensor
            the phones of input text, shape of (1, MAX_SEQ_LEN)
        tones : Tensor
            the tone of input text, shape of (1, MAX_SEQ_LEN)
        lang_ids : Tensor
            shape of (1, MAX_SEQ_LEN)
        bert : Tensor
            shape of (1, BERT_FEATURE_DIM, MAX_SEQ_LEN)
        ja_bert : Tensor
            shape of (1, JA_BERT_FEATURE_DIM, MAX_SEQ_LEN)
        phone_len : int
            the actual length of phones
        """
        bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
            text,
            tts_object.language,
            tts_object.hps,
            "cpu",
            tts_object.symbol_to_id,
        )
        phone_len = phones.size(0)
        phones = F.pad(phones, (0, MAX_SEQ_LEN - phones.size(0)))[:MAX_SEQ_LEN]
        tones = F.pad(tones, (0, MAX_SEQ_LEN - tones.size(0)))[:MAX_SEQ_LEN]
        lang_ids = F.pad(lang_ids, (0, MAX_SEQ_LEN - lang_ids.size(0)))[:MAX_SEQ_LEN]
        bert = F.pad(bert, (0, MAX_SEQ_LEN - bert.size(1), 0, 0))[:, :MAX_SEQ_LEN]
        ja_bert = F.pad(ja_bert, (0, MAX_SEQ_LEN - ja_bert.size(1), 0, 0))[
            :, :MAX_SEQ_LEN
        ]

        return phones, tones, lang_ids, bert, ja_bert, phone_len

    def tts_to_file(
        self,
        text: str,
        speaker_id: int,
        output_path: str,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        sdp_ratio: float = 0.2,
    ) -> None:
        """
        Synthesize audio from text.

        Parameters
        ----------
        text
            the text that need to transfer to audio
        speaker_id
            the id of the default speaker
        output_path
            the path to save the synthesized audio file
        noise_scale
            the noise scale of the synthesized audio
        length_scale
            the length scale of the synthesized audio
        noise_scale_w
            the weight noise scale of the synthesized audio
        sdp_ratio
            the sdp ratio of the synthesized audio
        """
        # Encoder input
        phones, tones, lang_ids, bert, ja_bert, phone_len = self.preprocess_text(
            self.tts_object, text
        )
        x = phones.unsqueeze(0)
        x_lengths = torch.tensor([phone_len], dtype=torch.int64)
        sid = torch.tensor([speaker_id], dtype=torch.int64)
        tone = tones.unsqueeze(0)
        language_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
        length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
        noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)
        x_encoded, m_p, logs_p, x_mask, g = self.encoder(
            x,
            x_lengths,
            tone,
            sid,
            language_ids,
            bert,
            ja_bert,
        )
        y_lengths, w_ceil = self.sdp(
            x_encoded, x_mask, g, sdp_ratio_pt, length_scale_pt, noise_scale_w_pt
        )

        # Flow input
        y_mask = torch.unsqueeze(
            torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1), dim=1
        ).to(torch.float32)

        attn_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
        attn = generate_path(w_ceil, attn_mask)
        attn_squeezed = attn.squeeze(1).to(torch.float32)

        m_p = m_p.to(torch.float32)
        logs_p = logs_p.to(torch.float32)
        noise_scale_pt = torch.tensor([noise_scale], dtype=torch.float32)
        z = self.flow(m_p, logs_p, y_mask, g, attn_squeezed, noise_scale_pt)

        # Decoder input
        z_buf = torch.zeros(
            [z.shape[0], z.shape[1], MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP],
            dtype=torch.float32,
        )
        z_buf[:, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[
            :, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
        ]
        audio_chunk = self.decoder(z_buf, g)

        audio = audio_chunk.squeeze()[: MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR]
        total_dec_seq_len = MAX_DEC_SEQ_LEN
        while total_dec_seq_len < y_lengths:
            z_buf = z[
                :,
                :,
                total_dec_seq_len - DEC_SEQ_OVERLAP : total_dec_seq_len
                + MAX_DEC_SEQ_LEN
                + DEC_SEQ_OVERLAP,
            ]
            audio_chunk = self.decoder(z_buf, g)

            audio_chunk = audio_chunk.squeeze()[
                DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
                * UPSAMPLE_FACTOR
            ]
            audio = torch.cat([audio, audio_chunk])
            total_dec_seq_len += MAX_DEC_SEQ_LEN

        length = int(y_lengths[0]) * UPSAMPLE_FACTOR

        audio = audio.squeeze()[:length]
        sf.write(
            output_path,
            audio.detach().cpu().numpy(),
            samplerate=self.tts_object.hps.data.sampling_rate,
        )

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: MeloTTS,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        language = collection_model.get_language()
        encoder_fpm = collection_model.components["encoder"]
        sdp_fpm = collection_model.components["sdp"]
        flow_fpm = collection_model.components["flow"]
        batch_size = get_batch_size(input_spec) or 1
        assert callable(encoder_fpm) and callable(sdp_fpm) and callable(flow_fpm)
        assert batch_size == 1, f"Batch size must be 1, found {batch_size}"

        tts_object = get_tts_object(language)
        speaker_id = next(iter(tts_object.hps.data.spk2id.values()))
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_IDS[language])
        t5_tokenizer = ByT5Tokenizer()
        noise_scale = 0.667
        length_scale = 1.0
        noise_scale_w = 0.8
        sdp_ratio = 0.2

        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            lang=language,
        )
        num_samples = num_samples or dataset.default_samples_per_job()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in input_spec]
        n = 0
        for text, _ in dataloader:
            if isinstance(text, tuple | list):
                text = text[0]  # batch_size is 1
            if component_name == "encoder":
                phones, tones, lang_ids, bert, ja_bert, phone_len = cls.preprocess_text(
                    tts_object, text=text
                )
                x = phones.unsqueeze(0)
                x_lengths = torch.tensor([phone_len], dtype=torch.int64)
                sid = torch.tensor([speaker_id], dtype=torch.int64)
                tone = tones.unsqueeze(0)
                language_ids = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                inputs[0].append(x)
                inputs[1].append(x_lengths)
                inputs[2].append(tone)
                inputs[3].append(sid)
                inputs[4].append(language_ids)
                inputs[5].append(bert)
                inputs[6].append(ja_bert)
            elif component_name == "sdp":
                phones, tones, lang_ids, bert, ja_bert, phone_len = cls.preprocess_text(
                    tts_object, text=text
                )
                x = phones.unsqueeze(0)
                x_lengths = torch.tensor([phone_len], dtype=torch.int64)
                sid = torch.tensor([speaker_id], dtype=torch.int64)
                tone = tones.unsqueeze(0)
                language_ids = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
                length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
                noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)
                x_encoded, m_p, logs_p, x_mask, g = encoder_fpm(
                    x,
                    x_lengths,
                    tone,
                    sid,
                    language_ids,
                    bert,
                    ja_bert,
                )
                inputs[0].append(x_encoded)
                inputs[1].append(x_mask)
                inputs[2].append(g)
                inputs[3].append(sdp_ratio_pt)
                inputs[4].append(length_scale_pt)
                inputs[5].append(noise_scale_w_pt)
            elif component_name == "flow":
                phones, tones, lang_ids, bert, ja_bert, phone_len = cls.preprocess_text(
                    tts_object, text=text
                )
                x = phones.unsqueeze(0)
                x_lengths = torch.tensor([phone_len], dtype=torch.int64)
                sid = torch.tensor([speaker_id], dtype=torch.int64)
                tone = tones.unsqueeze(0)
                language_ids = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
                length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
                noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)

                x_encoded, m_p, logs_p, x_mask, g = encoder_fpm(
                    x,
                    x_lengths,
                    tone,
                    sid,
                    language_ids,
                    bert,
                    ja_bert,
                )
                y_lengths, w_ceil = sdp_fpm(
                    x_encoded,
                    x_mask,
                    g,
                    sdp_ratio_pt,
                    length_scale_pt,
                    noise_scale_w_pt,
                )

                y_mask = torch.unsqueeze(
                    torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1),
                    dim=1,
                ).to(torch.float32)
                attn_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
                attn = generate_path(w_ceil, attn_mask)
                attn_squeezed = attn.squeeze(1).to(torch.float32)

                m_p = m_p.to(torch.float32)
                logs_p = logs_p.to(torch.float32)
                noise_scale_pt = torch.tensor([noise_scale], dtype=torch.float32)

                inputs[0].append(m_p)
                inputs[1].append(logs_p)
                inputs[2].append(y_mask)
                inputs[3].append(g)
                inputs[4].append(attn_squeezed)
                inputs[5].append(noise_scale_pt)

            elif component_name == "decoder":
                phones, tones, lang_ids, bert, ja_bert, phone_len = cls.preprocess_text(
                    tts_object, text=text
                )
                x = phones.unsqueeze(0)
                x_lengths = torch.tensor([phone_len], dtype=torch.int64)
                sid = torch.tensor([speaker_id], dtype=torch.int64)
                tone = tones.unsqueeze(0)
                language_ids = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
                length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
                noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)

                x_encoded, m_p, logs_p, x_mask, g = encoder_fpm(
                    x,
                    x_lengths,
                    tone,
                    sid,
                    language_ids,
                    bert,
                    ja_bert,
                )
                y_lengths, w_ceil = sdp_fpm(
                    x_encoded,
                    x_mask,
                    g,
                    sdp_ratio_pt,
                    length_scale_pt,
                    noise_scale_w_pt,
                )

                y_mask = torch.unsqueeze(
                    torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1),
                    dim=1,
                ).to(torch.float32)
                attn_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
                attn = generate_path(w_ceil, attn_mask)
                attn_squeezed = attn.squeeze(1).to(torch.float32)

                m_p = m_p.to(torch.float32)
                logs_p = logs_p.to(torch.float32)
                noise_scale_pt = torch.tensor([noise_scale], dtype=torch.float32)

                z = flow_fpm(m_p, logs_p, y_mask, g, attn_squeezed, noise_scale_pt)

                z_buf = torch.zeros(
                    [z.shape[0], z.shape[1], MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP],
                    dtype=torch.float32,
                )
                z_buf[:, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[
                    :, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
                ]
                inputs[0].append(z_buf)
                inputs[1].append(g)

                total_dec_seq_len = MAX_DEC_SEQ_LEN
                while total_dec_seq_len < min(
                    z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP, int(y_lengths[0])
                ):
                    z_buf = z[
                        :,
                        :,
                        total_dec_seq_len - DEC_SEQ_OVERLAP : total_dec_seq_len
                        + MAX_DEC_SEQ_LEN
                        + DEC_SEQ_OVERLAP,
                    ]
                    inputs[0].append(z_buf)
                    inputs[1].append(g)
                    total_dec_seq_len += MAX_DEC_SEQ_LEN
            elif component_name == "bert_wrapper":
                input_ = tokenizer(
                    text,
                    padding="max_length",
                    max_length=MAX_BERT_TOKENS,
                    return_tensors="pt",
                )
                inputs[0].append(
                    torch.tensor(input_["input_ids"].numpy(), dtype=torch.int32)
                )
                inputs[1].append(
                    torch.tensor(input_["attention_mask"].numpy(), dtype=torch.int32)
                )
                inputs[2].append(
                    torch.tensor(input_["token_type_ids"].numpy(), dtype=torch.int32)
                )
            elif component_name == "charsiu_encoder":
                calibrate_charsiu_encoder(text, t5_tokenizer, inputs)

            elif component_name == "charsiu_decoder":
                if n > 50:  # 100 samples is around 4.5G data
                    break
                assert hasattr(collection_model, "charsiu_encoder")
                assert hasattr(collection_model, "charsiu_decoder")
                T5Encoder = collection_model.charsiu_encoder
                T5Decoder = collection_model.charsiu_decoder
                calibrate_charsiu_decoder(
                    text, t5_tokenizer, T5Encoder, T5Decoder, inputs
                )
            else:
                raise NotImplementedError(component_name)
            n = n + 1

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
