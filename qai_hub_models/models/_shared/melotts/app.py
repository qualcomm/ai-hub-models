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

from qai_hub_models.datasets import DatasetSplit, get_dataset_from_name
from qai_hub_models.models._shared.melotts.model import (
    Decoder,
    Encoder,
    Flow,
    get_tts_object,
)
from qai_hub_models.models._shared.melotts.utils import (
    BERT_MODEL_IDS,
    LANGUAGE_MAP,
    download_unidic,
)
from qai_hub_models.utils.base_app import CollectionAppProtocol
from qai_hub_models.utils.base_model import BaseModel, CollectionModel
from qai_hub_models.utils.evaluate import sample_dataset
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

if TYPE_CHECKING:
    from melo.api import TTS

MAX_SEQ_LEN = 512
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_OVERLAP = 12
UPSAMPLE_FACTOR = 512

ENCODER_HIDDEN_DIM = 192
DECODER_Z_TIME_DIM = 64
MAX_BERT_TOKENS = 200
DEFAULT_TEXTS = {
    "ENGLISH": "This is an example of text to speech for English. How does it sound?",
    "SPANISH": "Este es un ejemplo de texto a voz en inglés. ¿Cómo suena?",
    "CHINESE": "中文是中国的语言文字。特指汉族的语言文字, 即汉语和汉字",
}
CommonVoice_FOLDER_NAME = "common_voice"
CommonVoice_VERSION = 1


def generate_path(duration: Tensor, mask: Tensor) -> Tensor:
    """
    Parameters
    ----------
    duration
        shape of [b, 1, t_x], duration time
    mask
        shape of [b, 1, t_y, t_x], attention mask

    Returns
    -------
    attention : Tensor
        the generated self attention
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)
    cum_duration_flat = cum_duration.view(b * t_x)
    x = torch.arange(
        t_y, dtype=cum_duration_flat.dtype, device=cum_duration_flat.device
    )
    path = (x.unsqueeze(0) < cum_duration_flat.unsqueeze(1)).to(mask.dtype)
    path = path.view(b, t_x, t_y)

    layer = [[0, 0], [1, 0], [0, 0]][::-1]
    pad_shape = [item for sublist in layer for item in sublist]

    path = path - F.pad(path, pad_shape)[:, :-1]
    return path.unsqueeze(1).transpose(2, 3) * mask


def get_text_for_tts_infer(
    *args: Any, **kwargs: Any
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    download_unidic()
    from melo.utils import get_text_for_tts_infer

    return get_text_for_tts_infer(*args, **kwargs)


class MeloTTSApp(CollectionAppProtocol):
    def __init__(
        self,
        encoder: Encoder,
        flow: Flow,
        decoder: Decoder,
        tts_object: "TTS",
        language: str,
    ) -> None:
        super().__init__()
        self.language = language
        self.encoder = encoder
        self.flow = flow
        self.decoder = decoder
        self.tts_object = tts_object
        self.speaker_id = next(iter(tts_object.hps.data.spk2id.values()))

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
        language = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)
        sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
        length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
        noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)

        y_lengths, x_mask, m_p, logs_p, g, w_ceil = self.encoder(
            x,
            x_lengths,
            tone,
            sid,
            language,
            bert,
            ja_bert,
            sdp_ratio_pt,
            length_scale_pt,
            noise_scale_w_pt,
        )

        # Flow input
        y_mask = torch.unsqueeze(
            torch.arange(MAX_SEQ_LEN * 3) < y_lengths[:, None], dim=1
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

        length = int(y_lengths[0]) * 512

        audio = audio.squeeze()[:length]
        sf.write(
            output_path,
            audio.squeeze().numpy(),
            samplerate=self.tts_object.hps.data.sampling_rate,
        )

    @classmethod
    def get_calibration_data(
        cls,
        model: BaseModel,
        calibration_dataset_name: str,
        num_samples: int | None,
        input_spec: InputSpec,
        collection_model: CollectionModel,
    ) -> DatasetEntries:
        assert hasattr(collection_model, "language") and callable(
            collection_model.language
        )
        language_ = collection_model.language()
        encoder_fpm = collection_model.components[f"Encoder_{LANGUAGE_MAP[language_]}"]
        flow_fpm = collection_model.components[f"Flow_{LANGUAGE_MAP[language_]}"]
        batch_size = get_batch_size(input_spec) or 1
        assert callable(encoder_fpm) and callable(flow_fpm)
        assert batch_size == 1, f"Batch size must be 1, found {batch_size}"

        tts_object = get_tts_object(language_)
        speaker_id = next(iter(tts_object.hps.data.spk2id.values()))
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_IDS[language_])
        noise_scale = (0.667,)
        length_scale = 1.0
        noise_scale_w = 0.8
        sdp_ratio = 0.2

        dataset = get_dataset_from_name(
            calibration_dataset_name, DatasetSplit.TRAIN, lang=LANGUAGE_MAP[language_]
        )
        num_samples = num_samples or dataset.default_samples_per_job()
        num_samples = (num_samples // batch_size) * batch_size
        print(f"Loading {num_samples} calibration samples.")
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in input_spec]
        for text, _ in dataloader:
            if isinstance(text, tuple | list):
                text = text[0]  # batch_size is 1
            if model._get_name() == f"Flow_{LANGUAGE_MAP[language_]}":
                phones, tones, lang_ids, bert, ja_bert, phone_len = cls.preprocess_text(
                    tts_object, text=text
                )
                x = phones.unsqueeze(0)
                x_lengths = torch.tensor([phone_len], dtype=torch.int64)
                sid = torch.tensor([speaker_id], dtype=torch.int64)
                tone = tones.unsqueeze(0)
                language = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
                length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
                noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)

                y_lengths, x_mask, m_p, logs_p, g, w_ceil = encoder_fpm(
                    x,
                    x_lengths,
                    tone,
                    sid,
                    language,
                    bert,
                    ja_bert,
                    sdp_ratio_pt,
                    length_scale_pt,
                    noise_scale_w_pt,
                )

                y_mask = torch.unsqueeze(
                    torch.arange(MAX_SEQ_LEN * 3) < y_lengths[:, None], dim=1
                ).to(torch.float32)
                attn_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
                attn = generate_path(w_ceil, attn_mask)
                attn_squeezed = attn.squeeze(1).to(torch.float32)

                m_p = m_p.to(torch.float32)
                logs_p = logs_p.to(torch.float32)
                noise_scale_pt = torch.tensor(noise_scale, dtype=torch.float32)

                inputs[0].append(m_p)
                inputs[1].append(logs_p)
                inputs[2].append(y_mask)
                inputs[3].append(g)
                inputs[4].append(attn_squeezed)
                inputs[5].append(noise_scale_pt)

            elif model._get_name() == f"Decoder_{LANGUAGE_MAP[language_]}":
                phones, tones, lang_ids, bert, ja_bert, phone_len = cls.preprocess_text(
                    tts_object, text=text
                )
                x = phones.unsqueeze(0)
                x_lengths = torch.tensor([phone_len], dtype=torch.int64)
                sid = torch.tensor([speaker_id], dtype=torch.int64)
                tone = tones.unsqueeze(0)
                language = lang_ids.unsqueeze(0)
                bert = bert.unsqueeze(0)
                ja_bert = ja_bert.unsqueeze(0)
                sdp_ratio_pt = torch.tensor([sdp_ratio], dtype=torch.float32)
                length_scale_pt = torch.tensor([length_scale], dtype=torch.float32)
                noise_scale_w_pt = torch.tensor([noise_scale_w], dtype=torch.float32)

                y_lengths, x_mask, m_p, logs_p, g, w_ceil = encoder_fpm(
                    x,
                    x_lengths,
                    tone,
                    sid,
                    language,
                    bert,
                    ja_bert,
                    sdp_ratio_pt,
                    length_scale_pt,
                    noise_scale_w_pt,
                )

                y_mask = torch.unsqueeze(
                    torch.arange(MAX_SEQ_LEN * 3) < y_lengths[:, None], dim=1
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
                while (
                    total_dec_seq_len < z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP
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
            elif model._get_name() == f"BertWrapper_{LANGUAGE_MAP[language_]}":
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
            else:
                raise NotImplementedError(model._get_name())

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
