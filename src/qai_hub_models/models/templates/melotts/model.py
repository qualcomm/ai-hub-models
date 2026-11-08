# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import math
import os
from abc import abstractmethod
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from melo import modules
from qai_hub.client import Device
from torch import Tensor
from torch.nn import Module
from transformers import (
    AutoModelForMaskedLM,
    BertForMaskedLM,
)
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    SampleInputsType,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common_voice import (
    LANG_CODE_MAP,
    CommonVoiceText,
    TTSLanguage,
)
from qai_hub_models.models.templates.common import replace_module_recursively
from qai_hub_models.models.templates.melotts.meloTTS_encoder import (
    FFNMod,
    OptimizedDurationPredictor,
    OptimizedTextEncoder,
)
from qai_hub_models.models.templates.melotts.meloTTS_flow import OptimizedFlow
from qai_hub_models.models.templates.melotts.utils import (
    download_unidic,
    write_melotts_supplementary_files,
)
from qai_hub_models.models.templates.voiceai_tts.language import BERT_MODEL_IDS
from qai_hub_models.models.templates.voiceai_tts.t5_g2p import (
    T5Decoder as _T5DecoderBase,
)
from qai_hub_models.models.templates.voiceai_tts.t5_g2p import (
    T5Encoder as _T5EncoderBase,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

if TYPE_CHECKING:
    from melo.api import TTS

SAMPLE_RATE = 44100
MAX_SEQ_LEN = 512
BERT_FEATURE_DIM = 1024
ENCODER_HIDDEN_DIM = 192
JA_BERT_FEATURE_DIM = 768
SPEAKER_EMBED_DIM = 256
FLOW_LENGTH_FACTOR = 3
DECODER_Z_TIME_DIM = 64
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_OVERLAP = 12
UPSAMPLE_FACTOR = 512
MAX_BERT_TOKENS = 200
UPSAMPLED_MAX_SEQ_LEN = MAX_SEQ_LEN * FLOW_LENGTH_FACTOR
LANG_CODE_TO_TTS_CODE = {"KO": "KR", "JA": "JP"}


@lru_cache(maxsize=1)
def get_tts_object(language: TTSLanguage) -> "TTS":
    download_unidic()

    import melo
    from melo.api import TTS

    lang_code = LANG_CODE_MAP[language]
    tts = TTS(LANG_CODE_TO_TTS_CODE.get(lang_code, lang_code), device="cpu")

    # Monkeypatch melo.attentions.FFN to replace torch.relu with torch.maximum.
    # This avoids the Conv2D+Relu op fusion bug in Qairt, which incorrectly converts
    # Conv2D to w8fp16 (int8 weight, fp16 activation) — unsupported on HTP.
    #
    # Can be removed when JIRA AISW-177186 is resolved.
    replace_module_recursively(tts.model, melo.attentions.FFN, FFNMod)

    return tts


class Encoder(BaseModel):
    def __init__(self, tts_object: "TTS", speed_adjustment: float = 1.0) -> None:
        super().__init__()
        self.model = tts_object.model
        self.language = tts_object.language
        self.hps = tts_object.hps
        self.symbol_to_id = tts_object.symbol_to_id
        self.sid = torch.tensor([0], dtype=torch.long)
        self.sdp_noise = torch.full((1, 2, MAX_SEQ_LEN), 0.5)
        self.length_scale = torch.tensor([1.0], dtype=torch.float)
        self.scale = self.length_scale / speed_adjustment
        self.register_buffer(
            "ones_triangular",
            torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), diagonal=0),
        )
        self.register_buffer(
            "indices", torch.arange(MAX_SEQ_LEN * 4, dtype=torch.float32)[None, None, :]
        )
        self.upsample_factor = UPSAMPLE_FACTOR
        self.encoder = OptimizedTextEncoder(self.model.enc_p)
        self.speaker_id = next(iter(tts_object.hps.data.spk2id.values()))

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "x": TensorSpec(shape=(1, MAX_SEQ_LEN), dtype="int32"),
            "x_lengths": TensorSpec(shape=(1,), dtype="int32"),
            "tone": TensorSpec(shape=(1, MAX_SEQ_LEN), dtype="int32"),
            "sid": TensorSpec(shape=(1,), dtype="int32"),
            "language": TensorSpec(shape=(1, MAX_SEQ_LEN), dtype="int32"),
            "bert": TensorSpec(
                shape=(1, BERT_FEATURE_DIM, MAX_SEQ_LEN),
                dtype="float32",
            ),
            "ja_bert": TensorSpec(
                shape=(1, JA_BERT_FEATURE_DIM, MAX_SEQ_LEN),
                dtype="float32",
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        This function is used for inference.
        """
        input_spec = self.get_input_spec()
        type_dic = dict(int64=torch.int64, int32=torch.int32, float32=torch.float32)
        inputs_list = [
            torch.zeros(sp[0], dtype=type_dic[sp[1]]) for sp in input_spec.values()
        ]
        dic = {}
        for i, input_name in enumerate(input_spec.keys()):
            dic[input_name] = [inputs_list[i].numpy()]
        return dic

    def get_output_spec(self) -> OutputSpec:
        return {
            "x_encoded": TensorSpec(),
            "m_p": TensorSpec(),
            "logs_p": TensorSpec(),
            "x_mask": TensorSpec(),
            "g": TensorSpec(),
        }

    def forward(
        self,
        x: Tensor,
        x_lengths: Tensor,
        tone: Tensor,
        sid: Tensor,
        language: Tensor,
        bert: Tensor,
        ja_bert: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """
        Process the phones and tone of the input text, use bert model to tokenize the text

        Parameters
        ----------
        x
            the phones of input text, shape of (1, MAX_SEQ_LEN), i.e., [1, 512]
        x_lengths
            the length of phones, shape of [1]
        tone
            the tone of input text, shape of (1, MAX_SEQ_LEN), i.e., [1, 512]
        sid
            speaker ID, scalar
        language
            shape of (1, MAX_SEQ_LEN), i.e., [1, 512]
        bert
            shape of (1, BERT_FEATURE_DIM, MAX_SEQ_LEN), i.e., [1, 1024, 512]
        ja_bert
            shape of (1, JA_BERT_FEATURE_DIM, MAX_SEQ_LEN), i.e., [1, 768, 512]

        Returns
        -------
        x_encoded : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN)
        m_p : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        logs_p : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        x_mask : Tensor
            shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512], mask of x
        g : Tensor | None
            shape of (1, SPEAKER_EMBED_DIM, 1), i.e., [1, 256, 1]
        """
        g = None
        assert callable(self.model.emb_g)
        if self.model.n_speakers > 0:  # type: ignore[operator, unused-ignore]
            sid = torch.clamp(sid, max=self.model.emb_g.num_embeddings - 1)
            g = self.model.emb_g(sid).unsqueeze(-1)

        x_encoded, m_p, logs_p, x_mask = self.encoder.forward(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )
        return x_encoded, m_p, logs_p, x_mask, g

    @property
    def context_graph_name(self) -> str:
        return "encoder"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime != TargetRuntime.ONNX:
            other_compile_options += " --truncate_64bit_tensors --truncate_64bit_io "
        return super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )

    def component_precision(self) -> Precision:
        return Precision.float


class SDP(BaseModel):
    def __init__(self, tts_object: "TTS", speed_adjustment: float = 1.0) -> None:
        super().__init__()
        self.model = tts_object.model
        self.language = tts_object.language
        self.hps = tts_object.hps
        self.symbol_to_id = tts_object.symbol_to_id
        self.sid = torch.tensor([0], dtype=torch.long)
        self.sdp_noise = torch.full((1, 2, MAX_SEQ_LEN), 0.5)
        self.length_scale = torch.tensor([1.0], dtype=torch.float)
        self.scale = self.length_scale / speed_adjustment
        self.register_buffer(
            "ones_triangular",
            torch.triu(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN), diagonal=0),
        )
        self.register_buffer(
            "indices", torch.arange(MAX_SEQ_LEN * 4, dtype=torch.float32)[None, None, :]
        )
        self.upsample_factor = 512
        self.dp = OptimizedDurationPredictor(self.model.dp)

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        g: Tensor,
        sdp_ratio: Tensor,
        length_scale: Tensor,
        noise_scale_w: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Predict the duration of audio.

        Parameters
        ----------
        x
            encoded hidden states of input_ids, shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN)
        x_mask
            mask of x, shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512]
        g
            embedding of speaker ID, shape of (1, SPEAKER_EMBED_DIM, 1), i.e., [1, 256, 1]
        sdp_ratio
            scalar, ratio of duration predictor
        length_scale
            scalar, scale of length
        noise_scale_w
            scalar, scale of noise

        Returns
        -------
        y_lengths : Tensor
            shape of [1]
        w_ceil : Tensor
            shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512]
        """
        assert callable(self.dp)
        # logw_sdp = self.sdp_forward(x, x_mask, g, noise_scale_w)
        logw_dp = self.dp(x, x_mask, g=g)

        # Dummy usage to force ONNX to include sdp_ratio as MeloTTS is currently using only duration predictor (DP) not SDP.
        dummy = sdp_ratio * 0.0 + noise_scale_w * 0.0
        logw_dp = logw_dp + dummy

        logw2 = logw_dp.clone()

        # for j in range(logw_dp.shape[2]):
        #     if x_mask[0, 0, j].item() == 0:
        #         logw2[0, 0, j] = -1e9

        w = torch.exp(logw2 + torch.log(self.scale * length_scale)) * x_mask
        w_ceil = torch.ceil(w)

        # y_lengths = torch.sum(w_ceil, [1, 2])
        y_lengths = torch.sum(w_ceil).unsqueeze(0)  # shape: [1, 1, 512]
        return y_lengths, w_ceil

    def sdp_forward(
        self, x: Tensor, x_mask: Tensor, g: Tensor, noise_scale_w: Tensor
    ) -> Tensor:
        sdp = self.model.sdp
        # assert callable(sdp)
        assert hasattr(sdp, "pre") and callable(sdp.pre)
        assert hasattr(sdp, "cond") and callable(sdp.cond)
        assert hasattr(sdp, "convs") and callable(sdp.convs)
        assert hasattr(sdp, "proj") and callable(sdp.proj)
        assert hasattr(sdp, "flows") and isinstance(sdp.flows, Iterable)
        x = torch.detach(x)
        x = sdp.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + sdp.cond(g)
        x = sdp.convs(x, x_mask)
        x = sdp.proj(x) * x_mask

        flows = list(sdp.flows)[::-1]
        flows = [*flows[:-2], flows[-1]]  # remove the useless vflow
        z = self.sdp_noise[:, :, : x.size(2)] * noise_scale_w

        half_channels = None
        for flow in flows:
            if isinstance(flow, modules.ConvFlow):
                z = self.conv_flow_reverse(flow, z, x_mask, x)
                half_channels = flow.half_channels
            elif isinstance(flow, modules.Flip):
                z = torch.flip(z, [1])
            elif isinstance(flow, modules.ElementwiseAffine):
                z = flow(z, x_mask, reverse=True)
            else:
                raise TypeError(f"Unexpected flow type: {type(flow)}")

        if half_channels is not None:
            z = z[:, :half_channels, :]
        else:
            z = z[:, : z.size(1) // 2, :]

        return z

    def conv_flow_reverse(
        self, flow: modules.ConvFlow, z: Tensor, x_mask: Tensor, x: Tensor
    ) -> Tensor:
        half_channels = flow.half_channels

        # Split z into x0 and x1
        x0, x1 = torch.split(z, [half_channels, half_channels], dim=1)

        h = flow.pre(x0)
        h = flow.convs(h, x_mask)
        h = flow.proj(h) * x_mask

        b, _c_h, t = h.shape
        h = h.reshape(b, half_channels, -1, t).permute(0, 1, 3, 2)  # [b, c, t, *]
        unnormalized_widths = h[..., : flow.num_bins] / math.sqrt(flow.filter_channels)
        unnormalized_heights = h[..., flow.num_bins : 2 * flow.num_bins] / math.sqrt(
            flow.filter_channels
        )
        unnormalized_derivatives = h[..., 2 * flow.num_bins :]

        x1_transformed = self.spline(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=True,
        )

        return torch.cat([x0, x1_transformed], dim=1) * x_mask

    def spline(
        self,
        inputs: Tensor,
        unnormalized_widths: Tensor,
        unnormalized_heights: Tensor,
        unnormalized_derivatives: Tensor,
        inverse: bool = False,
        tail_bound: float = 5.0,
    ) -> Tensor:
        num_bins = unnormalized_widths.shape[-1]

        widths = F.softmax(unnormalized_widths, dim=-1)
        heights = F.softmax(unnormalized_heights, dim=-1)

        triu_mask = torch.triu(torch.ones(num_bins, num_bins, device=widths.device))
        cumwidths = torch.matmul(widths.unsqueeze(-2), triu_mask).squeeze(-2)
        cumheights = torch.matmul(heights.unsqueeze(-2), triu_mask).squeeze(-2)

        cumwidths = (2 * tail_bound) * cumwidths - tail_bound
        cumheights = (2 * tail_bound) * cumheights - tail_bound

        cumwidths = torch.cat(
            [
                torch.full_like(cumwidths[..., :1], -tail_bound),
                cumwidths,
                torch.full_like(cumwidths[..., :1], tail_bound),
            ],
            dim=-1,
        )
        cumheights = torch.cat(
            [
                torch.full_like(cumheights[..., :1], -tail_bound),
                cumheights,
                torch.full_like(cumheights[..., :1], tail_bound),
            ],
            dim=-1,
        )

        if inverse:
            bin_idx = self.searchsorted(cumheights, inputs)
            bin_idx = torch.clamp(bin_idx, 0, num_bins - 2)
            bin_idx = bin_idx.unsqueeze(-1)

            input_cum_lower = torch.gather(cumheights, -1, bin_idx).squeeze(-1)
            input_cum_upper = torch.gather(
                cumheights, -1, torch.clamp(bin_idx + 1, max=cumheights.shape[-1] - 1)
            ).squeeze(-1)
            input_width_lower = torch.gather(cumwidths, -1, bin_idx).squeeze(-1)
            input_width_upper = torch.gather(
                cumwidths, -1, torch.clamp(bin_idx + 1, max=cumwidths.shape[-1] - 1)
            ).squeeze(-1)

            t = (inputs - input_cum_lower) / (input_cum_upper - input_cum_lower + 1e-7)
            outputs = input_width_lower + t * (input_width_upper - input_width_lower)

        else:
            bin_idx = self.searchsorted(cumwidths, inputs)
            bin_idx = torch.clamp(bin_idx, 0, num_bins - 2)
            bin_idx = bin_idx.unsqueeze(-1)

            input_cum_lower = torch.gather(cumwidths, -1, bin_idx).squeeze(-1)
            input_cum_upper = torch.gather(
                cumwidths, -1, torch.clamp(bin_idx + 1, max=cumwidths.shape[-1] - 1)
            ).squeeze(-1)
            input_height_lower = torch.gather(cumheights, -1, bin_idx).squeeze(-1)
            input_height_upper = torch.gather(
                cumheights, -1, torch.clamp(bin_idx + 1, max=cumheights.shape[-1] - 1)
            ).squeeze(-1)

            t = (inputs - input_cum_lower) / (input_cum_upper - input_cum_lower + 1e-7)
            outputs = input_height_lower + t * (input_height_upper - input_height_lower)

        return outputs

    def searchsorted(self, bin_locations: Tensor, inputs: Tensor) -> Tensor:
        return torch.sum(inputs.unsqueeze(-1) >= bin_locations, dim=-1) - 2

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "x_encoded": TensorSpec(
                shape=(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), dtype="float32"
            ),
            "x_mask": TensorSpec(shape=(1, 1, MAX_SEQ_LEN), dtype="float32"),
            "g": TensorSpec(shape=(1, SPEAKER_EMBED_DIM, 1), dtype="float32"),
            "sdp_ratio": TensorSpec(shape=(1,), dtype="float32"),
            "length_scale": TensorSpec(shape=(1,), dtype="float32"),
            "noise_scale_w": TensorSpec(shape=(1,), dtype="float32"),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        See the `sample_inputs` doc for the expected format.
        """
        rng = np.random.default_rng(seed=123)
        specs = input_spec or self.get_input_spec()
        return {
            name: [rng.normal(size=shape).astype(dtype)]
            for name, (shape, dtype) in specs.items()
        }

    def get_output_spec(self) -> OutputSpec:
        return {"y_lengths": TensorSpec(), "w_ceil": TensorSpec()}

    @property
    def context_graph_name(self) -> str:
        return "sdp"

    def component_precision(self) -> Precision:
        return Precision.float


class Flow(BaseModel):
    def __init__(self, tts_object: "TTS") -> None:
        super().__init__()
        self.model = tts_object.model
        self.language = tts_object.language
        self.hps = tts_object.hps
        self.symbol_to_id = tts_object.symbol_to_id
        assert isinstance(self.model.inter_channels, int)
        gen = torch.Generator()
        gen.manual_seed(0)
        self.register_buffer(
            "fixed_noise",
            torch.randn(1, self.model.inter_channels, MAX_SEQ_LEN * 3, generator=gen),
        )
        self.flow = OptimizedFlow(self.model.flow)

    def forward(
        self,
        m_p: Tensor,
        logs_p: Tensor,
        y_mask: Tensor,
        g: Tensor,
        attn_squeezed: Tensor,
        noise_scale: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        m_p
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        logs_p
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        y_mask
            shape of (1, 1, UPSAMPLED_MAX_SEQ_LEN), i.e., [1, 1, 1536]
        g
            embedding of speaker ID
            shape of (1, SPEAKER_EMBED_DIM, 1), i.e., [1, 256, 1]
        attn_squeezed
            shape of (1, UPSAMPLED_MAX_SEQ_LEN, MAX_SEQ_LEN), i.e., [1, 1536, 512]
        noise_scale
            scalar

        Returns
        -------
        output : Tensor
           the output of Flow module, shape of (1, ENCODER_HIDDEN_DIM, UPSAMPLED_MAX_SEQ_LEN), i.e., [1, 192, 1536]
        """
        m_p = torch.matmul(m_p, attn_squeezed.transpose(1, 2))
        logs_p = torch.matmul(logs_p, attn_squeezed.transpose(1, 2))
        seq_len = m_p.size(2)
        assert isinstance(self.fixed_noise, Tensor)
        noise = self.fixed_noise[:, : m_p.size(1), :seq_len]
        z_p = m_p + noise * torch.exp(logs_p) * noise_scale
        return self.flow.forward(z_p, y_mask, g, reverse=True)

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "m_p": TensorSpec(
                shape=(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                dtype="float32",
            ),
            "logs_p": TensorSpec(
                shape=(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                dtype="float32",
            ),
            "y_mask": TensorSpec(
                shape=(1, 1, UPSAMPLED_MAX_SEQ_LEN),
                dtype="float32",
            ),
            "g": TensorSpec(shape=(1, SPEAKER_EMBED_DIM, 1), dtype="float32"),
            "attn_squeezed": TensorSpec(
                shape=(1, UPSAMPLED_MAX_SEQ_LEN, MAX_SEQ_LEN),
                dtype="float32",
            ),
            "noise_scale": TensorSpec(shape=(1,), dtype="float32"),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, *args: Any, **kwargs: Any
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        See the `sample_inputs` doc for the expected format.
        """
        rng = np.random.default_rng(seed=123)
        specs = input_spec or self.get_input_spec()
        return {
            name: [rng.normal(size=shape).astype(dtype)]
            for name, (shape, dtype) in specs.items()
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "z": TensorSpec(),
        }

    @property
    def context_graph_name(self) -> str:
        return "flow"

    def component_precision(self) -> Precision:
        return Precision.w8a16


class Decoder(BaseModel):
    def __init__(self, tts_object: "TTS") -> None:
        super().__init__()
        self.model = tts_object.model

    def forward(self, z: Tensor, g: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z
            shape of (1, ENCODER_HIDDEN_DIM, DECODER_Z_TIME_DIM), i.e., [1, 192, 64]
        g
            shape of (1, SPEAKER_EMBED_DIM, 1), i.e., [1, 256, 1]

        Returns
        -------
        Tensor
           the synthesized audio clip array
        """
        assert callable(self.model.dec)
        return self.model.dec(z, g=g)

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "z": TensorSpec(
                shape=(1, ENCODER_HIDDEN_DIM, DECODER_Z_TIME_DIM),
                dtype="float32",
            ),
            "g": TensorSpec(shape=(1, SPEAKER_EMBED_DIM, 1), dtype="float32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "audio": TensorSpec(),
        }

    @property
    def context_graph_name(self) -> str:
        return "decoder"

    def component_precision(self) -> Precision:
        return Precision.w8a16


class T5Encoder(_T5EncoderBase):
    @property
    def context_graph_name(self) -> str:
        return "charsiu_encoder"

    def component_precision(self) -> Precision:
        return Precision.float


class T5Decoder(_T5DecoderBase):
    @property
    def context_graph_name(self) -> str:
        return "charsiu_decoder"

    def component_precision(self) -> Precision:
        return Precision.float


class BertWrapper(BaseModel):
    # The TTS language each concrete subclass loads its BERT model for.
    language: TTSLanguage

    def __init__(self, bert_model: BertForMaskedLM) -> None:
        super().__init__()
        self.model: BertForMaskedLM = bert_model
        self.bert = self.model.bert
        assert isinstance(self.model.bert, Module)
        self.embeddings = self.model.bert.embeddings
        self.encoder = self.model.bert.encoder

    @classmethod
    def from_pretrained(cls, language: TTSLanguage | None = None) -> Self:
        bert_model: BertForMaskedLM = (
            AutoModelForMaskedLM.from_pretrained(
                BERT_MODEL_IDS[language or cls.language]
            )
            .to("cpu")
            .eval()
        )
        return cls(bert_model)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        input_ids
            shape of (1, MAX_BERT_TOKENS)
        attention_mask
            shape of (1, MAX_BERT_TOKENS)
        token_type_ids
            shape of (1, MAX_BERT_TOKENS)

        Returns
        -------
        Tensor
           the last hidden states
        """
        embedding_output = self.embeddings(
            # TODO(17781): Undo clamp after we add a profile job option to set the input value range.
            input_ids=torch.clamp(
                input_ids,
                0,
                self.embeddings.word_embeddings.num_embeddings - 1,  # type: ignore[union-attr, arg-type, operator, unused-ignore]
            ),
            position_ids=None,
            # TODO(17781): Undo clamp after we add a profile job option to set the input value range.
            token_type_ids=torch.clamp(
                token_type_ids,
                0,
                self.embeddings.token_type_embeddings.num_embeddings - 1,  # type: ignore[union-attr, arg-type, operator, unused-ignore]
            ),
            inputs_embeds=None,
            past_key_values_length=0,
        )
        extended_attention_mask = -100.0 * (1 - attention_mask)
        encoder_outputs = self.encoder(  # type: ignore[operator]
            embedding_output,
            attention_mask=extended_attention_mask,
            output_hidden_states=True,
        )
        return encoder_outputs.hidden_states[-3]

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "input_ids": TensorSpec(shape=(1, MAX_BERT_TOKENS), dtype="int32"),
            "attention_mask": TensorSpec(shape=(1, MAX_BERT_TOKENS), dtype="int32"),
            "token_type_ids": TensorSpec(shape=(1, MAX_BERT_TOKENS), dtype="int32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "hidden_states": TensorSpec(),
        }

    @property
    def context_graph_name(self) -> str:
        return "bert"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime != TargetRuntime.ONNX:
            other_compile_options += " --truncate_64bit_tensors --truncate_64bit_io "
        return super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )

    def component_precision(self) -> Precision:
        return Precision.float


class MeloTTS(WorkbenchModelCollection):
    def __init__(
        self,
        encoder: Encoder,
        sdp: SDP,
        flow: Flow,
        decoder: Decoder,
        **extra_components: BaseModel,
    ) -> None:
        super().__init__(
            {
                "encoder": encoder,
                "sdp": sdp,
                "flow": flow,
                "decoder": decoder,
                **extra_components,
            }
        )
        self.encoder = encoder
        self.sdp = sdp
        self.flow = flow
        self.decoder = decoder
        self.speaker_id = encoder.speaker_id
        self.tts_object = get_tts_object(self.get_language())

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CommonVoiceText

    @classmethod
    @abstractmethod
    def get_language(cls) -> TTSLanguage:
        pass

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_melotts_supplementary_files(
            self.get_language(), output_dir, metadata, SAMPLE_RATE
        )
