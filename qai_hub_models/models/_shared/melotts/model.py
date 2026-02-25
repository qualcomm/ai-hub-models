# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import copy
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from melo import modules
from qai_hub.client import Device
from torch import Tensor
from torch.nn import Module
from transformers import (
    AutoModelForMaskedLM,
    BertForMaskedLM,
    T5ForConditionalGeneration,
)
from transformers.models.t5.modeling_t5 import T5Attention
from typing_extensions import Self

from qai_hub_models.models._shared.melotts.charsiu_model import T5AttentionMod
from qai_hub_models.models._shared.melotts.meloTTS_encoder import (
    OptimizedDurationPredictor,
    OptimizedTextEncoder,
)
from qai_hub_models.models._shared.melotts.meloTTS_flow import OptimizedFlow
from qai_hub_models.models._shared.melotts.utils import (
    BERT_MODEL_IDS,
    download_unidic,
)
from qai_hub_models.models.common import SampleInputsType
from qai_hub_models.utils.base_model import (
    BaseModel,
    Precision,
    TargetRuntime,
)
from qai_hub_models.utils.input_spec import InputSpec

if TYPE_CHECKING:
    from melo.api import TTS

NUM_BLOCKS = 4
MAX_SEQ_LEN = 512
MAX_NUM_INPUT_IDS = 50
BERT_FEATURE_DIM = 1024
ENCODER_HIDDEN_DIM = 192
JA_BERT_FEATURE_DIM = 768
SPEAKER_EMBED_DIM = 256
FLOW_LENGTH_FACTOR = 3
DECODER_Z_TIME_DIM = 64
UPSAMPLED_MAX_SEQ_LEN = MAX_SEQ_LEN * FLOW_LENGTH_FACTOR

MAP2 = {"ENGLISH": "EN_NEWEST", "SPANISH": "ES", "CHINESE": "ZH"}


@lru_cache(maxsize=1)
def get_tts_object(language: str) -> "TTS":
    download_unidic()
    from melo.api import TTS

    return TTS(MAP2[language], device="cpu")


@lru_cache(maxsize=1)
def get_bert_model(language: str) -> BertForMaskedLM:
    return (
        AutoModelForMaskedLM.from_pretrained(BERT_MODEL_IDS[language]).to("cpu").eval()
    )


@lru_cache(maxsize=1)
def get_t5model() -> T5ForConditionalGeneration:
    return T5ForConditionalGeneration.from_pretrained(
        "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"
    ).eval()


class Encoder(BaseModel):
    def __init__(self, tts_object: "TTS", speed_adjustment: float = 0.75) -> None:
        super().__init__()
        self.model = tts_object.model
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
        self.encoder = OptimizedTextEncoder(self.model.enc_p)
        self.dp = OptimizedDurationPredictor(self.model.dp)
        self.speaker_id = next(iter(tts_object.hps.data.spk2id.values()))

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "x": ((1, MAX_SEQ_LEN), "int32"),
            "x_lengths": ((1,), "int32"),
            "tone": ((1, MAX_SEQ_LEN), "int32"),
            "sid": ((1,), "int32"),
            "language": ((1, MAX_SEQ_LEN), "int32"),
            "bert": (
                (1, BERT_FEATURE_DIM, MAX_SEQ_LEN),
                "float32",
            ),
            "ja_bert": (
                (1, JA_BERT_FEATURE_DIM, MAX_SEQ_LEN),
                "float32",
            ),
            "sdp_ratio": ((1,), "float32"),
            "length_scale": ((1,), "float32"),
            "noise_scale_w": ((1,), "float32"),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
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

    @staticmethod
    def get_output_names() -> list[str]:
        return ["y_lengths", "x_mask", "m_p", "logs_p", "g", "w_ceil"]

    def forward(
        self,
        x: Tensor,
        x_lengths: Tensor,
        tone: Tensor,
        sid: Tensor,
        language: Tensor,
        bert: Tensor,
        ja_bert: Tensor,
        sdp_ratio: Tensor,
        length_scale: Tensor,
        noise_scale_w: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        x_mask : Tensor
            shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512], mask of x
        m_p : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        logs_p : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        g : Tensor
            shape of (1, SPEAKER_EMBED_DIM, 1), i.e., [1, 256, 1]
        w_ceil : Tensor
            shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512]
        """
        g = None
        assert callable(self.model.emb_g)
        if self.model.n_speakers > 0:
            # TODO(17781): Undo clamp after we add a tracing option to set the input value range.
            # This does not use a minimum of 0 because some models only have 1 speaker. That would result in a clamp(0, 0) operator, which is invalid in QNN.
            sid = torch.clamp(sid, max=self.model.emb_g.num_embeddings - 1)
            g = self.model.emb_g(sid).unsqueeze(-1)
        x, m_p, logs_p, x_mask = self.encoder.forward(
            x, x_lengths, tone, language, bert, ja_bert, g=g
        )

        logw_sdp = self.sdp_forward(x, x_mask, g, noise_scale_w)
        logw_dp = self.dp(x, x_mask, g=g)

        logw = logw_sdp * sdp_ratio + logw_dp * (1 - sdp_ratio)
        logw = logw.masked_fill(x_mask == 0, -1e9)

        w = torch.exp(logw + torch.log(self.scale * length_scale)) * x_mask
        w_ceil = torch.ceil(w)
        y_lengths = torch.sum(w_ceil, [1, 2])

        return y_lengths, x_mask, m_p, logs_p, g, w_ceil

    def sdp_forward(
        self, x: Tensor, x_mask: Tensor, g: Tensor | None, noise_scale_w: Tensor
    ) -> Tensor:
        """
        Predict the duration of current input clip.

        Parameters
        ----------
        x
            shape of [1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN]
        x_mask
            shape of [1, 1, MAX_SEQ_LEN]
        g
            shape of [1, SPEAKER_EMBED_DIM, 1]
        noise_scale_w
            scalar

        Returns
        -------
        z : Tensor
            shape of (1, 1, MAX_SEQ_LEN)
        """
        sdp = self.model.sdp
        x = torch.detach(x)
        assert hasattr(sdp, "pre") and callable(sdp.pre)
        x = sdp.pre(x)
        assert hasattr(sdp, "cond") and callable(sdp.cond)
        if g is not None:
            g = torch.detach(g)
            x = x + sdp.cond(g)
        assert hasattr(sdp, "convs") and callable(sdp.convs)
        x = sdp.convs(x, x_mask)
        assert hasattr(sdp, "proj") and callable(sdp.proj)
        x = sdp.proj(x) * x_mask

        assert hasattr(sdp, "flows") and isinstance(sdp.flows, Iterable)
        flows = list(sdp.flows)[::-1]
        flows = [
            *flows[:-2],
            flows[-1],
        ]

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
        x0, x1 = torch.split(z, [half_channels, half_channels], dim=1)
        return torch.cat([x0, x1], dim=1) * x_mask

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="encoder",
        )
        # # Must use --truncate_64bit_io when input tensors have type int64.
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors --truncate_64bit_io "
        if target_runtime.qairt_version_changes_compilation:
            compile_options += " --quantize_full_type float16 --quantize_io false "
        return compile_options

    @staticmethod
    def component_precision() -> Precision:
        return Precision.float


class Flow(BaseModel):
    def __init__(self, tts_object: "TTS") -> None:
        super().__init__()
        self.model = tts_object.model
        self.language = tts_object.language
        self.hps = tts_object.hps
        self.symbol_to_id = tts_object.symbol_to_id
        assert isinstance(self.model.hidden_channels, int)
        h = self.model.hidden_channels
        self.fixed_noise = torch.full([1, h, MAX_SEQ_LEN * 3], 0.5)
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
        z_p = m_p + self.fixed_noise * torch.exp(logs_p) * noise_scale
        return self.flow.forward(z_p, y_mask, g, reverse=True)

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "m_p": (
                (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                "float32",
            ),
            "logs_p": (
                (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                "float32",
            ),
            "y_mask": ((1, 1, UPSAMPLED_MAX_SEQ_LEN), "float32"),
            "g": ((1, SPEAKER_EMBED_DIM, 1), "float32"),
            "attn_squeezed": (
                (1, UPSAMPLED_MAX_SEQ_LEN, MAX_SEQ_LEN),
                "float32",
            ),
            "noise_scale": ((1,), "float32"),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None, **kwargs: Any
    ) -> SampleInputsType:
        """
        This is a default implementation that returns a single random data array
        for each input name based on the shapes and dtypes in `get_input_spec`.

        A subclass may choose to override this and fetch a batch of real input data
        from a data source.

        See the `sample_inputs` doc for the expected format.
        """
        rng = np.random.default_rng(seed=123)
        specs = input_spec or type(self).get_input_spec()
        return {
            name: [rng.normal(size=shape).astype(dtype)]
            for name, (shape, dtype) in specs.items()
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["z"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="flow",
        )
        if target_runtime.qairt_version_changes_compilation:
            compile_options += " --quantize_io false "
        return compile_options

    @staticmethod
    def calibration_dataset_name() -> str:
        return "common_voice_text"

    @staticmethod
    def component_precision() -> Precision:
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

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "z": (
                (1, ENCODER_HIDDEN_DIM, DECODER_Z_TIME_DIM),
                "float32",
            ),
            "g": ((1, SPEAKER_EMBED_DIM, 1), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["audio"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="decoder",
        )
        if target_runtime.qairt_version_changes_compilation:
            compile_options += " --quantize_io false "
        return compile_options

    @staticmethod
    def calibration_dataset_name() -> str:
        return "common_voice_text"

    @staticmethod
    def component_precision() -> Precision:
        return Precision.w8a16


class T5Encoder(BaseModel):
    def __init__(self, t5model: T5ForConditionalGeneration) -> None:
        super().__init__()
        self.model = t5model
        self.embed_tokens = t5model.encoder.embed_tokens
        self.block = t5model.encoder.block
        self.final_layer_norm = t5model.encoder.final_layer_norm

    def forward(
        self, input_ids: Tensor, encoder_attention_mask: Tensor
    ) -> list[Tensor]:
        """
        Parameters
        ----------
        input_ids
            shape of (1, MAX_NUM_INPUT_IDS)
        encoder_attention_mask
            shape of (1, MAX_NUM_INPUT_IDS)

        Returns
        -------
        list[Tensor]
           a list of key value states
        """
        input_embeds = self.embed_tokens(input_ids)
        extended_attention_mask = -10000.0 * (1 - encoder_attention_mask).unsqueeze(
            0
        ).unsqueeze(0)
        position_bias = None
        hidden_states = input_embeds

        for layer_module in self.block:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
            )
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

        hidden_states = self.final_layer_norm(hidden_states)

        outputs = []
        assert isinstance(self.model.decoder, Module) and isinstance(
            self.model.decoder.block, Iterable
        )
        for decoder_block in self.model.decoder.block:
            cross_attn = decoder_block.layer[1].EncDecAttention
            key_states = cross_attn.k(hidden_states)
            key_states = key_states.view(
                key_states.shape[0],
                -1,
                cross_attn.n_heads,
                cross_attn.key_value_proj_dim,
            ).transpose(1, 2)
            value_states = cross_attn.v(hidden_states)
            value_states = value_states.view(
                value_states.shape[0],
                -1,
                cross_attn.n_heads,
                cross_attn.key_value_proj_dim,
            ).transpose(1, 2)
            outputs.append(key_states)
            outputs.append(value_states)

        return outputs

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "input_ids": ((1, MAX_NUM_INPUT_IDS), "int32"),
            "encoder_attention_mask": ((1, MAX_NUM_INPUT_IDS), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return [
            "block_0_cross_key_states",
            "block_0_cross_value_states",
            "block_1_cross_key_states",
            "block_1_cross_value_states",
            "block_2_cross_key_states",
            "block_2_cross_value_states",
            "block_3_cross_key_states",
            "block_3_cross_value_states",
        ]

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_t5model())

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="charsiu_encoder",
        )
        if target_runtime.qairt_version_changes_compilation:
            compile_options += " --quantize_io false "
        return compile_options

    @staticmethod
    def component_precision() -> Precision:
        return Precision.float


def replace_submodules(module: Module) -> None:
    for name, submodule in module.named_children():
        if isinstance(submodule, T5Attention):
            setattr(module, name, T5AttentionMod(submodule))
        else:
            replace_submodules(submodule)


class T5Decoder(BaseModel):
    def __init__(
        self, t5model: T5ForConditionalGeneration, max_num_input_ids: int
    ) -> None:
        super().__init__()
        self.model = t5model
        self.embed_tokens = t5model.decoder.embed_tokens
        self.block = t5model.decoder.block
        self.final_layer_norm = t5model.decoder.final_layer_norm
        self.max_num_input_ids = max_num_input_ids

        n_heads = self.block[0].layer[0].SelfAttention.n_heads
        position_bias_len = n_heads * max_num_input_ids
        self.position_bias_embedding = torch.nn.Embedding(
            max_num_input_ids, position_bias_len
        )
        position_bias_weight = torch.zeros(
            [max_num_input_ids, position_bias_len], dtype=torch.float32
        )
        for idx in range(max_num_input_ids):
            position_bias = self.compute_position_bias(idx + 1, max_num_input_ids)
            position_bias_weight[idx, :] = position_bias.flatten()
        self.position_bias_embedding.weight = torch.nn.Parameter(position_bias_weight)

        replace_submodules(self.block)

    def compute_position_bias(self, key_length: int, max_length: int) -> Tensor:
        attn = self.block[0].layer[0].SelfAttention
        position_bias = attn.compute_bias(key_length, key_length)[:, :, -1:, :]
        position_bias_masked = torch.full(
            [
                position_bias.shape[0],
                position_bias.shape[1],
                position_bias.shape[2],
                max_length,
            ],
            -10000.0,
            dtype=torch.float32,
        )
        position_bias_masked[..., : key_length - 1] = position_bias[
            ..., : key_length - 1
        ]
        position_bias_masked[..., -1] = position_bias[..., -1]
        return position_bias_masked

    def forward(
        self,
        input_ids: Tensor,
        encoder_attention_mask: Tensor,
        position: Tensor,
        *past_key_values: Tensor,
    ) -> tuple[Tensor, ...]:
        """
        Parameters
        ----------
        input_ids
            shape of (1, 1)
        encoder_attention_mask
            shape of (1, q_len)
        position
            shape of (1, 1)
        *past_key_values
            a list of previous key-value states, each state is shape of
            (batch_size, n_heads, q_len - 1, dim_per_head) or (batch_size, n_heads, q_len, dim_per_head)

        Returns
        -------
        output : tuple[Tensor, ...]
            logits
                predicted logits
            present_key_values
                updated key values
        """
        input_embeds = self.embed_tokens(input_ids)
        encoder_extended_attention_mask = -10000.0 * (
            1 - encoder_attention_mask
        ).unsqueeze(0).unsqueeze(0)
        encoder_decoder_position_bias = encoder_extended_attention_mask
        hidden_states = input_embeds
        position_bias = None
        # present_key_value_states = ()
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        present_key_values = []

        position_bias = self.position_bias_embedding(position).view(
            1, self.block[0].layer[0].SelfAttention.n_heads, 1, -1
        )

        for i, layer_module in enumerate(self.block):
            past_key_value = []
            past_key_value.append(past_key_values[4 * i])
            past_key_value.append(past_key_values[4 * i + 1])
            past_key_value.append(past_key_values[4 * i + 2])
            past_key_value.append(past_key_values[4 * i + 3])
            layer_outputs = layer_module(
                hidden_states,
                position_bias=position_bias,
                encoder_hidden_states=True,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=past_key_value,
                use_cache=True,
                output_attentions=False,
            )
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            # Return just new key and value projection for self attn
            present_key_values.append(present_key_value_state[0])
            present_key_values.append(present_key_value_state[1])

        hidden_states = self.final_layer_norm(hidden_states)

        logits = self.model.lm_head(hidden_states)  # type: ignore[operator]
        return logits, *present_key_values

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        n_heads = 6
        q_len = 50
        dim_per_head = 64
        return {
            "input_ids": ((1, 1), "int32"),
            "encoder_attention_mask": ((1, q_len), "int32"),
            "position": ((1, 1), "int32"),
            "block_0_past_self_key_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_0_past_self_value_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_0_cross_key_states": ((1, n_heads, q_len, dim_per_head), "float32"),
            "block_0_cross_value_states": (
                (1, n_heads, q_len, dim_per_head),
                "float32",
            ),
            "block_1_past_self_key_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_1_past_self_value_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_1_cross_key_states": ((1, n_heads, q_len, dim_per_head), "float32"),
            "block_1_cross_value_states": (
                (1, n_heads, q_len, dim_per_head),
                "float32",
            ),
            "block_2_past_self_key_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_2_past_self_value_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_2_cross_key_states": ((1, n_heads, q_len, dim_per_head), "float32"),
            "block_2_cross_value_states": (
                (1, n_heads, q_len, dim_per_head),
                "float32",
            ),
            "block_3_past_self_key_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_3_past_self_value_states": (
                (1, n_heads, q_len - 1, dim_per_head),
                "float32",
            ),
            "block_3_cross_key_states": ((1, n_heads, q_len, dim_per_head), "float32"),
            "block_3_cross_value_states": (
                (1, n_heads, q_len, dim_per_head),
                "float32",
            ),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return [
            "logits",
            "block_0_present_self_key_states",
            "block_0_present_self_value_states",
            "block_1_present_self_key_states",
            "block_1_present_self_value_states",
            "block_2_present_self_key_states",
            "block_2_present_self_value_states",
            "block_3_present_self_key_states",
            "block_3_present_self_value_states",
        ]

    @classmethod
    def from_pretrained(cls) -> Self:
        # here the t5model is passed to T5Decoder by reference
        # use deepcopy to prevent cached t5model being modified, so the cache can be reused
        return cls(copy.deepcopy(get_t5model()), MAX_NUM_INPUT_IDS)

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="charsiu_decoder",
        )
        if target_runtime.qairt_version_changes_compilation:
            compile_options += " --quantize_io false "
        return compile_options

    @staticmethod
    def component_precision() -> Precision:
        return Precision.float


class BertWrapper(BaseModel):
    def __init__(self, bert_model: BertForMaskedLM) -> None:
        super().__init__()
        self.model: BertForMaskedLM = bert_model
        self.bert = self.model.bert
        assert isinstance(self.model.bert, Module)
        self.embeddings = self.model.bert.embeddings
        self.encoder = self.model.bert.encoder

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, token_type_ids: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        input_ids
            shape of (1, 200)
        attention_mask
            shape of (1, 200)
        token_type_ids
            shape of (1, 200)

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

    @staticmethod
    def get_input_spec() -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "input_ids": ((1, 200), "int32"),
            "attention_mask": ((1, 200), "int32"),
            "token_type_ids": ((1, 200), "int32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["hidden_states"]

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        compile_options = super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name="bert",
        )
        if target_runtime != TargetRuntime.ONNX:
            compile_options += " --truncate_64bit_tensors --truncate_64bit_io "
        if target_runtime.qairt_version_changes_compilation:
            compile_options += " --quantize_io false "
        return compile_options

    @staticmethod
    def calibration_dataset_name() -> str:
        return "common_voice_text"

    @staticmethod
    def component_precision() -> Precision:
        return Precision.w16a16
