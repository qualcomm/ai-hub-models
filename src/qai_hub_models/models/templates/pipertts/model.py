# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import os
from functools import lru_cache

import torch
from huggingface_hub import hf_hub_download
from piper_train.vits.lightning import VitsModel
from piper_train.vits.models import SynthesizerTrn
from qai_hub.client import Device
from torch import Tensor
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common_voice import CommonVoiceText, TTSLanguage
from qai_hub_models.models.templates.pipertts.model_patch import (
    patch_rational_quadratic_spline,
)
from qai_hub_models.models.templates.pipertts.pipertts_metadata_json import (
    write_pipertts_supplementary_files,
)
from qai_hub_models.models.templates.pipertts.util import (
    build_model_from_onnx,
)
from qai_hub_models.models.templates.voiceai_tts.t5_g2p import (
    T5Decoder,
    T5Encoder,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

SAMPLE_RATE = 22050
DEFAULT_NOISE_SCALE = 0.667
ITALIAN_NOISE_SCALE = 0.8
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_SCALE_W = 0.8
ENCODER_HIDDEN_DIM = 192
MAX_SEQ_LEN = 512
DEC_SEQ_OVERLAP = 12
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_LEN = MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP
UPSAMPLE_FACTOR = 256
UPSAMPLED_MAX_SEQ_LEN = MAX_SEQ_LEN * 3

VOICE_CHECKPOINTS = {
    "paola": "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium/it_IT-paola-medium.onnx",
    "kusal": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/kusal/medium/epoch%3D2652-step%3D1953828.ckpt",
    "thorsten": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/de/de_DE/thorsten/medium/epoch%3D3135-step%3D2702056.ckpt",
    "amy": "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/amy/medium/epoch%3D6679-step%3D1554200.ckpt",
}
VOICE_SPEED = {
    "paola": 0.85,
    "kusal": 1.0,
    "thorsten": 1.075,
    "amy": 1.0,
}


@lru_cache(maxsize=1)
def get_model(voice_id: str) -> SynthesizerTrn:
    patch_rational_quadratic_spline()
    checkpoint_url = VOICE_CHECKPOINTS[voice_id]
    model_path = CachedWebModelAsset(
        checkpoint_url, "pipertts", 1, checkpoint_url.split("/")[-1]
    ).fetch()

    if checkpoint_url.lower().endswith(".onnx"):
        config_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename="it/it_IT/paola/medium/it_IT-paola-medium.onnx.json",
        )
        model_g = build_model_from_onnx(model_path, config_path)
    else:
        model = VitsModel.load_from_checkpoint(
            model_path, map_location="cpu", weights_only=False, dataset=None
        )
        model_g = model.model_g

    model_g.eval()
    return model_g


class Encoder(BaseModel):
    def __init__(self, gen: SynthesizerTrn) -> None:
        super().__init__()
        self.gen = gen

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "x": TensorSpec(shape=(1, MAX_SEQ_LEN), dtype="int32"),
            "x_lengths": TensorSpec(shape=(1,), dtype="int32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "x_encoded": TensorSpec(),
            "m_p": TensorSpec(),
            "logs_p": TensorSpec(),
            "x_mask": TensorSpec(),
        }

    def forward(
        self, x: Tensor, x_lengths: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        x
            the phones of input text, shape of (1, MAX_SEQ_LEN), i.e., [1, 512]
        x_lengths
            the length of phones, shape of [1]

        Returns
        -------
        x_encoded : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        m_p : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        logs_p : Tensor
            shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        x_mask : Tensor
            shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512], mask of x_encoded
        """
        x_encoded, m_p, logs_p, x_mask = self.gen.enc_p(x, x_lengths)
        return x_encoded, m_p, logs_p, x_mask

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_model(PiperTTS.get_voice_id()))

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


class SDP(BaseModel):
    """Wrapper for the Piper encoder and duration predictor with deterministic behavior."""

    def __init__(self, gen: SynthesizerTrn, speed_adjustment: float = 1.0) -> None:
        super().__init__()
        self.gen = gen
        # Generate deterministic noise pattern
        self.register_buffer("sdp_noise_pattern", torch.ones(1, 2, MAX_SEQ_LEN) * 0.5)
        self.scale = 1.0 / speed_adjustment

    def forward(
        self,
        x_encoded: Tensor,
        x_mask: Tensor,
        length_scale: Tensor,
        noise_scale_w: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x_encoded
            encoded hidden representation from the Encoder, shape of (1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), i.e., [1, 192, 512]
        x_mask
            mask of x_encoded, shape of (1, 1, MAX_SEQ_LEN), i.e., [1, 1, 512]
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
        gen = self.gen
        g = None

        if gen.use_sdp:
            dp = gen.dp
            dp_x = dp.pre(torch.detach(x_encoded))

            dp_x = dp.convs(dp_x, x_mask)
            dp_x = dp.proj(dp_x) * x_mask

            sdp_noise = self.sdp_noise_pattern[:, :, : x_encoded.shape[2]]  # type: ignore[index, unused-ignore]
            z = (
                sdp_noise.expand(x_encoded.size(0), -1, -1).to(x_encoded.device)
                * noise_scale_w
            )

            flows = list(reversed(dp.flows))
            flows = [*flows[:-2], flows[-1]]  # remove flows[-2] for speed

            for flow in flows:
                z = flow(z, x_mask, g=dp_x, reverse=True)
            z0, _ = torch.split(z, [1, 1], 1)

            logw = z0
        else:
            logw = gen.dp(x_encoded, x_mask, g=g)

        # This is to add x_encoded in the graph, because other x_encoded are in the if else branch.
        # If remove this line, it will cause the generated onnx eliminating x_encoded as an input.
        logw = logw + x_encoded.sum() * 0
        w = torch.exp(logw + torch.log(self.scale * length_scale)) * x_mask
        w_ceil = torch.ceil(w)
        y_lengths = torch.sum(torch.sum(w_ceil, dim=2), dim=1)
        return y_lengths, w_ceil

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
            "length_scale": TensorSpec(shape=(1,), dtype="float32"),
            "noise_scale_w": TensorSpec(shape=(1,), dtype="float32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "y_lengths": TensorSpec(),
            "w_ceil": TensorSpec(),
        }

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_model(PiperTTS.get_voice_id()))

    @property
    def context_graph_name(self) -> str:
        return "sdp"


class Flow(BaseModel):
    def __init__(self, gen: SynthesizerTrn) -> None:
        super().__init__()
        self.flow = gen.flow
        hidden_channels = gen.hidden_channels
        self.register_buffer(
            "fixed_noise", torch.randn(1, hidden_channels, UPSAMPLED_MAX_SEQ_LEN) * 0.5
        )

    def forward(
        self,
        m_p: Tensor,
        logs_p: Tensor,
        y_mask: Tensor,
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
        attn_squeezed
            shape of (1, UPSAMPLED_MAX_SEQ_LEN, MAX_SEQ_LEN), i.e., [1, 1536, 512]
        noise_scale
            scalar, scale of noise

        Returns
        -------
        z: Tensor
           the output of Flow module, shape of (1, ENCODER_HIDDEN_DIM, UPSAMPLED_MAX_SEQ_LEN), i.e., [1, 192, 1536]
        """
        m_p = torch.matmul(m_p, attn_squeezed.transpose(1, 2))
        logs_p = torch.matmul(logs_p, attn_squeezed.transpose(1, 2))
        z_p = m_p + self.fixed_noise * torch.exp(logs_p) * noise_scale  # type: ignore [operator]
        return self.flow(z_p, y_mask, g=None, reverse=True)

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "m_p": TensorSpec(
                shape=(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), dtype="float32"
            ),
            "logs_p": TensorSpec(
                shape=(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN), dtype="float32"
            ),
            "y_mask": TensorSpec(shape=(1, 1, UPSAMPLED_MAX_SEQ_LEN), dtype="float32"),
            "attn_squeezed": TensorSpec(
                shape=(1, UPSAMPLED_MAX_SEQ_LEN, MAX_SEQ_LEN), dtype="float32"
            ),
            "noise_scale": TensorSpec(shape=(1,), dtype="float32"),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "z": TensorSpec(),
        }

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_model(PiperTTS.get_voice_id()))

    @property
    def context_graph_name(self) -> str:
        return "flow"


class Decoder(BaseModel):
    def __init__(self, gen: SynthesizerTrn) -> None:
        super().__init__()
        self.dec = gen.dec

    def forward(self, z: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z
            shape of (1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN), i.e., [1, 192, 64]

        Returns
        -------
        Tensor
           the synthesized audio clip array
        """
        return self.dec(z, g=None)

    def get_input_spec(self) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit compiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "z": TensorSpec(shape=(1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN), dtype="float32")
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "audio": TensorSpec(),
        }

    @classmethod
    def from_pretrained(cls) -> Self:
        return cls(get_model(PiperTTS.get_voice_id()))

    @property
    def context_graph_name(self) -> str:
        return "decoder"

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime.qairt_version_changes_compilation:
            other_compile_options += " --quantize_io  "
        return super().get_hub_compile_options(
            target_runtime,
            precision,
            other_compile_options,
            device,
            context_graph_name,
        )


class PiperTTS(WorkbenchModelCollection):
    def __init__(
        self,
        encoder: Encoder,
        sdp: SDP,
        flow: Flow,
        decoder: Decoder,
        charsiu_encoder: T5Encoder,
        charsiu_decoder: T5Decoder,
    ) -> None:
        super().__init__(
            {
                "encoder": encoder,
                "sdp": sdp,
                "flow": flow,
                "decoder": decoder,
                "charsiu_encoder": charsiu_encoder,
                "charsiu_decoder": charsiu_decoder,
            }
        )
        self.encoder = encoder
        self.sdp = sdp
        self.flow = flow
        self.decoder = decoder
        self.charsiu_encoder = charsiu_encoder
        self.charsiu_decoder = charsiu_decoder

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CommonVoiceText

    @classmethod
    def get_language(cls) -> TTSLanguage:
        return TTSLanguage.ENGLISH

    @classmethod
    def get_voice_id(cls) -> str:
        raise NotImplementedError(
            "Subclasses of PiperTTS must override get_voice_id() with an "
            "explicit voice identifier (e.g. 'kusal') matching a key in "
            "VOICE_CHECKPOINTS."
        )

    @classmethod
    def get_metadata_name_suffix(cls) -> str:
        return ""

    @classmethod
    def from_pretrained(cls) -> Self:
        model_g = get_model(cls.get_voice_id())
        return cls(
            Encoder(model_g),
            SDP(model_g, VOICE_SPEED[cls.get_voice_id()]),
            Flow(model_g),
            Decoder(model_g),
            T5Encoder.from_pretrained(),
            T5Decoder.from_pretrained(),
        )

    @classmethod
    def write_supplementary_files(
        cls, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_pipertts_supplementary_files(
            cls.get_language(),
            output_dir,
            metadata,
            SAMPLE_RATE,
            cls.get_metadata_name_suffix(),
        )
