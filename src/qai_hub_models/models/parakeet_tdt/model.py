# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

# mypy: warn-unused-ignores=false
import numpy as np
import torch
from qai_hub.client import Device
from torch import nn
from transformers import AutoModelForTDT  # type: ignore[attr-defined]
from typing_extensions import Self

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models.parakeet_tdt.dataset import LibriSpeechDataset
from qai_hub_models.models.parakeet_tdt.evaluator import WEREvaluator
from qai_hub_models.models.templates.parakeet_tdt.parakeet_tdt_metadata_json import (
    write_parakeet_tdt_supplementary_files,
)
from qai_hub_models.utils.base_collection_model import WorkbenchModelCollection
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
HF_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

# Audio constants (from processor_config.json)
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128

# Model dimension constants (from config.json)
ENCODER_HIDDEN_SIZE = 1024
JOINT_HIDDEN_SIZE = 640
DECODER_HIDDEN_SIZE = 640
NUM_DECODER_LSTM_LAYERS = 2
VOCAB_SIZE = 8193  # 8192 BPE tokens + 1 blank
NUM_DURATIONS = 5  # durations [0, 1, 2, 3, 4]
JOINT_OUTPUT_SIZE = VOCAB_SIZE + NUM_DURATIONS  # 8198
MAX_TIME_STEPS = (
    1320  # encoder input frames — latency-optimized for IQ9 (<100ms target)
)


LITEMP_PERCENTAGE: float = 10.0


class ParakeetEncoder(BaseModel):
    """
    Parakeet TDT Encoder.

    Takes mel features and attention mask, returns encoder output
    projected to joint hidden size.

    Consists of:
    - CNN subsampling (factor 8)
    - Relative positional encoding
    - 24x Conformer layers (hidden=1024)
    - Linear projector (1024 → 640)
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_features
            Mel spectrogram features of shape (batch, time, num_mel_bins).
        attention_mask
            Boolean mask of shape (batch, time).

        Returns
        -------
        torch.Tensor
            Encoder output of shape (batch, time_downsampled, joint_hidden_size).
        """
        # Cast int32 mask to bool (framework does not support bool dtype in InputSpec)
        out = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask.bool(),
        )
        return self.projector(out.last_hidden_state)

    def get_input_spec(
        self,
        batch_size: int = 1,
        time_steps: int = MAX_TIME_STEPS,
    ) -> InputSpec:
        # MAX_TIME_STEPS = 1320 frames (~13 s) — static encoder window.
        # Shorter audio is zero-padded; attention_mask marks real frames
        return {
            "input_features": TensorSpec(
                shape=(batch_size, time_steps, NUM_MEL_BINS),
                dtype="float32",
            ),
            "attention_mask": TensorSpec(
                shape=(batch_size, time_steps),
                dtype="int32",
                description="Boolean attention mask as int32. Cast to bool in forward().",
            ),
        }

    def get_output_spec(
        self,
        batch_size: int = 1,
        time_steps: int = MAX_TIME_STEPS,
    ) -> OutputSpec:
        return {
            "encoder_out": TensorSpec(
                shape=(batch_size, time_steps // 8, JOINT_HIDDEN_SIZE),
                dtype="float32",
                description="Encoder output projected to joint hidden size [B, T/8, 640]",
            ),
        }

    def component_precision(self) -> Precision:
        return Precision.float

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = Precision.float,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = "",
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    @classmethod
    def from_pretrained(cls, hf_model_id: str = HF_MODEL_ID) -> ParakeetEncoder:
        model = AutoModelForTDT.from_pretrained(hf_model_id, dtype=torch.float32)
        model.eval()
        return cls(
            encoder=model.encoder,
            projector=model.encoder_projector,
        )


class ParakeetDecoder(BaseModel):
    """
    Parakeet TDT Decoder (Predictor / RNN-T Prediction Network).

    Takes the previous token ID and LSTM hidden/cell states,
    returns the decoder output and updated LSTM states.

    Consists of:
    - Token embedding (8193 → 640)
    - 2-layer LSTM (hidden=640)
    - Linear projector (640 → 640)

    Note: LSTM states must be passed explicitly for stateful
    autoregressive decoding on device.
    """

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.embedding: nn.Embedding = decoder.embedding  # type: ignore[assignment]
        self.lstm: nn.LSTM = decoder.lstm  # type: ignore[assignment]
        self.projector: nn.Linear = decoder.decoder_projector  # type: ignore[assignment]

    def forward(
        self,
        token_id: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        token_id
            Previous predicted token ID of shape (batch, 1).
        h_in
            LSTM hidden state of shape (num_layers, batch, hidden_size).
        c_in
            LSTM cell state of shape (num_layers, batch, hidden_size).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            decoder_out of shape (batch, 1, hidden_size),
            h_out of shape (num_layers, batch, hidden_size),
            c_out of shape (num_layers, batch, hidden_size).
        """
        # Explicit gate ops instead of fused nn.LSTM: the fused kernel
        # gave 4-11 dB PSNR under w8a8_mixed_fp16 on HTP.
        emb = self.embedding(token_id.long())  # (batch, 1, 640)

        # Layer 0 LSTM gates
        x0 = emb[:, 0, :]  # (batch, 640)
        h0 = h_in[0]  # (batch, 640)
        c0 = c_in[0]  # (batch, 640)

        gates0 = (
            x0 @ self.lstm.weight_ih_l0.t()  # type: ignore[operator]
            + self.lstm.bias_ih_l0
            + h0 @ self.lstm.weight_hh_l0.t()  # type: ignore[operator]
            + self.lstm.bias_hh_l0
        )  # (batch, 2560)
        i0, f0, g0, o0 = gates0.chunk(4, dim=-1)  # each (batch, 640)
        c0_new = torch.sigmoid(f0) * c0 + torch.sigmoid(i0) * torch.tanh(g0)
        h0_new = torch.sigmoid(o0) * torch.tanh(c0_new)

        # Layer 1 LSTM gates
        h1 = h_in[1]  # (batch, 640)
        c1 = c_in[1]  # (batch, 640)

        gates1 = (
            h0_new @ self.lstm.weight_ih_l1.t()  # type: ignore[operator]
            + self.lstm.bias_ih_l1
            + h1 @ self.lstm.weight_hh_l1.t()  # type: ignore[operator]
            + self.lstm.bias_hh_l1
        )  # (batch, 2560)
        i1, f1, g1, o1 = gates1.chunk(4, dim=-1)  # each (batch, 640)
        c1_new = torch.sigmoid(f1) * c1 + torch.sigmoid(i1) * torch.tanh(g1)
        h1_new = torch.sigmoid(o1) * torch.tanh(c1_new)

        # ── Stack states + project ────────────────────────────────────────────
        out = h1_new.unsqueeze(1)  # (batch, 1, 640)
        h_out = torch.stack([h0_new, h1_new])  # (2, batch, 640)
        c_out = torch.stack([c0_new, c1_new])  # (2, batch, 640)
        decoder_out = self.projector(out)  # (batch, 1, 640)
        return decoder_out, h_out, c_out

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "token_id": TensorSpec(
                shape=(batch_size, 1),
                dtype="int32",
            ),
            "h_in": TensorSpec(
                shape=(NUM_DECODER_LSTM_LAYERS, batch_size, DECODER_HIDDEN_SIZE),
                dtype="float32",
            ),
            "c_in": TensorSpec(
                shape=(NUM_DECODER_LSTM_LAYERS, batch_size, DECODER_HIDDEN_SIZE),
                dtype="float32",
            ),
        }

    def get_output_spec(self, batch_size: int = 1) -> OutputSpec:
        return {
            "decoder_out": TensorSpec(
                shape=(batch_size, 1, DECODER_HIDDEN_SIZE),
                dtype="float32",
                description="Decoder output for joint network",
            ),
            "h_out": TensorSpec(
                shape=(NUM_DECODER_LSTM_LAYERS, batch_size, DECODER_HIDDEN_SIZE),
                dtype="float32",
                description="Updated LSTM hidden state",
            ),
            "c_out": TensorSpec(
                shape=(NUM_DECODER_LSTM_LAYERS, batch_size, DECODER_HIDDEN_SIZE),
                dtype="float32",
                description="Updated LSTM cell state",
            ),
        }

    def component_precision(self) -> Precision:
        return Precision.w8a8_mixed_fp16

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision = Precision.float,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = "",
    ) -> str:
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    @staticmethod
    def get_hub_litemp_percentage(precision: Precision) -> float:
        """Return the Lite-MP FP16 override percentage.

        The current AI Hub mixed-precision quantization flow requires an explicit
        Lite-MP percentage for ``w8a8_mixed_fp16``. The value 10 is the validated
        decoder/joint configuration used for this model.
        """
        return LITEMP_PERCENTAGE

    @classmethod
    def from_pretrained(cls, hf_model_id: str = HF_MODEL_ID) -> ParakeetDecoder:
        model = AutoModelForTDT.from_pretrained(hf_model_id, dtype=torch.float32)
        model.eval()
        return cls(decoder=model.decoder)


class ParakeetJoint(BaseModel):
    """
    Parakeet TDT Joint Network.

    Fuses encoder and decoder outputs to produce token and duration logits.

    Consists of:
    - Element-wise addition of encoder + decoder representations
    - ReLU activation
    - Linear head (640 → 8198)

    Output is split into:
    - Token logits: first 8193 values (vocab + blank)
    - Duration logits: last 5 values (durations 0-4)
    """

    def __init__(self, joint: nn.Module) -> None:
        super().__init__()
        self.activation: nn.ReLU = joint.activation  # type: ignore[assignment]
        self.head: nn.Linear = joint.head  # type: ignore[assignment]

    def forward(
        self,
        enc_frame: torch.Tensor,
        dec_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_frame
            Single encoder frame of shape (batch, 1, joint_hidden_size).
        dec_out
            Decoder output of shape (batch, 1, joint_hidden_size).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 1, 8198).
            First 8193 are token logits, last 5 are duration logits.
        """
        fused = enc_frame + dec_out
        return self.head(self.activation(fused))

    def get_input_spec(self, batch_size: int = 1) -> InputSpec:
        return {
            "enc_frame": TensorSpec(
                shape=(batch_size, 1, JOINT_HIDDEN_SIZE),
                dtype="float32",
            ),
            "dec_out": TensorSpec(
                shape=(batch_size, 1, JOINT_HIDDEN_SIZE),
                dtype="float32",
            ),
        }

    def get_output_spec(self, batch_size: int = 1) -> OutputSpec:
        return {
            "logits": TensorSpec(
                shape=(batch_size, 1, JOINT_OUTPUT_SIZE),
                dtype="float32",
                description="Token logits [0:8193] + duration logits [8193:8198]",
            ),
        }

    def component_precision(self) -> Precision:
        return Precision.w8a8_mixed_fp16

    @staticmethod
    def get_hub_litemp_percentage(precision: Precision) -> float:
        """Return the Lite-MP FP16 override percentage.

        The current AI Hub mixed-precision quantization flow requires an explicit
        Lite-MP percentage for ``w8a8_mixed_fp16``. The value 10 is the validated
        decoder/joint configuration used for this model.
        """
        return LITEMP_PERCENTAGE

    def _sample_inputs_impl(  # type: ignore[override]
        self,
        input_spec: InputSpec | None = None,
        **kwargs: object,
    ) -> dict[str, list]:
        # Realistic ranges from actual inference:
        # enc_frame: encoder output ≈ [-2, +2]
        # dec_out:   decoder output ≈ [-4, +0.4]
        rng = np.random.default_rng()
        return {
            "enc_frame": [rng.uniform(-2.0, 2.0, (1, 1, 640)).astype(np.float32)],
            "dec_out": [rng.uniform(-4.0, 0.4, (1, 1, 640)).astype(np.float32)],
        }

    @classmethod
    def from_pretrained(cls, hf_model_id: str = HF_MODEL_ID) -> ParakeetJoint:
        model = AutoModelForTDT.from_pretrained(hf_model_id, dtype=torch.float32)
        model.eval()
        return cls(joint=model.joint)


class ParakeetTDT(WorkbenchModelCollection):
    """
    Parakeet TDT 0.6B v3 — Multilingual ASR Model.

    3-component collection:
    - encoder : audio features → acoustic representations
    - decoder : token history  → prediction vectors (LSTM)
    - joint   : fuse encoder + decoder → token + duration logits
    """

    def __init__(
        self,
        encoder: ParakeetEncoder,
        decoder: ParakeetDecoder,
        joint: ParakeetJoint,
    ) -> None:
        super().__init__(
            {
                "encoder": encoder,
                "decoder": decoder,
                "joint": joint,
            }
        )
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint

    def get_component_mixed_precision(
        self, component_name: str, precision: Precision
    ) -> Precision:
        if precision == Precision.mixed:
            # encoder=w8a8, decoder=w8a8, joint=w8a8_mixed_fp16
            if component_name == "joint":
                return Precision.w8a8_mixed_fp16
            return Precision.w8a8
        return super().get_component_mixed_precision(component_name, precision)

    def get_component_litemp_percentage(
        self, component_name: str, precision: Precision
    ) -> float | None:
        if component_name == "joint" and precision in (
            Precision.mixed,
            Precision.mixed_with_float,
        ):
            return LITEMP_PERCENTAGE
        return None

    @classmethod
    def from_pretrained(cls, hf_model_id: str = HF_MODEL_ID) -> Self:
        """Load all 3 components from pretrained HuggingFace model."""
        model = AutoModelForTDT.from_pretrained(hf_model_id, dtype=torch.float32)
        model.eval()

        encoder = ParakeetEncoder(
            encoder=model.encoder,
            projector=model.encoder_projector,
        )
        decoder = ParakeetDecoder(decoder=model.decoder)
        joint = ParakeetJoint(joint=model.joint)

        return cls(encoder=encoder, decoder=decoder, joint=joint)

    def get_evaluator(self) -> BaseEvaluator:
        return WEREvaluator()

    def write_supplementary_files(
        self, output_dir: str | os.PathLike, metadata: ModelMetadata
    ) -> None:
        write_parakeet_tdt_supplementary_files(
            output_dir,
            metadata,
            model_name="parakeet-tdt-0.6b-v3",
            hf_model_id=HF_MODEL_ID,
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [LibriSpeechDataset]
