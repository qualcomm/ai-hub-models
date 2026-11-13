# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoProcessor

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_config import BaseQAIHMConfig

CONFIG_NAME = "config.json"
VOCAB_BIN_NAME = "vocab.bin"

PARAKEET_TDT_SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "el": "Greek",
    "cs": "Czech",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hr": "Croatian",
    "sr": "Serbian",
    "bg": "Bulgarian",
    "ro": "Romanian",
    "hu": "Hungarian",
    "fi": "Finnish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "uk": "Ukrainian",
    "ca": "Catalan",
}


class LanguageSpec(BaseQAIHMConfig):
    """Specification for a supported language."""

    code: str
    name: str


class ASRCapabilities(BaseQAIHMConfig):
    """ASR model capabilities."""

    streaming: bool = True
    word_timestamps: bool = False
    punctuation: bool = False


class ASRParameters(BaseQAIHMConfig):
    """ASR model parameters."""

    sample_rate: int = 16000
    max_time_steps: int = 1320
    vocab_size: int = 8193
    tdt_durations: list[int] = [0, 1, 2, 3, 4]


class ASRModelAssets(BaseQAIHMConfig):
    """Model asset file paths for 3-component TDT model."""

    encoder_path: str | None = None
    decoder_path: str | None = None
    joint_path: str | None = None
    vocab_path: str | None = None


class ASRRuntimeInfo(BaseQAIHMConfig):
    """Runtime configuration for TDT ASR model."""

    qnn_version: dict[str, int] | None = None
    arch: int = 64
    blank_id: int = 8192
    start_token_id: int = 0
    num_decoder_lstm_layers: int = 2
    decoder_hidden_size: int = 640
    encoder_output_size: int = 640
    max_symbols_per_step: int = 10
    lcs_delay: int = 25
    overlap_samples: int = 16000


class ParakeetTDTMetadata(BaseQAIHMConfig):
    """Metadata (config.json) consumed by the Voice AI ASR SDK."""

    name: str
    display_name: str
    version: str = "1.0.0"
    description: str
    model_type: str = "parakeet_tdt"
    supported_languages: list[LanguageSpec]
    capabilities: ASRCapabilities
    parameters: ASRParameters
    assets: ASRModelAssets | None = None
    runtime: ASRRuntimeInfo | None = None


def _qnn_version(tool_versions: ToolVersions | None) -> dict[str, int] | None:
    if tool_versions is None or tool_versions.qairt is None:
        return None
    framework = tool_versions.qairt.framework
    return {
        "major": int(framework.major),
        "minor": int(framework.minor),
        "patch": int(framework.patch or 0),
    }


def _write_vocab_bin(hf_model_id: str, path: Path) -> None:
    """Write vocab.bin from Parakeet BPE tokenizer sorted by token ID.

    Each token is written as UTF-8 bytes followed by a null terminator,
    matching the format expected by the Voice AI ASR SDK.
    """
    processor = AutoProcessor.from_pretrained(hf_model_id)
    vocab = processor.tokenizer.get_vocab()
    sorted_tokens = [token for token, _ in sorted(vocab.items(), key=lambda x: x[1])]
    with open(path, "wb") as f:
        for token in sorted_tokens:
            token_bytes = token.encode("utf-8")
            f.write(token_bytes)
            if b"\0" not in token_bytes:
                f.write(b"\0")


def write_parakeet_tdt_supplementary_files(
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
    model_name: str,
    hf_model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
) -> None:
    """Write vocab.bin and config.json used by the Voice AI ASR SDK for Parakeet TDT."""
    if metadata.runtime != TargetRuntime.VOICE_AI:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _find_asset(keyword: str) -> str:
        matches = [n for n in metadata.model_files if keyword in n.lower()]
        if len(matches) == 0:
            raise ValueError(
                f"No model file found containing {keyword!r} in "
                f"{list(metadata.model_files)}"
            )
        if len(matches) > 1:
            raise ValueError(f"Ambiguous model files for {keyword!r}: {matches}")
        return matches[0]

    asr = ParakeetTDTMetadata(
        name=model_name,
        display_name="Parakeet TDT 0.6B v3",
        description=(
            "NVIDIA Parakeet-TDT-0.6B-v3 is a 600M-parameter ASR model using "
            "Token-and-Duration Transducer (TDT) architecture with a FastConformer "
            "encoder and 2-layer LSTM predictor. Supports English and 25 European "
            "languages with streaming-capable chunked inference."
        ),
        supported_languages=[
            LanguageSpec(code=c, name=n)
            for c, n in PARAKEET_TDT_SUPPORTED_LANGUAGES.items()
        ],
        capabilities=ASRCapabilities(),
        parameters=ASRParameters(),
        assets=ASRModelAssets(
            encoder_path=_find_asset("encoder"),
            decoder_path=_find_asset("decoder"),
            joint_path=_find_asset("joint"),
            vocab_path=VOCAB_BIN_NAME,
        ),
        runtime=ASRRuntimeInfo(
            qnn_version=_qnn_version(metadata.tool_versions),
        ),
    )
    # Write vocab.bin from BPE tokenizer sorted by token ID
    _write_vocab_bin(hf_model_id, out_dir / VOCAB_BIN_NAME)
    metadata.supplementary_files[VOCAB_BIN_NAME] = (
        "Parakeet TDT vocabulary binary for converting model output token IDs to token strings"
    )

    asr.to_json(out_dir / CONFIG_NAME, exclude_defaults=False)
    metadata.supplementary_files[CONFIG_NAME] = (
        "ASR metadata JSON for Parakeet TDT 0.6B v3"
    )
