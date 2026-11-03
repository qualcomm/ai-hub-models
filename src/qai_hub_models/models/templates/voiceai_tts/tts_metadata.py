# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from pydantic import SerializeAsAny
from qai_hub_models_cli.common import LOCAL_STORE_DEFAULT_PATH
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from typing_extensions import Self

from qai_hub_models import TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.datasets.common_voice import LANG_CODE_MAP, TTSLanguage
from qai_hub_models.models.templates.voiceai_tts.generate_bert_binary_rules import (
    generate_bert_tokenizer_binary,
)
from qai_hub_models.models.templates.voiceai_tts.generate_unicode_bin import (
    generate_unicode_binary,
)
from qai_hub_models.models.templates.voiceai_tts.language import (
    BERT_MODEL_IDS,
    LANG_ID_MAP,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig


class TTSCapabilities(BaseQAIHMConfig):
    """Supported capabilities for a TTS model."""

    supports_gender: bool = False
    supports_style: bool = False
    supports_sample_rate: bool = False
    supports_ssml: bool = False
    supports_speed_control: bool = False
    supports_pitch_control: bool = False
    supports_volume_control: bool = False
    supports_resampling: bool = False


class QNNVersion(BaseQAIHMConfig):
    """Version of QNN SDK."""

    major: int
    minor: int
    patch: int = 0


class RuntimeInfo(BaseQAIHMConfig):
    """Runtime configuration information."""

    language: str
    qnn_version: QNNVersion
    arch_bit: int = 64
    scratch_mem_size_req: int = 3200000
    is_model_quantized: bool = False

    @classmethod
    def from_tool_versions(
        cls,
        language: str,
        tool_versions: ToolVersions,
        **kwargs: Any,
    ) -> RuntimeInfo:
        assert tool_versions.qairt is not None
        return cls(
            language=language,
            qnn_version=QNNVersion(
                major=int(tool_versions.qairt.framework.major),
                minor=int(tool_versions.qairt.framework.minor),
                patch=int(
                    tool_versions.qairt.framework.patch
                    if tool_versions.qairt.framework.patch
                    else 0
                ),
            ),
            **kwargs,
        )


class VoiceSpec(BaseQAIHMConfig):
    """Specification for a single voice."""

    name: str = "default"
    display_name: str = "Default Voice"
    language: str
    language_name: str
    gender: str = "neutral"
    style: str = "neutral"
    audio_encoding: int = 0
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume_gain: float = 0.0
    sample_rate: int
    language_code: int = 0
    description: str
    capabilities: TTSCapabilities


class TTSMetadata(BaseQAIHMConfig):
    """Base TTS metadata container shared by MeloTTS and PiperTTS."""

    name: str
    display_name: str
    version: str = "1.0.0"
    description: str
    voices: list[VoiceSpec]
    model_type: str
    runtime: RuntimeInfo | None = None
    assets: SerializeAsAny[BaseQAIHMConfig] | None = None

    @classmethod
    def from_tts_model(
        cls,
        model_type: str,
        display_model_name: str,
        language: TTSLanguage,
        tool_versions: ToolVersions,
        assets: BaseQAIHMConfig,
        sample_rate: int,
        is_model_quantized: bool = False,
        name_suffix: str = "",
    ) -> Self:
        lang_code = LANG_CODE_MAP[language].lower()
        lang_name = language.value.capitalize()
        runtime = RuntimeInfo.from_tool_versions(
            language=lang_code,
            tool_versions=tool_versions,
            is_model_quantized=is_model_quantized,
        )

        voices = [
            VoiceSpec(
                language=lang_code,
                language_name=lang_name,
                description=f"Default voice for {lang_name}",
                capabilities=TTSCapabilities(),
                speaking_rate=0.85 if language == TTSLanguage.ITALIAN else 1.0,
                sample_rate=sample_rate,
                language_code=LANG_ID_MAP[language],
            )
        ]

        return cls(
            name=f"{model_type}-tts-{lang_code}{name_suffix}",
            display_name=f"{display_model_name} {lang_name}",
            description=f"{display_model_name} text-to-speech model for {lang_name}",
            model_type=model_type,
            voices=voices,
            runtime=runtime,
            assets=assets,
        )


def build_ja_tokenizer(vocab_path: str) -> Tokenizer:
    vocab = {}
    with open(vocab_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            vocab[line.rstrip("\n")] = i
    tokenizer = Tokenizer(
        WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100)
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", vocab["[CLS]"]),
            ("[SEP]", vocab["[SEP]"]),
        ],
    )
    return tokenizer


def write_tts_supplementary_files(
    language: TTSLanguage,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
    create_metadata_fn: Callable[
        [TTSLanguage, str | None, str, ModelMetadata, int], TTSMetadata
    ],
    sample_rate: int,
    skip_tokenizer_languages: set[TTSLanguage] | None = None,
    metadata_description: str = "TTS metadata JSON",
) -> None:
    """
    Write supplementary files (tokenizer binary, unicode binary, config JSON)
    shared by both MeloTTS and PiperTTS.
    """
    if metadata.runtime != TargetRuntime.VOICE_AI:
        return
    lang_code = LANG_CODE_MAP[language].lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_bin_name: str | None = None
    if skip_tokenizer_languages is None or language not in skip_tokenizer_languages:
        if language == TTSLanguage.JAPANESE:
            tok_json_dir = os.path.join(
                LOCAL_STORE_DEFAULT_PATH, "bert-base-japanese-v3"
            )
            os.makedirs(tok_json_dir, exist_ok=True)
            tok_json_file = os.path.join(tok_json_dir, "tokenizer.json")
            tokenizer = build_ja_tokenizer(
                hf_hub_download(BERT_MODEL_IDS[language], "vocab.txt")
            )
            tokenizer.save(tok_json_file)
        elif language == TTSLanguage.KOREAN:
            tok_json_dir = os.path.join(LOCAL_STORE_DEFAULT_PATH, "bert-kor-base")
            os.makedirs(tok_json_dir, exist_ok=True)
            AutoTokenizer.from_pretrained(
                BERT_MODEL_IDS[language], use_fast=True
            ).save_pretrained(tok_json_dir)
            tok_json_file = os.path.join(tok_json_dir, "tokenizer.json")
        else:
            tok_json_file = hf_hub_download(BERT_MODEL_IDS[language], "tokenizer.json")

        tokenizer_bin_path = generate_bert_tokenizer_binary(
            tok_json_file,
            output_dir / f"bert_{lang_code}_tokenizer.bin",
        )
        tokenizer_bin_name = tokenizer_bin_path.name
        metadata.supplementary_files[tokenizer_bin_name] = (
            f"tokenizer binary for BERT {language.value.capitalize()} uncased vocabulary"
        )

    normalizer_bin_path = output_dir / "bert_normalizer.bin"
    generate_unicode_binary(normalizer_bin_path)
    metadata.supplementary_files[normalizer_bin_path.name] = (
        "optimized unicode binary for fast access"
    )

    tts_metadata = create_metadata_fn(
        language, tokenizer_bin_name, normalizer_bin_path.name, metadata, sample_rate
    )
    tts_metadata_path = output_dir / "config.json"
    tts_metadata.to_json(tts_metadata_path, exclude_defaults=False)
    metadata.supplementary_files[tts_metadata_path.name] = (
        f"{metadata_description} for {language.value.capitalize()}"
    )
