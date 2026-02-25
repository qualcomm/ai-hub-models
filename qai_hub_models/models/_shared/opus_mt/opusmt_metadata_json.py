# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
T2T (Text-to-Text) metadata schema for translation models.

This module defines the structure for t2t.json files that document
translation model capabilities, language pairs, and runtime parameters.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_config import BaseQAIHMConfig

# No constants defined here - they are passed as parameters


class LanguageSpec(BaseQAIHMConfig):
    """Specification for a language."""

    code: str
    name: str


class TranslationCapabilities(BaseQAIHMConfig):
    """Translation capabilities specification."""

    batch_translation: bool = True
    language_detection: bool = False
    formality_control: bool = False


class TranslationParameters(BaseQAIHMConfig):
    """Translation model parameters."""

    max_text_length: int = 1024
    max_batch_size: int = 100


class ModelAssets(BaseQAIHMConfig):
    """Model asset file paths."""

    encoder_path: str | None = None
    decoder_path: str | None = None
    tokenizer_path: str | None = None
    lookups_path: str | None = None


class RuntimeInfo(BaseQAIHMConfig):
    """Runtime configuration information."""

    qnn_version: dict[str, int] | None = None
    arch: int = 64
    enc_model_max_seq_len: int
    dec_model_max_seq_len: int
    rep_penalty: float = 1.2
    model_lang: str = "en"
    scratch_mem_size_req: int = 3200000


class T2TMetadata(BaseQAIHMConfig):
    """
    T2T (Text-to-Text) metadata for translation models.

    This represents the complete metadata for a translation model,
    including language pairs, capabilities, parameters, and runtime info.
    """

    name: str
    display_name: str
    version: str = "1.0.0"
    description: str
    source_languages: list[LanguageSpec]
    target_languages: list[LanguageSpec]
    capabilities: TranslationCapabilities
    parameters: TranslationParameters
    model_type: str = "opus"
    assets: ModelAssets | None = None
    runtime: RuntimeInfo | None = None
    tool_versions: ToolVersions | None = None

    def to_json(
        self,
        path: Path,
        indent: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Save T2T metadata to JSON file.

        Parameters
        ----------
        path
            Path to save the JSON file
        indent
            JSON indentation level
        **kwargs
            Additional arguments passed to json.dump

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary and handle nested objects
        data = self.model_dump()

        # Remove tool_versions from JSON output as it's not JSON serializable
        # and is only used internally for processing
        if "tool_versions" in data:
            del data["tool_versions"]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, **kwargs)

    @classmethod
    def from_opus_mt_model(
        cls,
        model_name: str,
        display_name: str,
        description: str,
        source_lang_code: str,
        source_lang_name: str,
        target_lang_code: str,
        target_lang_name: str,
        tool_versions: ToolVersions | None = None,
        **kwargs: Any,
    ) -> T2TMetadata:
        """
        Create T2TMetadata from OpusMT model information.

        Parameters
        ----------
        model_name
            Model identifier (e.g., "opus-en-zh")
        display_name
            Human-readable model name (e.g., "English to Chinese")
        description
            Model description
        source_lang_code
            Source language code (e.g., "en")
        source_lang_name
            Source language name (e.g., "English")
        target_lang_code
            Target language code (e.g., "zh")
        target_lang_name
            Target language name (e.g., "Chinese")
        tool_versions
            Tool version information
        **kwargs
            Additional parameters to override defaults

        Returns
        -------
        T2TMetadata
            Created T2T metadata object
        """
        # Create language specifications
        source_languages = [LanguageSpec(code=source_lang_code, name=source_lang_name)]
        target_languages = [LanguageSpec(code=target_lang_code, name=target_lang_name)]

        # Default capabilities
        capabilities = TranslationCapabilities()

        # Default parameters
        parameters = TranslationParameters()

        # Default runtime info with QNN version from tool_versions if available
        qnn_version = {"major": 2, "minor": 33, "patch": 0}  # Default
        if tool_versions and tool_versions.qairt is not None:
            # Parse QNN version string like "2.33.0" into major, minor, patch
            qnn_version = {
                "major": int(tool_versions.qairt.framework.major),
                "minor": int(tool_versions.qairt.framework.minor),
                "patch": int(
                    tool_versions.qairt.framework.patch
                    if tool_versions.qairt.framework.patch
                    else 0
                ),
            }

        runtime = RuntimeInfo(
            qnn_version=qnn_version,
            model_lang=source_lang_code,
            enc_model_max_seq_len=kwargs.get("enc_model_max_seq_len", 256),
            dec_model_max_seq_len=kwargs.get("dec_model_max_seq_len", 256),
        )

        # Override with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(capabilities, key):
                setattr(capabilities, key, value)
            elif hasattr(parameters, key):
                setattr(parameters, key, value)
            elif hasattr(runtime, key):
                setattr(runtime, key, value)

        return cls(
            name=model_name,
            display_name=display_name,
            description=description,
            source_languages=source_languages,
            target_languages=target_languages,
            capabilities=capabilities,
            parameters=parameters,
            runtime=runtime,
            tool_versions=tool_versions,
        )


# Language pair mapping for OpusMT models
OPUS_MT_LANGUAGE_PAIRS = {
    "Helsinki-NLP/opus-mt-en-es": {
        "source_code": "en",
        "source_name": "English",
        "target_code": "es",
        "target_name": "Spanish",
    },
    "Helsinki-NLP/opus-mt-en-zh": {
        "source_code": "en",
        "source_name": "English",
        "target_code": "zh",
        "target_name": "Chinese",
    },
    "Helsinki-NLP/opus-mt-es-en": {
        "source_code": "es",
        "source_name": "Spanish",
        "target_code": "en",
        "target_name": "English",
    },
    "Helsinki-NLP/opus-mt-zh-en": {
        "source_code": "zh",
        "source_name": "Chinese",
        "target_code": "en",
        "target_name": "English",
    },
}


def get_language_pair_info(hf_model_name: str) -> dict[str, str] | None:
    """
    Get language pair information for a given HuggingFace model name.

    Parameters
    ----------
    hf_model_name
        HuggingFace model ID, e.g. "Helsinki-NLP/opus-mt-en-es"

    Returns
    -------
    dict[str, str] | None
        Dictionary containing source_code, source_name, target_code, target_name
        or None if model is not supported
    """
    return OPUS_MT_LANGUAGE_PAIRS.get(hf_model_name)


def create_t2t_metadata(
    hf_model_name: str,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
    max_seq_len_enc: int,
    max_seq_len_dec: int,
    tokenizer_encode_name: str,
    tokenizer_decode_name: str,
) -> T2TMetadata | None:
    """
    Create T2T metadata and determine output path for an Opus-MT model.

    Parameters
    ----------
    hf_model_name
        HuggingFace model ID, e.g. "Helsinki-NLP/opus-mt-en-es"
    output_dir
        Output directory path for storing the generated JSON file
    metadata
        Model metadata containing tool versions and model files
    max_seq_len_enc
        Maximum sequence length for encoder
    max_seq_len_dec
        Maximum sequence length for decoder
    tokenizer_encode_name
        Name of the tokenizer encode binary file
    tokenizer_decode_name
        Name of the tokenizer decode binary file

    Returns
    -------
    T2TMetadata | None
        T2TMetadata object or None if generation failed
    """
    # Get language pair for this model
    lang_pair = get_language_pair_info(hf_model_name)
    if not lang_pair:
        return None

    # Create output directory and file path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract model asset files from metadata.model_files instead of hardcoding
    assets = ModelAssets()

    # Find encoder and decoder files from metadata.model_files
    encoder_file_name = None
    decoder_file_name = None

    for file_name in metadata.model_files:
        # Look for encoder file (contains "enc" in the name)
        if "enc" in file_name.lower():
            encoder_file_name = file_name
        # Look for decoder file (contains "dec" in the name)
        elif "dec" in file_name.lower():
            decoder_file_name = file_name

    # Set asset paths if files were found
    if encoder_file_name:
        assets.encoder_path = encoder_file_name
    if decoder_file_name:
        assets.decoder_path = decoder_file_name

    # Check for tokenizer files in output directory
    tokenizer_encode_file = output_path / tokenizer_encode_name
    tokenizer_decode_file = output_path / tokenizer_decode_name

    if tokenizer_encode_file.exists():
        assets.tokenizer_path = tokenizer_encode_file.name
    if tokenizer_decode_file.exists():
        assets.lookups_path = tokenizer_decode_file.name

    # Extract sequence lengths from metadata input specs
    enc_max_seq_len = max_seq_len_enc  # Default fallback
    dec_max_seq_len = max_seq_len_dec  # Default fallback

    # Try to extract sequence lengths from encoder input specs
    if encoder_file_name and encoder_file_name in metadata.model_files:
        encoder_metadata = metadata.model_files[encoder_file_name]
        if "encoder_attention_mask" in encoder_metadata.inputs:
            encoder_attention_mask = encoder_metadata.inputs["encoder_attention_mask"]
            if len(encoder_attention_mask.shape) >= 2:
                enc_max_seq_len = encoder_attention_mask.shape[1]

    # Try to extract sequence lengths from decoder input specs
    if decoder_file_name and decoder_file_name in metadata.model_files:
        decoder_metadata = metadata.model_files[decoder_file_name]
        if "encoder_attention_mask" in decoder_metadata.inputs:
            decoder_attention_mask = decoder_metadata.inputs["encoder_attention_mask"]
            if len(decoder_attention_mask.shape) >= 2:
                dec_max_seq_len = decoder_attention_mask.shape[1]

    # Create T2T metadata
    model_name = f"opus-{lang_pair['source_code']}-{lang_pair['target_code']}"
    display_name = f"{lang_pair['source_name']} to {lang_pair['target_name']}"
    description = f"Translation from {lang_pair['source_name']} to {lang_pair['target_name']} using Opus"

    t2t_metadata = T2TMetadata.from_opus_mt_model(
        model_name=model_name,
        display_name=display_name,
        description=description,
        source_lang_code=lang_pair["source_code"],
        source_lang_name=lang_pair["source_name"],
        target_lang_code=lang_pair["target_code"],
        target_lang_name=lang_pair["target_name"],
        tool_versions=metadata.tool_versions,
        # Pass the extracted sequence lengths to override defaults
        enc_model_max_seq_len=enc_max_seq_len,
        dec_model_max_seq_len=dec_max_seq_len,
    )
    t2t_metadata.assets = assets

    return t2t_metadata
