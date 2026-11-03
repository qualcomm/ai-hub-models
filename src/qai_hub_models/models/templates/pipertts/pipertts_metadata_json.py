# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
PiperTTS (Text-to-Speech) metadata schema.

This module defines the structure for ``tts.json`` files that document
PiperTTS model capabilities, voice specifications, runtime parameters,
and asset locations.
"""

from __future__ import annotations

import functools
import os

from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.common_voice import TTSLanguage
from qai_hub_models.models.templates.voiceai_tts.tts_metadata import (
    TTSMetadata,
    write_tts_supplementary_files,
)
from qai_hub_models.utils.base_config import BaseQAIHMConfig


class ModelAssets(BaseQAIHMConfig):
    """Paths to model asset files."""

    tokenizer: str | None = None
    normalizer: str | None = None
    encoder: str | None = None
    sdp: str | None = None
    flow: str | None = None
    decoder: str | None = None
    g2p_encoder: str | None = None
    g2p_decoder: str | None = None


def create_tts_metadata(
    language: TTSLanguage,
    tokenizer_bin_name: str | None,
    normalizer_bin_name: str,
    metadata: ModelMetadata,
    sample_rate: int,
    name_suffix: str = "",
) -> TTSMetadata:
    """
    Generate ``TTSMetadata`` for a PiperTTS model.

    Parameters
    ----------
    language
        Language code (e.g., "ENGLISH", "ITALIAN", "GERMAN").
    tokenizer_bin_name
        Name of the tokenizer binary file.
    normalizer_bin_name
        Name of the normalizer binary file.
    metadata
        ``ModelMetadata`` instance containing model files and tool versions.
    sample_rate
        Audio sample rate in Hz.
    name_suffix
        Suffix appended to the generated metadata ``name`` (e.g., "-amy").

    Returns
    -------
    TTSMetadata
        The generated TTS metadata object.
    """
    assets = ModelAssets()
    for file_name in metadata.model_files:
        lower = file_name.lower()
        if "charsiu_encoder" in lower:
            assets.g2p_encoder = file_name
        elif "charsiu_decoder" in lower:
            assets.g2p_decoder = file_name
        elif "encoder" in lower:
            assets.encoder = file_name
        elif "sdp" in lower:
            assets.sdp = file_name
        elif "flow" in lower:
            assets.flow = file_name
        elif "decoder" in lower:
            assets.decoder = file_name

    assets.tokenizer = tokenizer_bin_name
    assets.normalizer = normalizer_bin_name

    return TTSMetadata.from_tts_model(
        model_type="piper",
        display_model_name="Piper TTS",
        language=language,
        tool_versions=metadata.tool_versions,
        assets=assets,
        is_model_quantized=True,
        sample_rate=sample_rate,
        name_suffix=name_suffix,
    )


SKIP_TOKENIZER_LANGUAGES = {
    TTSLanguage.ITALIAN,
    TTSLanguage.GERMAN,
    TTSLanguage.ENGLISH,
}


def write_pipertts_supplementary_files(
    language: TTSLanguage,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
    sample_rate: int,
    name_suffix: str = "",
) -> None:
    """
    Write supplementary files for PiperTTS models.

    Parameters
    ----------
    language
        Language key (e.g., "ENGLISH", "ITALIAN", "GERMAN")
    output_dir
        Directory to write supplementary files to
    metadata
        Model metadata object to update with supplementary file info
    sample_rate
        Audio sample rate in Hz
    name_suffix
        Suffix appended to the generated metadata ``name``
    """
    write_tts_supplementary_files(
        language=language,
        output_dir=output_dir,
        metadata=metadata,
        create_metadata_fn=functools.partial(
            create_tts_metadata, name_suffix=name_suffix
        ),
        sample_rate=sample_rate,
        skip_tokenizer_languages=SKIP_TOKENIZER_LANGUAGES,
        metadata_description="PiperTTS metadata JSON",
    )
