# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
MeloTTS (Text-to-Speech) metadata schema.

This module defines the structure for ``tts.json`` files that document
MeloTTS model capabilities, voice specifications, runtime parameters,
and asset locations.
"""

from __future__ import annotations

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.utils.base_config import BaseQAIHMConfig

LANGUAGE_MAP = {"ENGLISH": "en", "SPANISH": "es", "CHINESE": "zh"}


# ----------------------------------------------------------------------
# Data model definitions
# ----------------------------------------------------------------------
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
    sample_rate: int = 44100
    language_code: int = 0
    description: str


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


class RuntimeInfo(BaseQAIHMConfig):
    """Runtime configuration information."""

    language: str
    qnn_version_major: int
    qnn_version_minor: int
    qnn_version_patch: int
    arch_bit: int = 64
    scratch_mem_size_req: int = 3200000


class ModelAssets(BaseQAIHMConfig):
    """Paths to model asset files."""

    bert_model: str | None = None
    bert_tokenizer: str | None = None
    bert_normalizer: str | None = None
    melo_encoder: str | None = None
    melo_flow: str | None = None
    melo_decoder: str | None = None
    g2p_encoder: str | None = None
    g2p_decoder: str | None = None


# ----------------------------------------------------------------------
# Main metadata container
# ----------------------------------------------------------------------
class TTSMetadata(BaseQAIHMConfig):
    """
    TTS metadata for MeloTTS models.

    Mirrors the structure of ``tts.json`` generated for each model.
    """

    name: str
    display_name: str
    version: str = "1.0.0"
    description: str
    voices: list[VoiceSpec]
    capabilities: TTSCapabilities
    model_type: str = "melo"
    runtime: RuntimeInfo | None = None
    assets: ModelAssets | None = None
    tool_versions: ToolVersions | None = None

    # ------------------------------------------------------------------
    # Builds a TTSMetadata instance from model files
    # ------------------------------------------------------------------
    @classmethod
    def from_melo_tts_model(
        cls,
        language: str,
        model_name: str,
        display_name: str,
        description: str,
        voice_specs: list[VoiceSpec] | None = None,
        capabilities: TTSCapabilities | None = None,
        runtime: RuntimeInfo | None = None,
        assets: ModelAssets | None = None,
        tool_versions: ToolVersions | None = None,
    ) -> TTSMetadata:
        """
        Construct a ``TTSMetadata`` object from the information
        available in a MeloTTS model.

        Parameters
        ----------
        language
            Language (e.g. ``EN``, ``ZH``, ``ES``).
        model_name
            Identifier for the model (e.g. ``melo-tts-en``).
        display_name
            Human-readable name.
        description
            Short description of the model.
        voice_specs
            List of :class:`VoiceSpec` describing each voice.
        capabilities
            Optional capabilities object; defaults to all ``False``.
        runtime
            Optional runtime information; if omitted a minimal default is used.
        assets
            Optional asset paths; if omitted a minimal default is used.
        tool_versions
            Optional tool-version information.

        Returns
        -------
        TTSMetadata
            Fully populated metadata instance.
        """
        if capabilities is None:
            capabilities = TTSCapabilities()
        if runtime is None:
            # Default runtime - QNN version is taken from ``tool_versions`` if present.
            qnn_version = {"major": 2, "minor": 33, "patch": 0}
            if tool_versions and tool_versions.qairt is not None:
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
                language=LANGUAGE_MAP[language],
                qnn_version_major=qnn_version["major"],
                qnn_version_minor=qnn_version["minor"],
                qnn_version_patch=qnn_version["patch"],
            )
        if assets is None:
            assets = ModelAssets()

        if voice_specs is None:
            voice_specs = [
                VoiceSpec(
                    language=LANGUAGE_MAP[language],
                    language_name=language.capitalize(),
                    description=f"Default voice for {language.capitalize()}",
                )
            ]

        return cls(
            name=model_name,
            display_name=display_name,
            description=description,
            voices=voice_specs,
            capabilities=capabilities,
            runtime=runtime,
            assets=assets,
            tool_versions=tool_versions,
        )


# ----------------------------------------------------------------------
# Helper to create metadata and write the JSON file
# ----------------------------------------------------------------------
def create_tts_metadata(
    language: str,
    tokenizer_bin_name: str,
    normalizer_bin_name: str,
    metadata: ModelMetadata,
) -> TTSMetadata:
    """
    Generate ``TTSMetadata`` for a MeloTTS model.

    Parameters
    ----------
    language
        Language code (e.g., ``EN``, ``ES``, ``ZH``).
    tokenizer_bin_name
        Name of the tokenizer binary file.
    normalizer_bin_name
        Name of the normalizer binary file.
    metadata
        ``ModelMetadata`` instance containing model files and tool versions.

    Returns
    -------
    TTSMetadata
        The generated TTS metadata object.
    """
    # ------------------------------------------------------------------
    # Determine asset file names from ``metadata.model_files``
    # ------------------------------------------------------------------
    assets = ModelAssets()
    for file_name in metadata.model_files:
        lower = file_name.lower()
        if "bertwrapper" in lower:
            assets.bert_model = file_name
        elif "t5encoder" in lower:
            assets.g2p_encoder = file_name
        elif "t5decoder" in lower:
            assets.g2p_decoder = file_name
        elif "encoder" in lower:
            assets.melo_encoder = file_name
        elif "decoder" in lower:
            assets.melo_decoder = file_name
        elif "flow" in lower:
            assets.melo_flow = file_name

    assets.bert_tokenizer = tokenizer_bin_name
    assets.bert_normalizer = normalizer_bin_name

    # ------------------------------------------------------------------
    # Build the metadata object
    # ------------------------------------------------------------------
    model_name = f"melo_tts_{LANGUAGE_MAP[language]}"
    display_name = f"MeloTTS {language.capitalize()}"
    description = f"MeloTTS text-to-speech model for {language.capitalize()}"

    return TTSMetadata.from_melo_tts_model(
        language=language,
        model_name=model_name,
        display_name=display_name,
        description=description,
        assets=assets,
        tool_versions=metadata.tool_versions,
    )
