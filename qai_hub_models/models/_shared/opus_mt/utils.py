# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.models._shared.opus_mt.model import (
    MAX_SEQ_LEN_DEC,
    MAX_SEQ_LEN_ENC,
    TOKENIZER_DECODE_NAME,
    TOKENIZER_ENCODE_NAME,
    generate_tokenizer_bins,
)
from qai_hub_models.models._shared.opus_mt.opusmt_metadata_json import (
    create_t2t_metadata,
)
from qai_hub_models.models.common import TargetRuntime

# Language code to name mapping for supplementary file descriptions
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "zh": "Chinese",
}


def write_opus_mt_supplementary_files(
    opus_mt_version: str,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
) -> None:
    """
    Write supplementary files for OpusMT models.

    Parameters
    ----------
    opus_mt_version
        HuggingFace model ID, e.g. "Helsinki-NLP/opus-mt-en-es"
    output_dir
        Directory to write supplementary files to
    metadata
        Model metadata object to update with supplementary file info
    """
    if metadata.runtime != TargetRuntime.QNN_CONTEXT_BINARY:
        return

    # Parse language pair from model name (e.g., "Helsinki-NLP/opus-mt-en-es" -> "en", "es")
    model_suffix = opus_mt_version.split("/")[-1]  # "opus-mt-en-es"
    parts = model_suffix.split("-")
    src_lang = parts[-2]  # "en"
    tgt_lang = parts[-1]  # "es"

    src_name = LANGUAGE_NAMES.get(src_lang, src_lang.capitalize())
    tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang.capitalize())
    lang_pair = f"{src_name}-to-{tgt_name}"

    # Generate tokenizer binaries and record them as supplementary files
    encode_path, decode_path = generate_tokenizer_bins(opus_mt_version, output_dir)
    if not encode_path or not decode_path:
        raise ValueError("Failed to generate tokenizer binaries")

    # Add the tokenizer binaries to the metadata's supplementary files
    metadata.supplementary_files[encode_path.name] = (
        f"{lang_pair} Tokenizer encode binary for optimal token access during tokenization"
    )
    metadata.supplementary_files[decode_path.name] = (
        f"{lang_pair} Tokenizer decode binary for converting model output token IDs to token strings"
    )

    # Generate T2T metadata JSON and record it as a supplementary file
    t2t_metadata = create_t2t_metadata(
        opus_mt_version,
        output_dir,
        metadata,
        MAX_SEQ_LEN_ENC,
        MAX_SEQ_LEN_DEC,
        TOKENIZER_ENCODE_NAME,
        TOKENIZER_DECODE_NAME,
    )
    if t2t_metadata:
        t2t_path = Path(output_dir) / "config.json"
        t2t_metadata.to_json(t2t_path, exclude_defaults=False)
        if t2t_path:
            metadata.supplementary_files[t2t_path.name] = (
                f"T2T metadata JSON for {lang_pair} translation"
            )
