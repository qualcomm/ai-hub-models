# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
from pathlib import Path

import unidic
from huggingface_hub import hf_hub_download
from platformdirs import user_cache_path
from unidic.download import download_version as _download_unidic

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.models._shared.melotts.generate_bert_binary_rules import (
    generate_bert_tokenizer_binary,
)
from qai_hub_models.models._shared.melotts.generate_unicode_bin import (
    generate_unicode_binary,
)
from qai_hub_models.models._shared.melotts.meloTTS_metadata_json import (
    create_tts_metadata,
)
from qai_hub_models.models.common import TargetRuntime
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

UNIDIC_CACHE_PATH = user_cache_path("unidic")
UNICODE_DATA_ASSET = CachedWebModelAsset(
    url="https://www.unicode.org/Public/3.0-Update/UnicodeData-3.0.0.txt",
    model_id="melotts_shared",
    model_asset_version=1,
    filename="UnicodeData-3.0.0.txt",
)


def download_unidic() -> None:
    """
    Downloads supporting files for the unidic package to a shared global cache location.

    The default location dumps these files directly into the python environment folder,
    which makes working with multiple environments tedious, since we need to re-download
    500+mb of supporting files for every new python env.
    """
    if not os.path.exists(unidic.DICDIR):
        if os.name != "nt":
            if not os.path.exists(UNIDIC_CACHE_PATH):
                # This will delete the unidic folder in the python env if it exists already.
                # So we call it first before moving the results and symlinking the original dst folder.
                _download_unidic()
                UNIDIC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(unidic.DICDIR, UNIDIC_CACHE_PATH)
            os.symlink(UNIDIC_CACHE_PATH, unidic.DICDIR)
        else:
            # Do nothing special on Windows since symlinking works poorly there.
            _download_unidic()

    try:
        import MeCab

        MeCab.Tagger()
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load TTS language data. Try deleting {unidic.DICDIR} and try again."
        ) from e
    except ImportError as e:
        raise ImportError(
            "MeloTTS is not installed correctly. Refer to the model README for installation instructions."
        ) from e


# Language mappings used by app.py
LANGUAGE_MAP = {"ENGLISH": "EN", "SPANISH": "ES", "CHINESE": "ZH"}
BERT_MODEL_IDS = {
    "ENGLISH": "bert-base-uncased",
    "CHINESE": "bert-base-multilingual-uncased",
    "SPANISH": "dccuchile/bert-base-spanish-wwm-uncased",
}


def write_melotts_supplementary_files(
    language: str,
    output_dir: str | os.PathLike,
    metadata: ModelMetadata,
) -> None:
    """
    Write supplementary files for MeloTTS models.

    Parameters
    ----------
    language
        Language key (e.g., "ENGLISH", "SPANISH", "CHINESE")
    output_dir
        Directory to write supplementary files to
    metadata
        Model metadata object to update with supplementary file info
    """
    if metadata.runtime != TargetRuntime.QNN_CONTEXT_BINARY:
        return
    lang_code = LANGUAGE_MAP[language].lower()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer_bin_path = generate_bert_tokenizer_binary(
        hf_hub_download(repo_id=BERT_MODEL_IDS[language], filename="tokenizer.json"),
        output_dir / f"bert_{lang_code}_tokenizer.bin",
    )
    metadata.supplementary_files[tokenizer_bin_path.name] = (
        f"tokenizer binary for BERT {language.capitalize()} uncased vocabulary"
    )

    # Unicode binary
    normalizer_bin_path = output_dir / "bert_normalizer.bin"
    generate_unicode_binary(normalizer_bin_path)
    metadata.supplementary_files[normalizer_bin_path.name] = (
        "optimized unicode binary for fast access"
    )

    # TTS metadata JSON
    tts_metadata = create_tts_metadata(
        language, tokenizer_bin_path.name, normalizer_bin_path.name, metadata
    )
    tts_metadata_path = output_dir / "config.json"
    tts_metadata.to_json(tts_metadata_path, exclude_defaults=False)
    metadata.supplementary_files[tts_metadata_path.name] = (
        f"TTS metadata JSON for {language.capitalize()}"
    )
