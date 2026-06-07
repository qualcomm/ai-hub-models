# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.hf_whisper.model import (
    HfWhisper,
    HfWhisperDecoder,
    HfWhisperEncoder,
)
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
WHISPER_VERSION = "vinai/PhoWhisper-small"


@CollectionModel.add_component(HfWhisperEncoder, "encoder")
@CollectionModel.add_component(HfWhisperDecoder, "decoder")
class PhoWhisperSmall(HfWhisper):
    @classmethod
    def get_hf_whisper_version(cls) -> str:
        return WHISPER_VERSION
