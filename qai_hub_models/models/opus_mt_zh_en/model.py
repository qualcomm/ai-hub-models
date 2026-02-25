# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os

from qai_hub_models.configs.metadata_yaml import ModelMetadata
from qai_hub_models.models._shared.opus_mt.model import (
    OpusMT,
    OpusMTDecoder,
    OpusMTEncoder,
)
from qai_hub_models.models._shared.opus_mt.utils import (
    write_opus_mt_supplementary_files,
)
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
OPUS_MT_VERSION = "Helsinki-NLP/opus-mt-zh-en"


@CollectionModel.add_component(OpusMTEncoder)
@CollectionModel.add_component(OpusMTDecoder)
class OpusMTZhEn(OpusMT):
    @classmethod
    def get_opus_mt_version(cls) -> str:
        return OPUS_MT_VERSION

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_opus_mt_supplementary_files(OPUS_MT_VERSION, output_dir, metadata)
