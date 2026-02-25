# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base as Sam2

from qai_hub_models.models._shared.sam2.model import (
    SAM2Decoder as SAM2DecoderBase,
)
from qai_hub_models.models._shared.sam2.model import (
    SAM2Encoder as SAM2EncoderBase,
)
from qai_hub_models.models._shared.sam2.model import (
    SAM2Loader as SAM2LoaderBase,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, SourceAsRoot
from qai_hub_models.utils.base_model import CollectionModel

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_MODEL_TYPE = "edgetam"
EDGETAM_SOURCE_REPOSITORY = "https://github.com/facebookresearch/EdgeTAM"
EDGETAM_COMMIT = "a1209a454c9950d531498074a95ecf3a3ba02dfd"

MODEL_REGISTERY = {
    DEFAULT_MODEL_TYPE: "edgetam.pt",
}

CONFIG_REGISTERY = {
    DEFAULT_MODEL_TYPE: "edgetam.yaml",
}


class EdgeTAMEncoder(SAM2EncoderBase):
    """Exportable EdgeTAM encoder that can be split into several parts."""

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> EdgeTAMEncoder:
        return EdgeTAMLoader.load(model_type)[1]


class EdgeTAMDecoder(SAM2DecoderBase):
    """This EdgeTAMDecoder is taken from the class SAM2ImagePredictor.predict from sam2."""

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> EdgeTAMDecoder:
        return EdgeTAMLoader.load(model_type)[2]


class EdgeTAMLoader(SAM2LoaderBase):
    """Helper class for loading and preparing a HTP-compatible EdgeTAM model."""

    @classmethod
    def load(
        cls,
        model_type: str = DEFAULT_MODEL_TYPE,
    ) -> tuple[Sam2, EdgeTAMEncoder, EdgeTAMDecoder]:
        sam2 = cls._load_sam2(model_type)
        cls._patch_sam2_for_qnn_compatibility(sam2)
        encoder = EdgeTAMEncoder(sam2)
        decoder = EdgeTAMDecoder(sam2)

        return sam2, encoder, decoder

    @classmethod
    def _load_sam2(cls, model_type: str = DEFAULT_MODEL_TYPE) -> Sam2:
        """Get the EdgeTAM model described by the given model type."""
        with SourceAsRoot(
            EDGETAM_SOURCE_REPOSITORY,
            EDGETAM_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            config_dir = Path(repo_path) / "sam2" / "configs"

            # Initialize Hydra config from the cloned repo
            cls._initialize_hydra_config(
                config_dir=config_dir,
                job_name="edgetam_inference",
            )

            if model_type not in MODEL_REGISTERY:
                raise RuntimeError(f"Weights not found for model type `{model_type}`.")

            asset = CachedWebModelAsset(
                "https://github.com/facebookresearch/EdgeTAM/raw/main/checkpoints/edgetam.pt",
                MODEL_ID,
                MODEL_ASSET_VERSION,
                f"{MODEL_REGISTERY[model_type]}",
            )
            asset.fetch()
            return build_sam2(
                CONFIG_REGISTERY[model_type], asset.local_cache_path, device="cpu"
            )


@CollectionModel.add_component(EdgeTAMEncoder)
@CollectionModel.add_component(EdgeTAMDecoder)
class EdgeTAM(CollectionModel):
    def __init__(
        self, sam2: Sam2, encoder: EdgeTAMEncoder, decoder: EdgeTAMDecoder
    ) -> None:
        super().__init__(*[encoder, decoder])
        self.sam2 = sam2
        self.encoder = encoder
        self.decoder = decoder

    @classmethod
    def from_pretrained(cls, model_type: str = DEFAULT_MODEL_TYPE) -> EdgeTAM:
        return cls(*EdgeTAMLoader.load(model_type))
