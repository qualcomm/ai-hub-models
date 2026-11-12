# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.models.nafnet_deblur.dataset import REDSDataset
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.models.templates.nafnet.model import (
    NAFNetModel,
    _load_nafnet_source_model,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
)
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_images/blurry.jpg"
)
DEFAULT_WEIGHTS = "NAFNet-REDS-width64.pth"
DEFAULT_CONFIG = "REDS/NAFNet-width64.yml"


class NafNetDeBlur(NAFNetModel):
    """Exportable NAFNet image restoration model for deblurring, end-to-end."""

    @classmethod
    def from_pretrained(
        cls,
        nafnet_weights: str | Path | None = None,
        yaml_path_nafnet: str | Path | None = None,
    ) -> Self:
        """Load NAFNet from a weightfile created by the source NAFNet repository.

        Parameters
        ----------
        nafnet_weights
            Local path to a ``.pth`` weights file.  When ``None`` the default
            pretrained weights are downloaded from the asset store.
        yaml_path_nafnet
            Local path to a ``.yml`` config file.  When ``None`` the default
            config bundled with the NAFNet repository is used.

        Returns
        -------
        Self
            Instantiated NafNetDeBlur model with pretrained weights loaded.
        """
        nafnet_model = _load_nafnet_source_model(
            nafnet_weights=Path(nafnet_weights)
            if nafnet_weights is not None
            else DEFAULT_WEIGHTS,
            yaml_path_nafnet=Path(yaml_path_nafnet)
            if yaml_path_nafnet is not None
            else DEFAULT_CONFIG,
            MODEL_ID=MODEL_ID,
            MODEL_ASSET_VERSION=MODEL_ASSET_VERSION,
        )
        return cls(nafnet_model)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 360,
        width: int = 640,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            ),
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "deblurred_image": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type]:
        return [REDSDataset]

    def get_calibration_dataset_cls(self) -> type:
        return REDSDataset
