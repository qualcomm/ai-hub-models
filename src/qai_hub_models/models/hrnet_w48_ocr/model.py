# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from typing_extensions import Self

from qai_hub_models.models.hrnet_w48_ocr.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.hrnet_w48_ocr.external_repos.hrnet_semantic_seg.lib.config import (
    config,
)
from qai_hub_models.models.hrnet_w48_ocr.external_repos.hrnet_semantic_seg.lib.models.seg_hrnet_ocr import (
    get_seg_model,
)
from qai_hub_models.models.templates.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "hrnet_ocr_cs_8162_torch11.pth"
MODEL_ASSET_VERSION = 1
_HRNET_OCR_REPO_ROOT = EXTERNAL_REPO_PATHS["hrnet_semantic_seg"]


class HRNET_W48_OCR(CityscapesSegmentor):
    """Exportable HRNET_W48_OCR Image segmentation, end-to-end."""

    @classmethod
    def from_pretrained(cls, weights: str | None = None) -> Self:
        """Load HRNET_W48_OCR from a weightfile created by the source repository."""
        if not weights:
            weights = CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()

        config_file = str(
            _HRNET_OCR_REPO_ROOT
            / "experiments"
            / "cityscapes"
            / "seg_hrnet_ocr_w48_trainval_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml"
        )
        config_list = ["MODEL.NUM_OUTPUTS", "1", "MODEL.PRETRAINED", str(weights)]

        config.defrost()
        config.merge_from_file(config_file)
        config.merge_from_list(config_list)
        config.freeze()

        model = get_seg_model(config)
        model.eval()
        return cls(model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Predict semantic segmentation an input `image`.

        Parameters
        ----------
        image
            A [1, 3, height, width] image.
            RGB, range [0 - 1]
            Assumes image has been resized and normalized using the
            Cityscapes preprocesser (in cityscapes_segmentation/app.py).

        Returns
        -------
        class_logits : torch.Tensor
            Raw logit probabilities as a tensor of shape
            [1, num_classes, modified_height, modified_width],
            where the modified height and width will be some factor smaller
            than the input image.
        """
        return self.model(normalize_image_torchvision(image))[1]
