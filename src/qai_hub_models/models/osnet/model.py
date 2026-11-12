# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
from typing_extensions import Self

from qai_hub_models import Precision
from qai_hub_models.datasets.reid.entire_id import ENTIReIDDataset
from qai_hub_models.datasets.reid.market1501 import Market1501Dataset
from qai_hub_models.models.osnet.external_repos.deep_person_reid.torchreid.models.osnet import (
    osnet_ibn_x1_0,
    osnet_x0_5,
    osnet_x0_25,
    osnet_x0_75,
    osnet_x1_0,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.reid.reid_evaluator import ReIDEvaluator
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import InputSpec, OutputSpec, TensorSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
DEFAULT_VARIANT = "osnet_x1_0"

VARIANTS = (
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
)

PRETRAINED_WEIGHTS: dict[str, CachedWebModelAsset] = {
    variant: CachedWebModelAsset.from_asset_store(
        MODEL_ID, MODEL_ASSET_VERSION, f"{variant}_reid.pth"
    )
    for variant in VARIANTS
}


def _load_pretrained_weights(model: torch.nn.Module, variant: str) -> None:
    cached_file = PRETRAINED_WEIGHTS[variant].fetch()
    state_dict = torch.load(cached_file, map_location="cpu", weights_only=True)
    model_dict = model.state_dict()
    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    matched_layers: list[str] = []
    discarded_layers: list[str] = []

    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        if key in model_dict and model_dict[key].shape == value.shape:
            new_state_dict[key] = value
            matched_layers.append(key)
        else:
            discarded_layers.append(key)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f'Pretrained weights from "{cached_file}" did not match model keys.',
            stacklevel=2,
        )
    elif len(discarded_layers) > 0:
        print(f"Discarded unmatched pretrained layers: {discarded_layers}")


class OSNet(BaseModel):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model=model)

    @classmethod
    def from_pretrained(
        cls,
        variant: str = DEFAULT_VARIANT,
        pretrained: bool = True,
    ) -> Self:
        if variant not in PRETRAINED_WEIGHTS:
            raise NotImplementedError(f"Unsupported OSNet variant {variant}")

        osnet_factories = {
            "osnet_x1_0": osnet_x1_0,
            "osnet_x0_75": osnet_x0_75,
            "osnet_x0_5": osnet_x0_5,
            "osnet_x0_25": osnet_x0_25,
            "osnet_ibn_x1_0": osnet_ibn_x1_0,
        }
        backbone = osnet_factories[variant](num_classes=1000, pretrained=False)

        if pretrained:
            _load_pretrained_weights(backbone, variant)

        return cls(model=backbone)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run OSNet and return person re-identification embeddings.

        Parameters
        ----------
        image
            RGB image tensor of range [0, 1] and shape [N, 3, H, W].

        Returns
        -------
        embeddings : torch.Tensor
            Feature embedding tensor of shape [N, feature_dim].
            L2-normalized.
        """
        image = normalize_image_torchvision(image)
        embeddings = self.model(image)
        return F.normalize(embeddings, p=2, dim=1)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 256,
        width: int = 128,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                apply_runtime_channel_reordering=True,
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {"embeddings": TensorSpec()}

    def get_evaluator(self) -> BaseEvaluator:
        return ReIDEvaluator()

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [Market1501Dataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return ENTIReIDDataset

    def get_hub_quantize_options(
        self, precision: Precision, other_options: str = ""
    ) -> str:
        # min_max: tf_enhanced clips OSNet embeddings and tanks w8a8 ReID mAP.
        options = super().get_hub_quantize_options(precision, other_options)
        if "--range_scheme" not in options:
            options = (options + " --range_scheme min_max").strip()
        return options
