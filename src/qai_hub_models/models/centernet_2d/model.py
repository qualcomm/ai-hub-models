# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.datasets.coco import CocoDataset
from qai_hub_models.models.centernet_2d.evaluator import (
    CenternetDetectionEvaluator,
)
from qai_hub_models.models.templates.centernet.external_repos.centernet.src.lib.models.decode import (
    ctdet_decode,
)
from qai_hub_models.models.templates.centernet.model import CenterNet
from qai_hub_models.models.templates.corevision.image_processing import (
    pre_process_with_affine,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    InputSpec,
    OutputSpec,
)

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

IMAGE = CachedWebModelAsset.from_asset_store(MODEL_ID, MODEL_ASSET_VERSION, "image.jpg")

# checkpoint download from https://drive.google.com/file/d/18Q3fzzAsha_3Qid6mn4jcIFPeOGUaj1d
DEFAULT_WEIGHTS = CachedWebModelAsset.from_asset_store(
    MODEL_ID,
    MODEL_ASSET_VERSION,
    "ctdet_coco_dla_2x.pth",
)


class CenterNet2D(CenterNet):
    """
    CenterNet2D Object Detection

    Parameters
    ----------
    model
        Centernet 2D bbox model.
    ctdet_decode
        2D bbox detection decoder function.
    """

    def __init__(
        self,
        model: nn.Module,
        ctdet_decode: Callable[
            [
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                bool,
                int,
            ],
            torch.Tensor,
        ],
    ) -> None:
        super().__init__()
        self.model = model
        self.decode = ctdet_decode

    @classmethod
    def from_pretrained(cls, ckpt_path: str = "default") -> Self:
        heads = {"hm": 80, "wh": 2, "reg": 2}
        if ckpt_path == "default":
            ckpt_path = str(DEFAULT_WEIGHTS.fetch())
        model = super()._load_pose_net(ckpt_path, heads)

        return cls(model, ctdet_decode)

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run CenterNet2D model and returns hm, wh, reg.

        Parameters
        ----------
        image
            Preprocessed image with range[0-1] in RGB format.
            Shape [B, C, H, W] where B = batch size, C = 3, H = img height, W = img width.

        Returns
        -------
        hm : torch.Tensor
            Heatmap with the shape of [B, num_classes, H//4, W//4] where num_classes = 80.
        wh : torch.Tensor
            Width/Height value with the shape of [B, 2, H//4, W//4].
        reg : torch.Tensor
            2D regression value with the shape of [B, 2, H//4, W//4].
        """
        image = image[:, [2, 1, 0]]
        mean = torch.Tensor([0.408, 0.447, 0.470]).reshape(1, 3, 1, 1)
        std = torch.Tensor([0.289, 0.274, 0.278]).reshape(1, 3, 1, 1)
        image = (image - mean) / std

        hm, wh, reg = self.model(image)[-1].values()
        hm = torch.sigmoid(hm)
        return hm, wh, reg

    def get_output_spec(self) -> OutputSpec:
        return {
            "hm": TensorSpec(),
            "wh": TensorSpec(),
            "reg": TensorSpec(),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(IMAGE.fetch())
        image_array = np.array(image)
        h, w = self.get_input_spec()["image"][0][2:]
        height, width = image_array.shape[0:2]
        c = np.array([width / 2, height / 2], dtype=np.float32)
        s = np.array([max(height, width), max(height, width)], dtype=np.float32)

        img = pre_process_with_affine(image_array, c, s, 0, (h, w))
        return {
            "image": [img.numpy()],
        }

    def get_evaluator(self) -> BaseEvaluator:
        image_height, image_width = self.get_input_spec()["image"][0][2:]
        return CenternetDetectionEvaluator(
            image_height // 4,
            image_width // 4,
            score_threshold=0.20,
            decode=self.decode,
        )

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [CocoDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return CocoDataset
