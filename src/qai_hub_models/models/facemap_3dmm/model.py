# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from torch import nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.models.facemap_3dmm.dataset import (
    CocoFaceDataset,
    FaceMap3DMMDataset,
)
from qai_hub_models.models.facemap_3dmm.evaluator import (
    FaceMap3DMMEvaluator,
)
from qai_hub_models.models.facemap_3dmm.resnet_score_rgb import resnet18_wd2
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_torch,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
DEFAULT_WEIGHTS = "resnet_wd2_weak_score_1202_3ch.pth.tar"
MODEL_ASSET_VERSION = 1
INPUT_IMAGE_PATH = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "face_img.jpg"
)


class FaceMap_3DMM(BaseModel):
    """Exportable FaceMap_3DMM, end-to-end."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(cls) -> Self:
        resnet_model = resnet18_wd2(pretrained=False)

        checkpoint_path = CachedWebModelAsset.from_asset_store(
            MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
        )
        pretrained_dict = load_torch(checkpoint_path)["state_dict"]
        resnet_model.load_state_dict(pretrained_dict)
        resnet_model.to(torch.device("cpu")).eval()

        return cls(resnet_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run ResNet18_0.5 3Ch on `image`, and produce 265 outputs

        Parameters
        ----------
        image
            Pixel values pre-processed for encoder consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        parameters_3dmm : torch.Tensor
            3DMM model parameters for facial landmark reconstruction: Shape [batch, 265]
        """
        return self.model(image * 255)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 128,
        width: int = 128,
    ) -> InputSpec:
        return {
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_PATH)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    def get_output_spec(self) -> OutputSpec:
        return {
            "parameters_3dmm": TensorSpec(),
        }

    def get_evaluator(self) -> BaseEvaluator:
        return FaceMap3DMMEvaluator(*self.get_input_spec()["image"][0][2:])

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [FaceMap3DMMDataset, CocoFaceDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return FaceMap3DMMDataset
