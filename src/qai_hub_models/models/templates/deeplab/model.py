# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
from collections.abc import Sequence

import torch

from qai_hub_models import SampleInputsType
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.pascal_voc import VOCSegmentationDataset
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.corevision.labels import write_labels_file
from qai_hub_models.models.templates.deeplab.dataset import CocoVocSegDataset
from qai_hub_models.models.templates.segmentation.segmentation_evaluator import (
    SegmentationOutputEvaluator,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
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

NUM_CLASSES = 21


class DeepLabV3Model(BaseModel):
    def __init__(
        self,
        deeplabv3_model: torch.nn.Module,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.model = deeplabv3_model
        self.normalize_input = normalize_input

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(NUM_CLASSES)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run DeepLabV3_Plus_Mobilenet on `image`, and produce a tensor of classes for segmentation

        Parameters
        ----------
        image
            Pixel values pre-processed for model consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB

        Returns
        -------
        segmentation_mask : torch.Tensor
            BxHxW tensor of class indices per pixel
        """
        if self.normalize_input:
            image = normalize_image_torchvision(image)
        model_out = self.model(image)
        if isinstance(model_out, dict):
            model_out = model_out["out"]
        return model_out.argmax(1).byte()

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 520,
        width: int = 520,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a profile job.
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
            )
        }

    def get_output_spec(self) -> OutputSpec:
        return {
            "mask": TensorSpec(
                io_type=IoType.TENSOR,
                description="Semantic segmentation mask",
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image_address = CachedWebModelAsset.from_asset_store(
            "deeplabv3_plus_mobilenet", 2, "deeplabv3_demo.png"
        )
        image = load_image(image_address)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    @classmethod
    def get_eval_dataset_classes(cls) -> Sequence[type[BaseDataset]]:
        return [VOCSegmentationDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        # Pascal VOC is non-commercial, so it can't ship as calibration data.
        # Calibrate from COCO (CC-BY, auto-downloadable) remapped to VOC labels;
        # eval stays on VOC (fetched from private S3 for CI).
        return CocoVocSegDataset

    def write_supplementary_files(
        self,
        output_dir: str | os.PathLike,
        metadata: ModelMetadata,
    ) -> None:
        write_labels_file("voc", output_dir, metadata)
