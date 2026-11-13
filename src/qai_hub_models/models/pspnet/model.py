# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
from torch import Tensor, nn
from typing_extensions import Self

from qai_hub_models import SampleInputsType
from qai_hub_models.datasets.ade20k import ADE10SegmentationDataset
from qai_hub_models.models.pspnet.external_repos import EXTERNAL_REPO_PATHS
from qai_hub_models.models.pspnet.external_repos.semseg.model.pspnet import (
    PSPNet as PSPNetImpl,
)
from qai_hub_models.models.pspnet.external_repos.semseg.util import (
    config as semseg_config,
)
from qai_hub_models.models.templates.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
    normalize_image_torchvision,
)
from qai_hub_models.models.templates.segmentation.segmentation_evaluator import (
    SegmentationOutputEvaluator,
)
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_torch,
)
from qai_hub_models.utils.base_dataset import BaseDataset
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID: str = __name__.split(".")[-2]
MODEL_ASSET_VERSION: int = 3

# Default model checkpoint path from asset store
DEFAULT_MODEL_PATH: str = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "pspnet101_ade20k_modified.pth"
)
INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "ADE_val_00001515.jpg"
)
NUM_CLASSES = 150

_SEMSEG_REPO_ROOT = EXTERNAL_REPO_PATHS["semseg"]


class PSPNet(CityscapesSegmentor):
    # PSPNet model wrapper class extending BaseModel.

    @classmethod
    def from_pretrained(
        cls, ckpt: str | CachedWebModelAsset = DEFAULT_MODEL_PATH
    ) -> Self:
        """
        Load a pretrained PSPNet model from a checkpoint.

        Parameters
        ----------
        ckpt
            Path to the checkpoint file or a cached model asset. Defaults to DEFAULT_MODEL_PATH.

        Returns
        -------
        model : Self
            An instance of PSPNet initialized with pretrained weights.
        """
        # Load configuration
        config_path = str(
            _SEMSEG_REPO_ROOT / "config" / "ade20k" / "ade20k_pspnet101.yaml"
        )
        args = semseg_config.load_cfg_from_cfg_file(config_path)
        # Initialize model
        model: nn.Module = PSPNetImpl(
            layers=args.layers,
            classes=args.classes,
            zoom_factor=args.zoom_factor,
            pretrained=False,
        )
        # Load weights
        checkpoint = load_torch(ckpt)
        model.load_state_dict(checkpoint, strict=False)

        return cls(model)

    def forward(self, image: Tensor) -> Tensor:
        """
        Perform a forward pass through the PSPNet model.

        Parameters
        ----------
        image
            Pixel values pre-processed for model consumption.
            Range: float[0, 1]
            3-channel Color Space: RGB
            Shape: [1, 3, H, W], where (H - 1) and (W - 1) are divisible by 8.


        Returns
        -------
        segmentation_mask : Tensor
            Returns segmentation prediction mask of shape (B, C, H, W):(Batch_Size, 150, 473, 473).
            Representing the class scores for each pixel.
        """
        input_tensor = torch.cat([image, image.flip(3)], 0)
        input_tensor = normalize_image_torchvision(input_tensor)
        output = self.model(input_tensor)
        output = torch.nn.functional.softmax(output, dim=1)
        return ((output[0] + output[1].flip(2)) / 2).unsqueeze(0)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 473,
        width: int = 473,
    ) -> InputSpec:
        # Get the input specification ordered (name -> (shape, type)) pairs for this model.
        #
        # This can be used with the qai_hub python API to declare
        # the model input specification upon submitting a compile job.
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

    def get_output_spec(self) -> OutputSpec:
        return {
            "mask": TensorSpec(
                apply_runtime_channel_reordering=True,
            ),
        }

    def _sample_inputs_impl(
        self, input_spec: InputSpec | None = None
    ) -> SampleInputsType:
        image = load_image(INPUT_IMAGE_ADDRESS)
        if input_spec is not None:
            h, w = input_spec["image"][0][2:]
            image = image.resize((w, h))
        else:
            h, w = self.get_input_spec()["image"][0][2:]
            image = image.resize((w, h))
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}

    def get_evaluator(self) -> BaseEvaluator:
        return SegmentationOutputEvaluator(NUM_CLASSES, resize_to_gt=True)

    @classmethod
    def get_eval_dataset_classes(cls) -> list[type[BaseDataset]]:
        return [ADE10SegmentationDataset]

    def get_calibration_dataset_cls(self) -> type[BaseDataset]:
        return ADE10SegmentationDataset
