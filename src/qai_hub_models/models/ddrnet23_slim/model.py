# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import Precision, SampleInputsType, TargetRuntime
from qai_hub_models.models.ddrnet23_slim.external_repos.ddrnet_pytorch.lib.models.ddrnet_23_slim import (
    BasicBlock,
    DualResNet,
)
from qai_hub_models.models.templates.cityscapes_segmentation.model import (
    CityscapesSegmentor,
)
from qai_hub_models.models.templates.corevision.image_processing import (
    app_to_net_image_inputs,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image
from qai_hub_models.utils.input_spec import (
    ColorFormat,
    ImageMetadata,
    InputSpec,
    IoType,
    OutputSpec,
    TensorSpec,
)

MODEL_ID = __name__.split(".")[-2]
# Originally from https://drive.google.com/file/d/1d_K3Af5fKHYwxSo8HkxpnhiekhwovmiP/view
DEFAULT_WEIGHTS = "DDRNet23s_imagenet.pth"
MODEL_ASSET_VERSION = 1
NUM_CLASSES = 19

INPUT_IMAGE_ADDRESS = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "test_input_image.png"
)


class DDRNet(CityscapesSegmentor):
    """Exportable DDRNet image segmenter, end-to-end."""

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | None = None) -> Self:
        """Load DDRNetSlim from a weightfile created by the source DDRNetSlim repository."""
        ddrnetslim_model = DualResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=NUM_CLASSES,
            planes=32,
            spp_planes=128,
            head_planes=64,
            # No need to use aux loss for inference
            augment=False,
        )

        checkpoint_to_load = (
            checkpoint_path
            if checkpoint_path
            else CachedWebModelAsset.from_asset_store(
                MODEL_ID, MODEL_ASSET_VERSION, DEFAULT_WEIGHTS
            ).fetch()
        )

        pretrained_dict = torch.load(
            checkpoint_to_load, map_location=torch.device("cpu"), weights_only=False
        )
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        model_dict = ddrnetslim_model.state_dict()
        pretrained_dict = {
            k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict
        }
        model_dict.update(pretrained_dict)
        ddrnetslim_model.load_state_dict(model_dict)

        ddrnetslim_model.to(torch.device("cpu"))

        return cls(ddrnetslim_model)

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 1024,
        width: int = 2048,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench. Default resolution is 2048x1024
        so this expects an image where width is twice the height.
        """
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

    def get_hub_compile_options(
        self,
        target_runtime: TargetRuntime,
        precision: Precision,
        other_compile_options: str = "",
        device: Device | None = None,
        context_graph_name: str | None = None,
    ) -> str:
        if target_runtime == TargetRuntime.QNN_DLC:
            other_compile_options += " -O2"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

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
        return {"image": [app_to_net_image_inputs(image)[1].numpy()]}
