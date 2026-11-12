# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace

import torch
from qai_hub.client import Device
from typing_extensions import Self

from qai_hub_models import (
    Precision,
    TargetRuntime,
)
from qai_hub_models.models.act.external_repos.act.detr.models.detr_vae import build
from qai_hub_models.models.templates.corevision.image_processing import (
    normalize_image_torchvision,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_torch
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
MODEL_ASSET_VERSION = 1

# Checkpoint is trained using the repo script
ACT_CKPT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "act.ckpt"
)


def remove_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """
    Parameters
    ----------
    state_dict
        A dictionary that stores all the learned parameters of the model.
    prefix
        The word that needs to be removed at start of the keys in state_dict.

    Returns
    -------
    result: dict[str, torch.Tensor]
        A dictionary that stores all the learned parameters of the model with the prefix removed.
    """
    result = {}

    for k, v in state_dict.items():
        tokens = k.split(".")

        if tokens[0] == prefix:
            tokens = tokens[1:]

        key = ".".join(tokens)
        result[key] = v

    return result


class ACT(BaseModel):
    """Exportable ACT model end-to-end."""

    @classmethod
    def from_pretrained(cls, ckpt_path: str | None = None) -> Self:
        """Load ACT trained checkpoint from the specified path."""
        ckpt_path = str(ACT_CKPT.fetch()) if ckpt_path is None else ckpt_path

        # The values assigned in config are obtained by printing 'args' in the below mentioned line of the repo (print(args))
        # https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/detr/models/detr_vae.py#L231
        config = {
            "lr_backbone": 1e-05,
            "masks": False,
            "backbone": "resnet18",
            "dilation": False,
            "hidden_dim": 512,
            "position_embedding": "sine",
            "dropout": 0.1,
            "nheads": 8,
            "camera_names": ["top"],
            "dim_feedforward": 3200,
            "enc_layers": 4,
            "dec_layers": 7,
            "num_queries": 100,
            "pre_norm": False,
        }

        config_namespace = SimpleNamespace(**config)

        model = build(config_namespace)
        state_dict = load_torch(ckpt_path)
        state_dict = remove_prefix(state_dict, "model")
        model.load_state_dict(state_dict)

        return cls(model)

    def forward(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Run ACT model and returns action chunks.

        In the below dimensions, state_dim represents the robot's state at any timestep. It is assigned to 14 because the robot's state is
        represented by 14 numerical values. [LEFT_ARM[7] | RIGHT_ARM[7]]

        The 7 values for each arm represent positions for waist, shoulder, elbow, forearm roll, wrist angle, wrist rotate and gripper point
        respectively.

        Parameters
        ----------
        qpos
            Input tensor of shape (batch_size, state_dim) representing the current state where state_dim = 14.
        image
            Input tensor of shape (batch_size, 3, H, W) representing a RGB image. Range - 0 to 1

        Returns
        -------
        actions:torch.Tensor
            Output tensor of shape (1, num_actions, state_dim) representing the predicted chunk of actions where state_dim = 14.
        """
        image = normalize_image_torchvision(image)
        return self.model(qpos, image.unsqueeze(1), env_state=None)[0]

    def get_input_spec(
        self,
        batch_size: int = 1,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "qpos": TensorSpec(
                shape=(batch_size, 14),
                dtype="float32",
                io_type=IoType.TENSOR,
                description="Robot joint positions [left_arm(7) | right_arm(7)]",
            ),
            "image": TensorSpec(
                shape=(batch_size, 3, height, width),
                dtype="float32",
                io_type=IoType.IMAGE,
                value_range=(0.0, 1.0),
                image_metadata=ImageMetadata(
                    color_format=ColorFormat.RGB,
                ),
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
        if (
            target_runtime == TargetRuntime.TFLITE
            and "--truncate_64bit_tensors" not in other_compile_options
        ):
            other_compile_options += " --truncate_64bit_tensors"
        return super().get_hub_compile_options(
            target_runtime, precision, other_compile_options, device, context_graph_name
        )

    def get_output_spec(self) -> OutputSpec:
        return {
            "actions": TensorSpec(),
        }
