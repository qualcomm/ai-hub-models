# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from typing_extensions import Self

from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_torch,
)
from qai_hub_models.utils.base_model import BaseModel
from qai_hub_models.utils.image_processing import normalize_image_torchvision
from qai_hub_models.utils.input_spec import InputSpec

MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1
ACT_SOURCE_REPOSITORY = "https://github.com/tonyzhaozh/act.git"
ACT_SOURCE_REPO_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"

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
        with SourceAsRoot(
            ACT_SOURCE_REPOSITORY,
            ACT_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ) as repo_path:
            sys.path.append(os.path.join(repo_path, "detr"))
            from detr.models.detr_vae import build
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

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = 480,
        width: int = 640,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub Workbench.
        """
        return {
            "qpos": ((batch_size, 14), "float32"),
            "image": ((batch_size, 3, height, width), "float32"),
        }

    @staticmethod
    def get_output_names() -> list[str]:
        return ["actions"]
