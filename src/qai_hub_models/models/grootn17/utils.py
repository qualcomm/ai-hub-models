# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from qai_hub_models.configs.tensor_spec import TensorSpec
from qai_hub_models.utils.input_spec import InputSpec


@dataclass(frozen=True)
class ActionLayout:
    """Checkpoint-owned order and dimensions of decoded physical actions."""

    modality_keys: tuple[str, ...]
    action_dims: tuple[int, ...]
    action_horizon: int

    def __post_init__(self) -> None:
        if (
            not self.modality_keys
            or len(set(self.modality_keys)) != len(self.modality_keys)
            or len(self.modality_keys) != len(self.action_dims)
            or any(dim <= 0 for dim in self.action_dims)
            or self.action_horizon <= 0
        ):
            raise ValueError(
                "Action layout requires unique keys, positive dimensions and horizon."
            )

    @property
    def dof_slices(self) -> list[tuple[str, slice]]:
        offset = 0
        slices = []
        for key, dim in zip(self.modality_keys, self.action_dims, strict=True):
            slices.append((key, slice(offset, offset + dim)))
            offset += dim
        return slices


def concatenate_action_modalities(
    actions: Mapping[str, np.ndarray],
    layout: ActionLayout,
    *,
    has_batch_axis: bool,
) -> np.ndarray:
    """Validate decoded actions or raw GT and return float32 [1, H, D]."""
    parts = []
    for key, dim in zip(layout.modality_keys, layout.action_dims, strict=True):
        if key not in actions:
            raise KeyError(f"Missing action modality '{key}'.")
        values = np.asarray(actions[key])
        if has_batch_axis:
            if values.ndim < 2 or values.shape[0] != 1:
                raise ValueError(
                    f"Predicted action modality '{key}' must have batch size 1; "
                    f"received shape {values.shape}."
                )
            values = values[0]
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != dim:
            raise ValueError(
                f"Action modality '{key}' must have shape (H, {dim}); "
                f"received shape {values.shape}."
            )
        if values.shape[0] < layout.action_horizon:
            raise ValueError(
                f"Action modality '{key}' has {values.shape[0]} steps, fewer than "
                f"the configured action horizon {layout.action_horizon}."
            )
        parts.append(values[: layout.action_horizon])
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)[None]


def get_vit_input_spec(num_patches: int, patch_dim: int) -> InputSpec:
    return {"pixel_values": TensorSpec(shape=(num_patches, patch_dim), dtype="float32")}


def get_llm_input_spec(
    seq_len: int,
    hidden_size: int,
    num_deepstack: int,
    num_vis_tokens: int,
    batch_size: int = 1,
) -> InputSpec:
    spec = {
        "inputs_embeds": TensorSpec(
            shape=(batch_size, seq_len, hidden_size), dtype="float32"
        ),
        "llm_attention_mask": TensorSpec(
            shape=(batch_size, 1, seq_len, seq_len), dtype="float32"
        ),
    }
    if num_deepstack:
        spec["visual_pos_masks"] = TensorSpec(
            shape=(batch_size, seq_len), dtype="float32"
        )
        spec.update(
            {
                f"deepstack_{i}": TensorSpec(
                    shape=(num_vis_tokens, hidden_size), dtype="float32"
                )
                for i in range(num_deepstack)
            }
        )
    return spec


def get_vlm_proj_input_spec(
    seq_len: int, hidden_size: int, batch_size: int = 1
) -> InputSpec:
    return {
        "backbone_features": TensorSpec(
            shape=(batch_size, seq_len, hidden_size), dtype="float32"
        ),
        "vlm_attention_mask": TensorSpec(
            shape=(batch_size, 1, seq_len, seq_len), dtype="float32"
        ),
    }


def get_dit_input_spec(
    seq_len: int,
    state_dim: int,
    action_horizon: int,
    action_dim: int,
    num_heads: int,
    head_dim: int,
    num_kv_pairs: int,
    num_steps: int,
    batch_size: int = 1,
) -> InputSpec:
    spec = {
        "state": TensorSpec(shape=(batch_size, 1, state_dim), dtype="float32"),
        "actions": TensorSpec(
            shape=(batch_size, action_horizon, action_dim), dtype="float32"
        ),
        "image_mask": TensorSpec(shape=(batch_size, seq_len), dtype="float32"),
        "backbone_mask": TensorSpec(shape=(batch_size, seq_len), dtype="float32"),
    }
    for kind in ("keys", "values"):
        for i in range(num_kv_pairs):
            spec[f"vlm_proj_{kind}_{i}"] = TensorSpec(
                shape=(batch_size, num_heads, seq_len, head_dim), dtype="float32"
            )
    spec["timestep_indices"] = TensorSpec(
        shape=(batch_size, num_steps),
        dtype="int32",
        value_range=(0, num_steps - 1),
        description=(
            "Indices into the fixed timestep schedule; normally [0, 1, 2, 3]. "
            "Runtime selectors prevent weights from being baked separately for each unrolled step."
        ),
    )
    return spec
