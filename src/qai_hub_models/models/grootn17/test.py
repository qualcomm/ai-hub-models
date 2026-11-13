# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch

from qai_hub_models.models.grootn17.app import build_app
from qai_hub_models.models.grootn17.constants import DEFAULT_EMBODIMENT_TAG
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.dataset.lerobot_episode_loader import (
    LeRobotEpisodeLoader,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.dataset.sharded_single_step_dataset import (
    extract_step_data,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.types import (
    VLAStepData,
)
from qai_hub_models.models.grootn17.model import (
    DEFAULT_DATASET_ASSET,
    load_checkpoint,
)
from qai_hub_models.utils.evaluate.helpers import EvalMode


def test_one_sample() -> None:
    """
    Run a single forward pass and assert:
    1. App-path and direct-policy-path produce close action predictions.
    2. Predicted actions are close to ground truth (RMSE sanity check).
    """
    # PyTorch exposes a writable bool through a ContextProp descriptor.
    torch.backends.mkldnn.enabled = False  # type: ignore[assignment]

    episode_index, step_index = 2, 0
    policy = load_checkpoint(
        checkpoint="DEFAULT",
        embodiment_tag=DEFAULT_EMBODIMENT_TAG,
        device="cpu",
    )
    modality_configs = policy.get_modality_config()

    dataset = LeRobotEpisodeLoader(
        dataset_path=str(DEFAULT_DATASET_ASSET.fetch(extract=True)),
        modality_configs=modality_configs,
        video_backend="ffmpeg",
    )
    episode_data = dataset[episode_index]
    step_data: VLAStepData = extract_step_data(
        episode_data,
        step_index=step_index,
        modality_configs=modality_configs,
        embodiment_tag=policy.embodiment_tag,
        allow_padding=False,
    )

    cpu_rng_state = torch.get_rng_state()

    # Direct policy path
    observation = {
        "video": {
            key: np.stack([np.asarray(frame) for frame in frames])[None].astype(
                np.uint8
            )
            for key, frames in step_data.images.items()
        },
        "state": {
            key: np.asarray(values, dtype=np.float32)[None]
            for key, values in step_data.states.items()
        },
        "language": {
            key: [[step_data.text]]
            for key in modality_configs["language"].modality_keys
        },
    }
    torch.manual_seed(42)
    pred_actions_policy, _ = policy.get_action(observation)

    # App path
    app = build_app(policy, EvalMode.FP, host_device="cpu")
    torch.set_rng_state(cpu_rng_state)
    pred_actions_app = app.predict_action_chunk(step_data)

    modality_keys = app.config.action_layout.modality_keys

    # Parity check — app vs policy (both at horizon step 0)
    pred_app_concat = np.concatenate(
        [pred_actions_app[key][0, 0] for key in modality_keys], axis=-1
    ).astype(np.float64)
    pred_policy_concat = np.concatenate(
        [pred_actions_policy[key][0, 0] for key in modality_keys], axis=-1
    ).astype(np.float64)

    assert pred_app_concat.shape == pred_policy_concat.shape, (
        f"Shape mismatch: app={pred_app_concat.shape}, policy={pred_policy_concat.shape}"
    )
    np.testing.assert_allclose(
        pred_app_concat,
        pred_policy_concat,
        rtol=0.01,
        atol=0.01,
        err_msg=f"App and native N1.7 first-step predictions differ for episode {episode_index}, step {step_index}.",
    )

    for key in modality_keys:
        np.testing.assert_allclose(
            pred_actions_app[key],
            pred_actions_policy[key],
            rtol=0.01,
            atol=0.01,
            err_msg=f"App and native N1.7 action chunks differ for {key}.",
        )

    # Ground-truth sanity check: RMSE of the first horizon step.
    gt_concat = np.concatenate(
        [step_data.actions[key] for key in modality_keys], axis=-1
    )
    rmse = float(np.sqrt(np.mean((pred_app_concat - gt_concat[0]) ** 2)))
    assert rmse < 0.1, f"App vs GT RMSE {rmse:.6f} exceeds threshold 0.1"

    print("PASS: test_one_sample")
