# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from qai_hub_models.models.act.model import (
    ACT_SOURCE_REPO_COMMIT,
    ACT_SOURCE_REPOSITORY,
    MODEL_ASSET_VERSION,
    MODEL_ID,
)
from qai_hub_models.utils.asset_loaders import SourceAsRoot
from qai_hub_models.utils.image_processing import numpy_image_to_torch

with SourceAsRoot(
    ACT_SOURCE_REPOSITORY,
    ACT_SOURCE_REPO_COMMIT,
    MODEL_ID,
    MODEL_ASSET_VERSION,
):
    # Set MuJoCo rendering backend before any MuJoCo operations
    os.environ["MUJOCO_GL"] = "egl"
    from sim_env import BOX_POSE, make_sim_env
    from utils import sample_box_pose, set_seed

"""
The below value for stats can be obtained by following the below steps

1. Run the record_sim_episodes.py in repo to generate and store the dataset. Use the below command
    python3 record_sim_episodes.py \
    --task_name sim_transfer_cube_scripted \
    --dataset_dir <data save dir> \
    --num_episodes 50

2. Use the load_data() in utils.py to get the stats value.
   https://github.com/tonyzhaozh/act/blob/742c753c0d4a5d87076c8f69e5628c79a8cc5488/utils.py#L111
   Eg: _,_,stats,_ = load_data(dataset_dir=<data save dir>, num_episodes=50, camera_names=['top'], batch_size_train=8, batch_size_val=8)
"""

STATS = {
    "qpos_mean": [
        -0.00537633,
        -0.48886874,
        1.0137362,
        -0.00614296,
        -0.5179885,
        1.1218411,
        0.6895667,
        -0.01392809,
        -0.32401496,
        0.48143452,
        0.01457404,
        0.7713909,
        -0.0268569,
        0.70872617,
    ],
    "qpos_std": [
        0.01,
        0.52801144,
        0.19804354,
        0.01640159,
        0.3591711,
        0.5998051,
        0.25531325,
        0.11724292,
        0.5045925,
        0.44841644,
        0.15253916,
        0.30429485,
        0.2432673,
        0.2474613,
    ],
    "action_mean": [
        -0.00541835,
        -0.4803254,
        1.0102423,
        -0.00421979,
        -0.5297607,
        1.1214392,
        0.5875,
        -0.01391168,
        -0.3181911,
        0.47349554,
        0.01695838,
        0.7741717,
        -0.02683119,
        0.59625,
    ],
    "action_std": [
        0.01,
        0.51985204,
        0.19778018,
        0.01628644,
        0.3604608,
        0.59961,
        0.42410523,
        0.11886074,
        0.4958926,
        0.44617707,
        0.15482497,
        0.29403326,
        0.2432429,
        0.38613003,
    ],
}


class ACTApp:
    def __init__(
        self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        self.model = model

    def predict(
        self, *args: Any, **kwargs: Any
    ) -> list[dict[str, np.ndarray]] | np.ndarray:
        # See predict_action_chunks
        return self.predict_action_chunks(*args, **kwargs)

    def predict_action_chunks(
        self,
        episode_len: int,
        raw_output: bool | None = False,
    ) -> list[dict[str, np.ndarray]] | np.ndarray:
        """
        Set the simulated environment, box position and run the ACT model with inputs representing current position
        to predict the next 100 action chunks.

        Parameters
        ----------
        episode_len
            Number of steps in the simulation. Range - 1 to 400
        raw_output
            Used for testing the output

        Returns
        -------
        image_list: list[dict[str, np.ndarray]] | np.ndarray
            Output list of dict with length equal to episode_len. Each dict has 3 keys, each representing different angles in the output
            video generated. Size of each value for key in dict is (height, width, channel)
        """
        set_seed(0)

        # creates and initializes a simulated environment
        env = make_sim_env("sim_transfer_cube_scripted")

        # returns position of the cube to be transferred
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()

        # query frequency determines, how often the model needs to be queried to predict future actions.
        image_list, query_frequency = [], 100

        for t in range(episode_len):
            # The value is updated in each iteration with the recent predicted actions
            obs = ts.observation
            image_list.append(obs["images"])

            # Running the model once will predict the next query_frequency number of actions
            if t % query_frequency == 0:
                image = np.stack([obs["images"]["top"]], axis=0)
                qpos = obs["qpos"]
                qpos_numpy = (
                    qpos - np.array(STATS["qpos_mean"], dtype=np.float32)
                ) / np.array(STATS["qpos_std"], dtype=np.float32)
                qpos_torch = torch.from_numpy(qpos_numpy).float().unsqueeze(0)
                image_torch = numpy_image_to_torch(image)
                actions_torch = self.model(qpos_torch, image_torch)

                actions = (
                    actions_torch.numpy()
                    * np.array(STATS["action_std"], dtype=np.float32)
                ) + np.array(STATS["action_mean"], dtype=np.float32)

                if raw_output:
                    return actions
            action = actions[0, t % query_frequency]

            # updates env with the predicted action
            target_qpos = action
            ts = env.step(target_qpos)

        return image_list
