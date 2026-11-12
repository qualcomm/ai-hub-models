# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.grootn17 import MODEL_ID, Model
from qai_hub_models.models.grootn17.app import build_app
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.dataset.lerobot_episode_loader import (
    LeRobotEpisodeLoader,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.data.dataset.sharded_single_step_dataset import (
    extract_step_data,
)
from qai_hub_models.models.grootn17.model import (
    DEFAULT_DATASET_ASSET,
    load_checkpoint,
)
from qai_hub_models.models.grootn17.utils import concatenate_action_modalities
from qai_hub_models.utils.args import (
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.evaluate.helpers import EvalMode


def _plot_action_chunks(
    pred_concat: np.ndarray,
    gt_concat: np.ndarray,
    modality_keys: list[str],
    action_dims: list[int],
    output_dir: Path,
    eval_mode: EvalMode,
) -> None:
    """Plot predicted vs ground-truth action chunks side-by-side per DoF."""
    total_dof = sum(action_dims)
    time_steps = np.arange(pred_concat.shape[0])  # action_horizon

    dof_labels: list[str] = []
    dof_labels.extend(
        f"{key} [{d}]"
        for key, dim in zip(modality_keys, action_dims, strict=True)
        for d in range(dim)
    )

    n_cols = 2
    n_rows = (total_dof + n_cols - 1) // n_cols
    mode_label = eval_mode.value.upper()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), squeeze=False)
    fig.suptitle(
        f"Predicted vs Ground Truth Action Chunk  [{mode_label}]",
        fontsize=13,
        fontweight="bold",
    )

    for dof_idx in range(total_dof):
        row, col = divmod(dof_idx, n_cols)
        ax = axes[row][col]
        ax.plot(
            time_steps,
            gt_concat[:, dof_idx],
            label="Ground Truth",
            color="steelblue",
            linewidth=1.5,
        )
        ax.plot(
            time_steps,
            pred_concat[:, dof_idx],
            label="Predicted",
            color="darkorange",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title(f"DoF {dof_idx}  —  {dof_labels[dof_idx]}", fontsize=9)
        ax.set_xlabel("Action Horizon (t)")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7)

    # Hide unused subplots (when total_dof is odd)
    for dof_idx in range(total_dof, n_rows * n_cols):
        row, col = divmod(dof_idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / f"{MODEL_ID}_demo_action_chunk_plot_{mode_label}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Action chunk plot saved at {plot_path}")


# Entry point
def main() -> None:
    # Argument parsing
    parser = get_model_cli_parser(Model)
    parser = get_on_device_demo_parser(
        parser,
        add_output_dir=True,
        supported_precisions={Precision.float},
        available_target_runtimes=[TargetRuntime.QNN_CONTEXT_BINARY],
        default_device="Samsung Galaxy S25 (Family)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to LeRobot-format dataset directory. Downloads the default DROID sample if omitted.",
    )
    args = parser.parse_args()
    validate_on_device_demo_args(args, MODEL_ID)

    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(f"./build/{MODEL_ID}/demo")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    policy = load_checkpoint(
        checkpoint=args.checkpoint,
        embodiment_tag=args.embodiment_tag,
        device=args.host_device,
    )

    # Build GrootApp
    app = build_app(
        policy,
        args.eval_mode,
        device=args.device,
        hub_model_id=args.hub_model_id,
        host_device=args.host_device,
    )

    # Load dataset
    modality_configs = policy.get_modality_config()
    dataset_path = (
        args.dataset_path
        if args.dataset_path is not None
        else str(DEFAULT_DATASET_ASSET.fetch(extract=True))
    )
    dataset = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        video_backend="ffmpeg",
    )
    # Use episode 2, step 0 for the demo.
    episode_data = dataset[2]
    action_horizon = app.config.action_horizon
    if action_horizon > len(episode_data):
        raise ValueError(
            f"Step 0 does not contain a complete {action_horizon}-step action window in an episode with {len(episode_data)} steps."
        )
    step_data = extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_configs,
        embodiment_tag=policy.embodiment_tag,
        allow_padding=False,
    )

    # Run inference
    pred_concat = app.predict_action_tensor(step_data).numpy()[0]

    # Evaluate predictions
    layout = app.config.action_layout
    gt_concat = concatenate_action_modalities(
        step_data.actions, layout, has_batch_axis=False
    )[0]

    rmse = np.sqrt(np.mean((gt_concat - pred_concat) ** 2))
    print(
        f"Action RMSE over {action_horizon}-step horizon "
        f"[{args.eval_mode.value.upper()}]: {rmse:.6f}"
    )

    _plot_action_chunks(
        pred_concat,
        gt_concat,
        list(layout.modality_keys),
        list(layout.action_dims),
        output_dir,
        args.eval_mode,
    )


if __name__ == "__main__":
    main()
