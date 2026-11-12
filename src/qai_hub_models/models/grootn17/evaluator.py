# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MetricMetadata


class LeRobotEvaluator(BaseEvaluator):
    """
    Measures Action RMSE between predicted and ground-truth action chunks from a LeRobot dataset

    Each call to add_batch() accepts NumPy arrays or Torch tensors, optionally
    wrapped in a one-element tuple/list, with shape
    (action_horizon, total_dof) or (batch_size, action_horizon, total_dof).

    RMSE is computed over all accumulated elements.
    """

    def __init__(
        self,
        dof_slices: list[tuple[str, slice]] | None = None,
    ) -> None:
        self._sum_sq_error: float = 0.0
        self._n_elements: int = 0
        self._n_chunks: int = 0
        self._n_action_rows: int = 0

        # Per-modality-group tracking
        self._dof_slices: list[tuple[str, slice]] = dof_slices or []
        self._group_sum_sq: list[float] = [0.0] * len(self._dof_slices)
        self._group_n_elem: list[int] = [0] * len(self._dof_slices)

    @staticmethod
    def _to_array(value: Any) -> np.ndarray:
        if isinstance(value, (tuple, list)):
            if len(value) != 1:
                raise ValueError(
                    "Expected a tuple/list containing exactly one evaluator value, "
                    f"but received {len(value)} values."
                )
            value = value[0]

        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", dtype=torch.float64).numpy()
        if isinstance(value, np.ndarray):
            return value.astype(np.float64, copy=False)
        raise TypeError(
            "Evaluator inputs must be NumPy arrays or Torch tensors, optionally "
            "wrapped in a one-element tuple."
        )

    # BaseEvaluator interface
    def add_batch(
        self,
        pred_chunk: Any,
        gt_chunk: Any,
    ) -> None:
        pred_arr = self._to_array(pred_chunk)
        gt_arr = self._to_array(gt_chunk)

        if pred_arr.shape != gt_arr.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred_arr.shape} vs gt {gt_arr.shape}"
            )
        if pred_arr.ndim not in (2, 3):
            raise ValueError("Action chunks must have shape [H, D] or [B, H, D].")
        sq_err = (pred_arr - gt_arr) ** 2
        self._sum_sq_error += float(sq_err.sum())
        self._n_elements += sq_err.size
        self._n_chunks += pred_arr.shape[0] if pred_arr.ndim == 3 else 1
        self._n_action_rows += int(np.prod(pred_arr.shape[:-1]))
        # Per-modality-group accumulation — slice last (DOF) axis
        for i, (_, sl) in enumerate(self._dof_slices):
            group_sq = sq_err[..., sl]
            self._group_sum_sq[i] += float(group_sq.sum())
            self._group_n_elem[i] += group_sq.size

    def get_accuracy_score(self) -> float:
        """Returns RMSE (lower is better). Raises if no data was accumulated."""
        if self._n_elements == 0:
            raise RuntimeError(
                "LeRobotEvaluator has no accumulated data. "
                "Ensure add_batch() was called at least once."
            )
        return float(np.sqrt(self._sum_sq_error / self._n_elements))

    def get_group_scores(self) -> list[tuple[str, float]]:
        """
        Per-modality-group RMSE scores
        Returns [(group_name, rmse), ...] in dof_slices order.
        Returns empty list if no dof_slices were provided.
        """
        results = []
        for i, (name, _) in enumerate(self._dof_slices):
            n = self._group_n_elem[i]
            rmse = float(np.sqrt(self._group_sum_sq[i] / n)) if n > 0 else 0.0
            results.append((name, rmse))
        return results

    def formatted_accuracy(self) -> str:
        lines = [
            f"Action RMSE: {self.get_accuracy_score():.6f} "
            f"(over {self._n_chunks} chunks, {self._n_action_rows} action rows)"
        ]
        group_scores = self.get_group_scores()
        if group_scores:
            lines.append("  Per-modality-group RMSE:")
            for name, rmse in group_scores:
                lines.append(f"    [{name}]: {rmse:.6f}")
        return "\n".join(lines)

    def get_metric_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="Action RMSE",
            unit="",
            description="Root mean squared error over decoded mixed physical action coordinates (EEF pose, gripper, and joints)",
            range=(0.0, float("inf")),
            higher_is_better=False,
        )

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._sum_sq_error = 0.0
        self._n_elements = 0
        self._n_chunks = 0
        self._n_action_rows = 0
        self._group_sum_sq = [0.0] * len(self._dof_slices)
        self._group_n_elem = [0] * len(self._dof_slices)
