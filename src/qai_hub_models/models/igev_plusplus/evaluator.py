# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import EPE as EPE_METRIC
from qai_hub_models.utils.metrics import MetricMetadata


class DisparityEvaluator(BaseEvaluator):
    """Evaluator for stereo disparity estimation.

    Protocol: 500-pair FlyingThings3D TEST subset, GT disparity rescaled
    by W/orig_w. EPE is measured on rescaled GT and is NOT comparable to
    paper results. D1 uses a 3px threshold on rescaled GT and is not
    comparable to full-resolution D1.

    Metrics:
      - EPE  : mean absolute disparity error over valid pixels (lower is better)
      - D1   : percentage of valid pixels with |error| > 3px (lower is better)

    Valid pixels: GT >= 0 (invalid pixels marked -1 in dataset).
    Uses disp > 0 cutoff; empirically all FlyingThings3D disparities > 1.0.
    """

    BAD_THRESHOLD: float = 3.0

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._epe_sum: float = 0.0
        self._d1_bad: int = 0
        self._d1_total: int = 0
        self._count: int = 0

    def add_batch(
        self,
        output: torch.Tensor,
        gt: torch.Tensor,
    ) -> None:
        """
        Parameters
        ----------
        output
            Predicted disparity [B, 1, H, W] or [B, H, W].
        gt
            Ground-truth disparity [B, H, W] float32.
        """
        if output.dim() == 4:
            output = output.squeeze(1)

        pred = output.detach().cpu().float()
        gt = gt.detach().cpu().float()

        # Invalid pixels are marked -1 in dataset.py (uses disp > 0; empirically
        # all FlyingThings3D disparities > 1.0, so this matches upstream valid_gt >= 0.5).
        valid = gt >= 0

        epe = torch.abs(pred - gt)

        for b in range(pred.shape[0]):
            v = valid[b]
            if v.sum() == 0:
                continue
            epe_vals = epe[b][v]
            self._epe_sum += epe_vals.mean().item()
            # Accumulate raw counts for correct pooled D1 (matches upstream).
            self._d1_bad += int((epe_vals > self.BAD_THRESHOLD).sum().item())
            self._d1_total += int(v.sum().item())
            self._count += 1

    def get_accuracy_score(self) -> float:
        """Mean per-image end-point error in pixels (lower is better)."""
        if self._count == 0:
            return float("nan")
        return self._epe_sum / self._count

    def epe(self) -> float:
        return self.get_accuracy_score()

    def d1(self) -> float:
        if self._d1_total == 0:
            return float("nan")
        return 100.0 * self._d1_bad / self._d1_total

    def formatted_accuracy(self) -> str:
        return f"EPE={self.epe():.4f}px  D1={self.d1():.2f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return EPE_METRIC
