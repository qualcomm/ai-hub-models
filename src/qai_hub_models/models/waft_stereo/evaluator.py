# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.models.templates.corevision.image_processing import (
    undo_resize_pad,
)
from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import D1_ALL, MetricMetadata

# KITTI D1 threshold: a pixel is an outlier if its disparity error exceeds
# both an absolute floor and a fraction of the ground-truth disparity.
D1_ABS_THRESHOLD = 3.0
D1_REL_THRESHOLD = 0.05


class WaftStereoD1Evaluator(BaseEvaluator):
    """Computes the KITTI D1-all stereo outlier rate."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._bad_count = 0.0
        self._valid_count = 0.0

    def add_batch(
        self,
        output: torch.Tensor,
        gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """
        output: shape [N, 1, H, W], float32, predicted disparity in pixels at
            network resolution.
        gt: (disp_native, scale, pad_left, pad_top) - ground-truth disparity
            at native resolution (values <= 0 or >= 1e3 are invalid and
            excluded) plus the resize_pad params applied to the model's
            input, used to restore `output` to native resolution before
            comparison. Native resolution can vary per sample, so this loops
            over the batch instead of vectorizing.
        """
        disp_native, scale, pad_left, pad_top = gt
        for i in range(output.shape[0]):
            height, width = disp_native[i].shape[-2:]
            scale_i = float(scale[i])
            pred = undo_resize_pad(
                output[i : i + 1],
                (width, height),
                scale_i,
                (int(pad_left[i]), int(pad_top[i])),
            )
            # Disparity is a pixel offset, so unlike RGB values it must be
            # rescaled by the same factor used to resize the spatial dims.
            pred = (pred / scale_i).squeeze(0).squeeze(0)
            gt_i = disp_native[i]
            valid = (gt_i > 0.0) & (gt_i < 1e3)
            error = (pred - gt_i).abs()
            bad = (error > D1_ABS_THRESHOLD) & (error > D1_REL_THRESHOLD * gt_i) & valid
            # Accumulate raw counts rather than a per-image percentage, so
            # images with fewer valid pixels don't get equal weight to ones
            # with more - matches upstream's eval_disp.py.
            self._valid_count += valid.float().sum().item()
            self._bad_count += bad.float().sum().item()

    def get_accuracy_score(self) -> float:
        if self._valid_count == 0.0:
            raise ValueError("get_accuracy_score() called before any add_batch() call.")
        return 100 * self._bad_count / self._valid_count

    def formatted_accuracy(self) -> str:
        if self._valid_count == 0.0:
            raise ValueError("formatted_accuracy() called before any add_batch() call.")
        return f"D1-all: {self.get_accuracy_score():.2f}%"

    def get_metric_metadata(self) -> MetricMetadata:
        return D1_ALL
