# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch


@torch.jit.unused
def _onnx_merge_levels_optimized(
    levels: torch.Tensor, unmerged_results: list[torch.Tensor]
) -> torch.Tensor:
    """
    Replaces torchvision's _onnx_merge_levels for performance optimization.
    Uses gather operation (120ms) instead of scatter_elements (8000ms).

    Parameters
    ----------
    levels
        Tensor of shape (batch_size,) indicating which FPN level each ROI belongs to.
    unmerged_results
        List of feature tensors from different FPN levels, each with shape
        (num_rois_at_level, channels, height, width).

    Returns
    -------
    torch.Tensor
        Merged feature tensor of shape (batch_size, channels, height, width).
    """
    _, _, h, w = unmerged_results[0].shape
    max_batch = levels.shape[0]

    # Pad all results to same batch size
    padded_results = []
    for result in unmerged_results:
        if result.size(0) < max_batch:
            padding = torch.zeros(
                (
                    max_batch - result.size(0),
                    result.size(1),
                    result.size(2),
                    result.size(3),
                )
            )
            padded_result = torch.cat([result, padding], dim=0)
        else:
            padded_result = result
        padded_results.append(padded_result.flatten(2, 3).unsqueeze(0))

    stacked = torch.concat(padded_results, dim=0)

    num_levels = len(unmerged_results)
    # Clamp levels to valid range [0, num_levels-1]
    levels = levels.clamp(0, num_levels - 1)
    one_hot = (levels.unsqueeze(1) == torch.arange(num_levels).unsqueeze(0)).float()

    # Cumsum within each level: [batch_size, num_levels]
    cumsum_per_level = torch.cumsum(one_hot, dim=0)

    # Extract position for each item's level
    position_in_level = (cumsum_per_level * one_hot).sum(dim=1).long() - 1

    # Gather
    res = stacked[levels, position_in_level]

    return res.reshape(max_batch, -1, h, w)
