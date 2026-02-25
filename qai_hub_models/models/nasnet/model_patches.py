# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import torch


def CellStem1_forward(
    self: Any, x_conv0: torch.Tensor, x_stem_0: torch.Tensor
) -> torch.Tensor:
    """
    Parameters
    ----------
    self
        The CellStem1 module instance.
    x_conv0
        Input tensor of shape (B, C, H, W).
    x_stem_0
        Input tensor of shape (B, C, H, W).

    Returns
    -------
    x_out : torch.Tensor
        Output tensor of shape (B, C, H, W).
    """
    x_left = self.conv_1x1(x_stem_0)

    x_relu = self.act(x_conv0)

    # -- Begin Qualcomm Change
    # path 1
    # AvgPool takes more time for quantized variants.
    # Replaced it with manual implementation for faster execution.
    x_relu_1 = x_relu[:, :, ::2, ::2]
    x_path1 = self.path_1[1](x_relu_1)

    # path 2
    # CropandResize node is not supported in QNN.
    # Replaced it with manual implementation.
    h, w = x_relu.shape[2:]
    x_relu_list = x_relu.split(1, dim=2)
    x_relu = torch.concat((*x_relu_list[1:h], x_relu_list[0]), dim=2)
    x_relu_list = x_relu.split(1, dim=3)
    x_relu = torch.concat((*x_relu_list[1:w], x_relu_list[0]), dim=3)

    # AvgPool takes more time for quantized variants.
    # Replaced it with manual implementation for faster execution.
    x_relu = x_relu[:, :, ::2, ::2]
    x_path2 = self.path_2[2](x_relu)
    # -- End Qualcomm Change

    # final path
    x_right = self.final_path_bn(torch.cat([x_path1, x_path2], 1))

    x_comb_iter_0_left = self.comb_iter_0_left(x_left)
    x_comb_iter_0_right = self.comb_iter_0_right(x_right)
    x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

    x_comb_iter_1_left = self.comb_iter_1_left(x_left)
    x_comb_iter_1_right = self.comb_iter_1_right(x_right)
    x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

    x_comb_iter_2_left = self.comb_iter_2_left(x_left)
    x_comb_iter_2_right = self.comb_iter_2_right(x_right)
    x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

    x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
    x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

    x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
    x_comb_iter_4_right = self.comb_iter_4_right(x_left)
    x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

    return torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)


def FirstCell_forward(self: Any, x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    self
        The FirstCell module instance.
    x
        Input tensor of shape (B, C, H, W).
    x_prev
        Input tensor of shape (B, C, H, W).

    Returns
    -------
    x_out : torch.Tensor
        Output tensor of shape (B, C, H, W).
    """
    x_relu = self.act(x_prev)

    # -- Begin Qualcomm Change
    # path 1
    # AvgPool takes more time for quantized variants.
    # Replaced it with manual implementation for faster execution.
    x_relu_1 = x_relu[:, :, ::2, ::2]
    x_path1 = self.path_1[1](x_relu_1)

    # path 2
    # CropandResize node is not supported in QNN.
    # Replaced it with manual implementation.
    h, w = x_relu.shape[2:]
    x_relu_list = x_relu.split(1, dim=2)
    x_relu = torch.concat((*x_relu_list[1:h], x_relu_list[0]), dim=2)
    x_relu_list = x_relu.split(1, dim=3)
    x_relu = torch.concat((*x_relu_list[1:w], x_relu_list[0]), dim=3)

    # AvgPool takes more time for quantized variants.
    # Replaced it with manual implementation for faster execution.
    x_relu = x_relu[:, :, ::2, ::2]
    x_path2 = self.path_2[2](x_relu)
    # -- End Qualcomm Change

    x_left = self.final_path_bn(torch.cat([x_path1, x_path2], 1))
    x_right = self.conv_1x1(x)

    x_comb_iter_0_left = self.comb_iter_0_left(x_right)
    x_comb_iter_0_right = self.comb_iter_0_right(x_left)
    x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

    x_comb_iter_1_left = self.comb_iter_1_left(x_left)
    x_comb_iter_1_right = self.comb_iter_1_right(x_left)
    x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

    x_comb_iter_2_left = self.comb_iter_2_left(x_right)
    x_comb_iter_2 = x_comb_iter_2_left + x_left

    x_comb_iter_3_left = self.comb_iter_3_left(x_left)
    x_comb_iter_3_right = self.comb_iter_3_right(x_left)
    x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

    x_comb_iter_4_left = self.comb_iter_4_left(x_right)
    x_comb_iter_4 = x_comb_iter_4_left + x_right

    return torch.cat(
        [
            x_left,
            x_comb_iter_0,
            x_comb_iter_1,
            x_comb_iter_2,
            x_comb_iter_3,
            x_comb_iter_4,
        ],
        1,
    )
