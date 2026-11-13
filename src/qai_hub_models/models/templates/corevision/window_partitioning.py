# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def window_partition_5d(
    x: torch.Tensor, window_size: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    ---
    Lifted from segment_anything.modeling.image_encoder.window_partition
    Modified by Qualcomm to work in 5D rather than 6D.
    ---

    Partition into non-overlapping windows with padding if needed.

    Parameters
    ----------
    x
        Input tokens with [B, H, W, C].
    window_size
        Window size.

    Returns
    -------
    windows : torch.Tensor
        Windows after partition with [B * num_windows, window_size, window_size, C].
    padded_hw : tuple[int, int]
        Padded height and width before partition (Hp, Wp).
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # -- Begin Qualcomm Modification --
    x = x.reshape(B * Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 2, 1, 3, 4).contiguous().view(-1, window_size, window_size, C)
    )
    # -- End Qualcomm Modification --
    return windows, (Hp, Wp)


def window_unpartition_5d(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: tuple[int, int],
    hw: tuple[int, int],
) -> torch.Tensor:
    """
    ---
    Lifted from segment_anything.modeling.image_encoder.window_unpartition
    Modified by Qualcomm to work in 5D rather than 6D.
    ---

    Window unpartition into original sequences and removing padding.

    Parameters
    ----------
    windows
        Input tokens with [B * num_windows, window_size, window_size, C].
    window_size
        Window size.
    pad_hw
        Padded height and width (Hp, Wp).
    hw
        Original height and width (H, W) before padding.

    Returns
    -------
    x : torch.Tensor
        Unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    # -- Begin Qualcomm Modification --
    x = windows.reshape(
        B * Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, Hp, Wp, -1)
    # -- End Qualcomm Modification --

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def window_reverse_optimized(
    self: Any, windows: torch.Tensor, H: int, W: int
) -> torch.Tensor:
    """
    Reverse window partition to original image.

    Parameters
    ----------
    self
        The instance of the class containing window_size attribute.
    windows
        Windows of shape (num_windows*B, window_size, window_size, C).
    H
        Height of image.
    W
        Width of image.

    Returns
    -------
    output : torch.Tensor
        Output tensor of shape (B, H, W, C).
    """
    window_size: int = self.window_size
    # optimization
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    num_channels = windows.shape[-1]
    windows = windows.view(
        -1, W // window_size, window_size, window_size * num_channels
    )
    return windows.permute(0, 2, 1, 3).contiguous().view(-1, H, W, num_channels)


def window_partition_optimized(self: Any, x: torch.Tensor) -> torch.Tensor:
    """
    Partition image into windows.

    Parameters
    ----------
    self
        The instance of the class containing window_size attribute.
    x
        Input tensor of shape (B, H, W, C).

    Returns
    -------
    windows : torch.Tensor
        Windows of shape (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    window_size: int = self.window_size
    # optimization
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    input_feature = x.view(
        B * H // window_size,
        window_size,
        W // window_size,
        window_size * C,
    )
    windows = input_feature.permute(0, 2, 1, 3).contiguous()
    return windows.view(-1, window_size, window_size, C)


def WindowMSA_forward_optimized(
    self: Any, x: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Optimized forward pass for Window Multi-head Self-Attention.

    Parameters
    ----------
    self
        The instance of the class containing window_size, num_heads, qkv, scale,
        relative_position_bias_table, relative_position_index, softmax, attn_drop,
        proj, and proj_drop attributes.
    x
        Input features with shape of (num_windows*B, N, C).
    mask
        Mask with shape of (num_windows, Wh*Ww, Wh*Ww), value should be
        between (-inf, 0].

    Returns
    -------
    output : torch.Tensor
        Output with shape of (num_windows*B, N, C).
    """
    B, N, C = x.shape

    # optimization
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3 * self.num_heads, C // self.num_heads)
        .permute(0, 2, 1, 3)
    )
    q, k, v = (
        qkv[:, 0 : 1 * self.num_heads],
        qkv[:, 1 * self.num_heads : 2 * self.num_heads],
        qkv[:, 2 * self.num_heads : 3 * self.num_heads],
    )

    q = q * self.scale
    attn = q @ k.transpose(-2, -1)

    relative_position_bias = self.relative_position_bias_table[
        self.relative_position_index.view(-1)
    ].view(
        self.window_size[0] * self.window_size[1],
        self.window_size[0] * self.window_size[1],
        -1,
    )  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(
        2, 0, 1
    ).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        # optimization
        attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
            1
        ).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = attn + mask.unsqueeze(1).unsqueeze(0).repeat(B // nW, 1, 1, 1, 1).view(
            -1, 1, N, N
        )
    attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    return self.proj_drop(x)
