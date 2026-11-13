# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch


def pixel_shuffle(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """Rank-≤5 equivalent of ``F.pixel_shuffle``.

    ``F.pixel_shuffle`` decomposes into a rank-6 intermediate
    ``(b, C, rh, rw, H, W)`` during TFLite/QNN lowering, triggering a DMA
    execution failure on the HTP DSP.  This version folds ``b*C`` into a
    single dim so the peak intermediate stays at rank 5
    ``(b*C, rh, rw, H, W)``, which HTP handles correctly.

    Parameters
    ----------
    x:
        ``(b, C*r^2, H, W)`` input tensor.
    upscale_factor:
        Spatial upscale factor ``r``.

    Returns
    -------
    torch.Tensor
        ``(b, C, H*r, W*r)`` output tensor.
    """
    b, c_r2, h, w = x.shape
    r = upscale_factor
    c = c_r2 // (r * r)
    x = x.reshape(b * c, r, r, h, w)  # (b*C, rh, rw, H, W)  rank 5
    x = x.permute(0, 3, 1, 4, 2).contiguous()  # (b*C, H, rh, W, rw)  rank 5
    return x.reshape(b, c, h * r, w * r)  # (b, C, H*r, W*r)     rank 4


# Keyed by (h, w, dtype, device); not populated during torch.export (fake tensors
# have no real device/storage and must not be cached as constants).
_PIXEL_GRID_CACHE: dict[
    tuple[int, int, torch.dtype, torch.device], tuple[torch.Tensor, torch.Tensor]
] = {}


def _make_pixel_grid(
    h: int, w: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid_y, grid_x) pixel-index grids, cached for eager reuse."""
    key = (h, w, dtype, device)
    if key not in _PIXEL_GRID_CACHE and not torch.compiler.is_compiling():
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=dtype, device=device),
            torch.arange(0, w, dtype=dtype, device=device),
            indexing="ij",
        )
        _PIXEL_GRID_CACHE[key] = (grid_y, grid_x)
    if key in _PIXEL_GRID_CACHE:
        return _PIXEL_GRID_CACHE[key]
    # Compiling path: fake tensors can't be cached, so always compute inline.
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=dtype, device=device),
        torch.arange(0, w, dtype=dtype, device=device),
        indexing="ij",
    )
    return grid_y, grid_x


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Gather-based nearest-neighbor flow warp; HTP-safe replacement for ``F.grid_sample``.

    ``F.grid_sample`` (regardless of interpolation mode) uses a DMA-backed
    kernel path on the HTP DSP that raises err 1100 at inference time.  This
    implementation avoids ``grid_sample`` entirely: it rounds flow offsets to
    integer pixel positions and uses ``Tensor.gather`` to sample, which maps
    to basic memory ops with no special DSP kernel requirements.

    The ``interpolation`` and ``align_corners`` parameters are accepted for API
    compatibility with callers that also work on non-HTP backends but are
    otherwise unused — the warp is always nearest-neighbor.

    Parameters
    ----------
    x:
        ``(1, C, H, W)`` feature map to warp.
    flow:
        ``(1, H, W, 2)`` optical flow in pixel units ``(dx, dy)``.
    interpolation:
        Ignored; kept for API compatibility.
    padding_mode:
        ``'zeros'`` (default) — out-of-bounds samples return 0.
        ``'border'`` — out-of-bounds positions clamp to the nearest edge pixel.
    align_corners:
        Ignored; kept for API compatibility.

    Returns
    -------
    torch.Tensor
        ``(1, C, H, W)`` warped feature map.
    """
    b, c, h, w = x.shape
    grid_y, grid_x = _make_pixel_grid(h, w, x.dtype, x.device)
    src_x = (grid_x.unsqueeze(0) + flow[..., 0]).round().to(torch.int32)  # (1, H, W)
    src_y = (grid_y.unsqueeze(0) + flow[..., 1]).round().to(torch.int32)  # (1, H, W)
    if padding_mode == "zeros":
        valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    src_x = src_x.clamp(0, w - 1)
    src_y = src_y.clamp(0, h - 1)
    flat_idx = (src_y * w + src_x).view(b, 1, h * w).expand(b, c, h * w)
    out = x.view(b, c, h * w).gather(2, flat_idx).view(b, c, h, w)
    if padding_mode == "zeros":
        return out.masked_fill(~valid.unsqueeze(1), 0.0)
    return out
