# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
On-device (QNN/QAIRT) compatibility adaptations for GroundingDINO-T.

Upstream GroundingDINO (https://github.com/IDEA-Research/GroundingDINO) uses
several ops and patterns that either fail to export to ONNX, aren't
supported by QNN, or get mis-optimized by the QAIRT compiler once compiled
and linked for the NPU. Each adaptation below (``P1``-``P13``) was found by
comparing this model's ONNX graph and QAIRT compile/link logs against a
reference (piqaro-optimized) graph and reading the compiler's own error
messages -- e.g. a crash naming ``StridedSlice(BOOL_8)`` or
``Batch_MatMul_Bias_w_scale.B`` points directly at the offending op pattern,
and an op-count diff (Erf/MatMul/Conv counts) against the reference graph
shows which rewrite closes the gap. All of the adaptations here are pure
PyTorch-level rewrites applied before/at export -- no changes to model
outputs or accuracy are intended. The ``Batch_MatMul_Bias_w_scale.B`` issue is fixed in
QAIRT 2.49.

P1  Position encoding: replace ``cumsum`` over the padding mask with
    ``torch.arange`` (``PositionEmbeddingSineHW``). ``cumsum`` produces a
    dynamic op in the exported ONNX graph; ``arange`` is a constant-shape
    equivalent since the mask shape is static per input resolution. Only
    equivalent to ``cumsum(~mask)`` if padding is a contiguous right/bottom
    rectangle -- true here since every caller always passes an all-zero
    mask: this model runs at one fixed resolution and never treats any
    position as padding, even the black letterbox bars
    ``letterbox_resize_pad_pil`` can add for non-4:3 images (those pixels
    are attended to as ordinary content, not masked out). Also
    substitutes a minimal ``NestedTensor`` (tensors + mask, no behavior
    beyond that) for upstream's, which guards an ``if mask == "auto":``
    string comparison that ``torch.export``'s symbolic tracer can't
    represent. ``SwinTransformer.forward`` (imported unmodified from
    upstream) constructs ``NestedTensor(...)`` by looking up the name in
    its own module's globals, so ``apply_patches()`` substitutes it there
    (same technique as P8/P10).

P2  ``get_valid_ratio``: replace boolean mask inversion (``~mask``) with
    float32 arithmetic (``1.0 - mask``), avoiding a bool ``StridedSlice`` op.
    ``Transformer`` is imported unmodified from upstream; its ``forward()``
    calls this as a bound method (``self.get_valid_ratio``), so
    ``apply_patches()`` assigns our version onto the class directly.

P3  ``gen_encoder_output_proposals``: full float32 reimplementation of
    upstream's proposal generation, avoiding ``StridedSlice(BOOL_8)`` and an
    unsupported ``Logit(FLOAT_32)`` op. Called directly from P4's
    ``Transformer_forward`` (which replaces ``Transformer.forward`` wholesale,
    so there's no longer a module-level name for ``Transformer.forward()``
    to look up).

P4  ``Transformer.forward`` is overridden (``Transformer_forward``) to also
    compute ``spatial_shapes`` as a plain Python ``list[tuple[int, int]]``,
    alongside the existing tensor. ``_get_reference_points``/
    ``_gen_encoder_output_proposals``/``_multi_scale_deformable_attn_pytorch``
    read the list directly instead of unpacking per-level ``(H_, W_)`` from
    the tensor: once a Python int is round-tripped through
    ``torch.as_tensor()``, reading it back per-element produces an unbacked
    symbolic int that ``torch.export``'s Dynamo tracer can't resolve into a
    concrete size for ops like ``torch.linspace(steps=...)``
    (``GuardOnDataDependentSymNode``). The tensor itself is still built and
    used for genuine tensor ops (``level_start_index``, ``MSDeformAttn``'s
    ``offset_normalizer``) that don't hit this limitation.

    This model only ever runs at one fixed resolution, so the list is the
    same on every call: it's computed once (``_feature_map_sizes`` in
    ``model.py``) and stashed as a ``spatial_shapes_list`` instance
    attribute on every ``TransformerEncoder``/``MSDeformAttn`` submodule in
    ``apply_patches()``, right after weight loading (same post-construction
    tagging pattern P8 uses for its static bias buffer). Each of those two
    classes is called by an unmodified-upstream caller with a fixed
    signature (``self.encoder(...)``, ``self.self_attn(...)``/
    ``self.cross_attn(...)``) that can't be made to pass the list as a new
    argument, so each instance carries its own copy instead.

P5  ``ContrastiveEmbed`` (open-vocabulary classification head): substitute
    ``_CE_MASK_FILL = -10.0`` for masked positions, tighter than P7's
    ``MASK_FILL`` since this head's score range is narrower -- preserves
    ~112 int8 quantization bins for the real signal instead of wasting them
    on the sentinel value.

P6  Every ``nn.Linear`` is replaced with ``_LinearAsConv1x1``, a
    ``Conv2d(kernel_size=1)`` wrapper applied post-weight-load via
    ``_upgrade_linears_to_conv1x1()``. The NPU has a dedicated fast kernel
    for Conv ops that MatMul/Gemm doesn't get, so this produces ``Conv`` ops
    in ONNX instead of ``MatMul``/``Gemm``.

P7  ``BiMultiHeadAttention.forward`` (vision<->language cross-attention, the
    only remaining caller of ``masked_fill_`` with upstream's raw
    ``float("-inf")``) is overridden (``BiMultiHeadAttention_forward``) to,
    at its two ``masked_fill_`` call sites: (a) bool-cast the mask at the
    point of use, since image feature masks are kept float32 elsewhere in
    the graph to avoid ``BOOL_8`` but this function expects a bool mask; and
    (b) replace ``float("-inf")`` with ``MASK_FILL = -300.0``, since
    ``-inf`` can propagate as NaN through FP16 paths on-device. ``-300.0``
    is the smallest magnitude that still reliably drives masked softmax
    weights to ~0 while keeping the quantization range tight. Scoped to
    this one upstream class via assignment in ``apply_patches()`` (same
    technique as P4/P11/P12) rather than a global ``torch.Tensor.masked_fill``/
    ``masked_fill_`` monkeypatch, so it can't affect other models or code
    running in the same process.

P8  ``WindowAttention``'s relative position bias: pre-index the bias table
    into a static buffer (``relative_position_index``) at init time instead
    of a dynamic ``Gather`` in the hot path.

P9  Swin's window partition/reverse are rewritten to stay at rank<=5 (see
    ``window_partition_5d``/``window_unpartition_5d`` in
    ``qai_hub_models.models.templates.corevision.window_partitioning``), since QNN doesn't support
    the rank-6 reshape upstream's ``window_partition``/``window_reverse``
    produce. Applied via component-safe patched-forward functions
    (``SwinTransformerBlock_forward``, ``BasicLayer_forward``) assigned onto
    the upstream ``SwinTransformerBlock``/``BasicLayer`` classes by
    ``apply_patches()``.

P10 ``MSDeformAttn``: sampling offsets/locations are kept rank-5 by merging
    the ``num_levels``/``num_points`` dims into one, avoiding a rank-6
    ``[bs, nQ, nH, nL, nP, 2]`` reshape. Sampling itself
    (``_multi_scale_deformable_attn_pytorch``) is a pure-PyTorch
    ``grid_sample``-based reimplementation of upstream's CUDA op.
    ``DeformableTransformerEncoderLayer``/``DeformableTransformerDecoderLayer``
    are imported unmodified from upstream; only the module-level
    ``MSDeformAttn`` name they construct is substituted (same
    module-attribute-substitution technique as P8's ``WindowAttention``),
    since it's looked up in ``transformer.py``'s own globals at
    ``__init__`` time.

P11 ``TransformerEncoder``: the text self-attention mask is cast to float32
    (``(~mask.bool()).float()``) instead of upstream's direct bool ``~mask``,
    avoiding a bool op in the exported graph (same theme as P1/P2/P7).
    Applied via ``TransformerEncoder_forward``, assigned onto upstream's
    ``TransformerEncoder`` class by ``apply_patches()``.

P12 ``TransformerDecoder.forward`` is overridden (``TransformerDecoder_forward``)
    to drop a debug block (``if output.isnan().any() | output.isinf().any():
    print(...)``) that is otherwise identical to upstream. The ``.any() |
    .any()`` is a tensor-valued condition for an ``if``, i.e. data-dependent
    control flow, which ``torch.export``'s Dynamo tracer refuses to trace
    (``Dynamic control flow is not supported``) regardless of whether the
    branch is ever taken at runtime. Purely a debug print with no effect on
    outputs/accuracy, so dropping it is safe.

P13 ``GroundingDINOTCollectionVision``'s ``input_proj`` FPN projections:
    replace ``nn.GroupNorm(32, 256)`` with ``_ExportSafeGroupNorm``, a
    manual reimplementation of the exact same math (reshape into groups,
    per-group mean/variance, normalize, per-channel affine) using only
    ``reshape``/``mean``/elementwise ops. AI Hub's ONNX backend returns an
    unresolved (``None``) output shape specifically across the native ONNX
    ``GroupNormalization`` op once quantization Q/DQ nodes are spliced
    around it under ``mixed_with_float``'s ``--quantize_io`` ONNX compile
    (confirmed by downloading and inspecting the compiled graph: the op's
    own input has a fully resolved shape in ``value_info``, its output does
    not), which fails ``ModelFileMetadata.from_hub_model()``'s shape
    assertion when AI Hub downloads this component's metadata -- compile,
    quantize, profile, and inference all otherwise succeed. Parameter names
    (``weight``/``bias``) are kept identical to ``nn.GroupNorm`` so the
    checkpoint's ``strict=True`` state-dict load in
    ``GroundingDINOTLoader._load_model_parts()`` is unaffected.
"""

from __future__ import annotations

import logging
import math
from typing import Any, cast

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn

from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO import (
    transformer as _transformer_upstream,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.backbone import (
    swin_transformer as _swin_upstream,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.fuse_modules import (
    BiMultiHeadAttention as _UpstreamBiMultiHeadAttention,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.transformer import (
    Transformer,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.transformer import (
    TransformerDecoder as _UpstreamTransformerDecoder,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.transformer import (
    TransformerEncoder as _UpstreamTransformerEncoder,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.models.GroundingDINO.utils import (
    gen_sineembed_for_position,
    get_sine_pos_embed,
)
from qai_hub_models.models.grounding_dino_t.external_repos.groundingdino.groundingdino.util.misc import (
    inverse_sigmoid as _inverse_sigmoid,
)
from qai_hub_models.models.templates.corevision.window_partitioning import (
    window_partition_5d,
    window_unpartition_5d,
)

_LOGGER = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# P1: NestedTensor + position encoding with torch.arange (no cumsum)
# -------------------------------------------------------------------------
class NestedTensor:
    """Minimal self-contained stand-in for upstream's NestedTensor: a tensor paired with its padding mask."""

    def __init__(self, tensors: torch.Tensor, mask: torch.Tensor) -> None:
        """Store the padded tensor batch and its boolean padding mask."""
        self.tensors = tensors
        self.mask = mask

    def decompose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (tensors, mask) pair."""
        return self.tensors, self.mask


class PositionEmbeddingSineHW(nn.Module):
    """Sinusoidal 2-D position embedding with independent H/W temperatures.

    P1: see module docstring.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperatureH: int = 10000,
        temperatureW: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        """Store embedding dimensionality, per-axis temperatures, and optional normalization scale."""
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Compute the [bs, 2*num_pos_feats, H, W] sine/cosine position embedding for a feature map."""
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        # Begin Qualcomm modification
        # ref: groundingdino.models.GroundingDINO.position_encoding.PositionEmbeddingSineHW
        # P1: see module docstring. Not asserted here since a data-dependent
        # check can't be traced by torch.export (see P12).
        not_mask = 1.0 - mask.float()  # float32 inversion
        H, W = x.shape[-2], x.shape[-1]
        y_embed = (
            torch.arange(1, H + 1, dtype=torch.float32, device=x.device)
            .view(1, -1, 1)
            .expand_as(not_mask)
        )
        x_embed = (
            torch.arange(1, W + 1, dtype=torch.float32, device=x.device)
            .view(1, 1, -1)
            .expand_as(not_mask)
        )
        # End Qualcomm modification

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (
            2 * torch.div(dim_tx, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (
            2 * torch.div(dim_ty, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# -------------------------------------------------------------------------
# P2: get_valid_ratio (float32 mask inversion, no bool StridedSlice)
# -------------------------------------------------------------------------
def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
    """Compute the fraction of each spatial dimension that is unpadded (valid), per batch element.

    P2: see module docstring.
    """
    _, H, W = mask.shape
    # Begin Qualcomm modification
    # ref: groundingdino.models.GroundingDINO.transformer.Transformer.get_valid_ratio
    # P2: see module docstring.
    valid_H = torch.sum(1.0 - mask[:, :, 0], 1)
    valid_W = torch.sum(1.0 - mask[:, 0, :], 1)
    # End Qualcomm modification
    return torch.stack([valid_W.float() / W, valid_H.float() / H], -1)


# -------------------------------------------------------------------------
# P3: gen_encoder_output_proposals (float32 reimplementation)
# -------------------------------------------------------------------------
def _gen_encoder_output_proposals(
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    spatial_shapes: list[tuple[int, int]],
    learnedwh: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate per-pixel region proposals from encoder memory.

    P3: see module docstring.

    Parameters
    ----------
    memory
        Encoder output, shape [bs, sum(H*W), C].
    memory_padding_mask
        Float32 padding mask (1.0 = padded), shape [bs, sum(H*W)].
    spatial_shapes
        (H, W) per feature level, as plain Python ints (P4) -- see module
        docstring for why these can't be read from the ``spatial_shapes``
        tensor.
    learnedwh
        Optional learned log-width/height offset; if None, a level-scaled
        default of 0.05 * 2**lvl is used.

    Returns
    -------
    output_memory : torch.Tensor
        Memory with padded/invalid positions zeroed, shape [bs, sum(H*W), C].
    output_proposals : torch.Tensor
        Proposal boxes in inverse-sigmoid (cxcywh) space, shape [bs, sum(H*W), 4].
    """
    # Begin Qualcomm modification
    # ref: groundingdino.models.GroundingDINO.utils.gen_encoder_output_proposals
    # P3: see module docstring.
    N_, _, _ = memory.shape
    proposals = []
    _cur = 0

    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
            N_, H_, W_, 1
        )
        valid_H = torch.sum(1.0 - mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(1.0 - mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * (
            learnedwh.sigmoid() * (2.0**lvl)
            if learnedwh is not None
            else 0.05 * (2.0**lvl)
        )
        proposals.append(torch.cat((grid, wh), -1).view(N_, -1, 4))
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals) - torch.log(1 - output_proposals)
    # P7: explicit bool-cast (see module docstring).
    memory_padding_mask_bool = (
        memory_padding_mask.bool()
        if memory_padding_mask.dtype != torch.bool
        else memory_padding_mask
    )
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask_bool.unsqueeze(-1), -MASK_FILL
    )
    output_proposals = output_proposals.masked_fill(
        (1.0 - output_proposals_valid.float()).bool(), -MASK_FILL
    )
    output_memory = memory.masked_fill(memory_padding_mask_bool.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(
        (1.0 - output_proposals_valid.float()).bool(), float(0)
    )
    return output_memory, output_proposals
    # End Qualcomm modification


# -------------------------------------------------------------------------
# P4: Transformer.forward (plain-Python spatial_shapes list, no unbacked
# symints from reading per-level (H_, W_) back out of a tensor)
# -------------------------------------------------------------------------
# Begin Qualcomm modification
# ref: groundingdino.models.GroundingDINO.transformer.Transformer.forward
def Transformer_forward(
    self: Any,
    srcs: list[torch.Tensor],
    masks: list[torch.Tensor],
    refpoint_embed: torch.Tensor | None,
    pos_embeds: list[torch.Tensor],
    tgt: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    text_dict: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor
]:
    """Forward for upstream's Transformer.

    P4: see module docstring.
    """
    assert text_dict is not None

    src_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes: list[tuple[int, int]] = []
    for lvl, (src, mask, pos_embed) in enumerate(
        zip(srcs, masks, pos_embeds, strict=True)
    ):
        bs, _c, h, w = src.shape
        spatial_shapes.append((h, w))

        src = src.flatten(2).transpose(1, 2)  # bs, hw, c
        mask = mask.flatten(1)  # bs, hw
        pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
        if self.num_feature_levels > 1 and self.level_embed is not None:
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
        else:
            lvl_pos_embed = pos_embed
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        src_flatten.append(src)
        mask_flatten.append(mask)
    src_flatten_t = torch.cat(src_flatten, 1)  # bs, sum{hw}, c
    mask_flatten_t = torch.cat(mask_flatten, 1)  # bs, sum{hw}
    lvl_pos_embed_flatten_t = torch.cat(lvl_pos_embed_flatten, 1)  # bs, sum{hw}, c
    spatial_shapes_t = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=src_flatten_t.device
    )
    level_start_index = torch.cat(
        (spatial_shapes_t.new_zeros((1,)), spatial_shapes_t.prod(1).cumsum(0)[:-1])
    )
    valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

    memory, memory_text = self.encoder(
        src_flatten_t,
        pos=lvl_pos_embed_flatten_t,
        level_start_index=level_start_index,
        spatial_shapes=spatial_shapes_t,
        valid_ratios=valid_ratios,
        key_padding_mask=mask_flatten_t,
        memory_text=text_dict["encoded_text"],
        text_attention_mask=~text_dict["text_token_mask"],
        position_ids=text_dict["position_ids"],
        text_self_attention_masks=text_dict["text_self_attention_masks"],
    )
    text_dict["encoded_text"] = memory_text

    if self.two_stage_type == "standard":
        output_memory, output_proposals = _gen_encoder_output_proposals(
            memory, mask_flatten_t, spatial_shapes
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        enc_outputs_class_unselected = self.enc_out_class_embed(
            output_memory, text_dict
        )

        topk_logits = enc_outputs_class_unselected.max(-1)[0]
        enc_outputs_coord_unselected = (
            self.enc_out_bbox_embed(output_memory) + output_proposals
        )  # (bs, sum{hw}, 4) unsigmoid
        topk = self.num_queries

        topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )  # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = torch.gather(
            output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid()  # sigmoid

        tgt_undetach = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
        )
        if self.embed_init_tgt:
            tgt_ = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        else:
            tgt_ = tgt_undetach.detach()

        if refpoint_embed is not None:
            assert tgt is not None
            refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
            tgt = torch.cat([tgt, tgt_], dim=1)
        else:
            refpoint_embed, tgt = refpoint_embed_, tgt_
    else:
        raise NotImplementedError(
            f"Transformer_forward (P4) only supports two_stage_type='standard', "
            f"got {self.two_stage_type!r}"
        )

    hs, references = self.decoder(
        tgt=tgt.transpose(0, 1),
        memory=memory.transpose(0, 1),
        memory_key_padding_mask=mask_flatten_t,
        pos=lvl_pos_embed_flatten_t.transpose(0, 1),
        refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
        level_start_index=level_start_index,
        spatial_shapes=spatial_shapes_t,
        valid_ratios=valid_ratios,
        tgt_mask=attn_mask,
        memory_text=text_dict["encoded_text"],
        text_attention_mask=~text_dict["text_token_mask"],
    )

    hs_enc = tgt_undetach.unsqueeze(0)
    ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)

    return hs, references, hs_enc, ref_enc, init_box_proposal
    # End Qualcomm modification


# -------------------------------------------------------------------------
# P8: Swin WindowAttention (static relative position bias)
# -------------------------------------------------------------------------
class WindowAttention(nn.Module):
    """Swin window attention with P8 (static bias)."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """Build the QKV projection, output projection, and static relative-position-bias buffer."""
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # Begin Qualcomm modification
        # ref: groundingdino.models.GroundingDINO.backbone.swin_transformer.WindowAttention
        # P8: see module docstring.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.relative_position_index: torch.Tensor
        # End Qualcomm modification

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _cache_relative_position_bias(self) -> None:
        """Precompute and register the fully-indexed bias as a static buffer.

        Called once by apply_patches() after _load_weights() so the
        relative_position_bias_table values are already loaded from the
        checkpoint. After this call, forward() reads self.relative_position_bias
        directly — zero Gather ops in the exported graph.
        """
        # Begin Qualcomm modification
        # P8: see module docstring.
        N = self.window_size[0] * self.window_size[1]
        bias = (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(N, N, -1)
            .permute(2, 0, 1)
            .contiguous()
        )  # (nH, N, N)
        self.register_buffer("relative_position_bias", bias)
        self.relative_position_bias: torch.Tensor
        # End Qualcomm modification

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute windowed self-attention over x, adding the static relative-position bias and optional shift mask."""
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Begin Qualcomm modification
        # P8: see module docstring.
        attn = attn + self.relative_position_bias.unsqueeze(0)
        # End Qualcomm modification

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


# -------------------------------------------------------------------------
# P5: ContrastiveEmbed (-10.0 fill, tighter quantization range)
# -------------------------------------------------------------------------
_CE_MASK_FILL: float = -10.0
# P5: see module docstring.
# https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/models/GroundingDINO/utils.py


class ContrastiveEmbed(nn.Module):
    """Classification head that scores each query against every text token via a dot product (open-vocabulary contrastive matching)."""

    # Begin Qualcomm modification
    # ref: https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/models/GroundingDINO/utils.py
    def __init__(self, max_text_len: int = 256) -> None:
        """Store the fixed text sequence length outputs are padded/truncated to."""
        super().__init__()
        self.max_text_len = max_text_len

    def forward(
        self, x: torch.Tensor, text_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Dot-product each query embedding against encoded text tokens, masking padding positions.

        P5: see module docstring.
        """
        y = text_dict["encoded_text"]
        text_token_mask = text_dict["text_token_mask"]
        res = x @ y.transpose(-1, -2)
        # P5: see module docstring.
        res = res.masked_fill(~text_token_mask[:, None, :], _CE_MASK_FILL)
        new_res = torch.full(
            (*res.shape[:-1], self.max_text_len),
            _CE_MASK_FILL,
            dtype=res.dtype,
            device=res.device,
        )
        new_res[..., : res.shape[-1]] = res
        return new_res

    # End Qualcomm modification


# -------------------------------------------------------------------------
# P6: nn.Linear -> Conv1x1 (fast NPU Conv kernel instead of MatMul/Gemm)
# -------------------------------------------------------------------------
# Begin Qualcomm modification
# P6: see module docstring.
class _LinearAsConv1x1(nn.Module):
    """Drop-in for nn.Linear(in_features, out_features) when input is 3-D [B,S,C].

    Transposes [B,S,C] -> [B,C,S,1] before Conv2d and back after, so the ONNX
    graph contains a Conv op (1x1 kernel) instead of MatMul/Gemm.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """Wrap an nn.Conv2d(kernel_size=1) with the same signature as nn.Linear."""
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped 1x1 Conv2d to a 2-D [N,C] or 3-D [B,S,C] input, matching nn.Linear's forward."""
        if x.dim() == 2:
            return F.linear(x, self.conv.weight.squeeze(-1).squeeze(-1), self.conv.bias)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        x = self.conv(x)
        return x.squeeze(-1).permute(0, 2, 1)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> _LinearAsConv1x1:
        """Build a _LinearAsConv1x1 from an nn.Linear, copying its weight/bias."""
        out = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        with torch.no_grad():
            out.conv.weight.copy_(linear.weight.unsqueeze(-1).unsqueeze(-1))
            if linear.bias is not None and out.conv.bias is not None:
                out.conv.bias.copy_(linear.bias)
        return out


def _upgrade_linears_to_conv1x1(module: nn.Module) -> None:
    """Recursively replace every nn.Linear with _LinearAsConv1x1.

    Skips nn.MultiheadAttention: its forward accesses out_proj.weight
    directly and would break if out_proj were swapped to _LinearAsConv1x1.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, _LinearAsConv1x1.from_linear(child))
        elif isinstance(child, (_LinearAsConv1x1, nn.Conv2d, nn.MultiheadAttention)):
            pass
        else:
            _upgrade_linears_to_conv1x1(child)


# End Qualcomm modification


# -------------------------------------------------------------------------
# P7: BiMultiHeadAttention.forward (bool-cast + -inf -> MASK_FILL substitution,
# scoped to this one upstream class instead of a global torch.Tensor patch)
# -------------------------------------------------------------------------
# Begin Qualcomm modification
# ref: https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/models/GroundingDINO/fuse_modules.py
MASK_FILL: float = -300.0
# P7: see module docstring. (P5/P3 use their own fill values instead.)


def BiMultiHeadAttention_forward(
    self: Any,
    v: torch.Tensor,
    l: torch.Tensor,
    attention_mask_v: torch.Tensor | None = None,
    attention_mask_l: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward for upstream's BiMultiHeadAttention (vision<->language cross-attention).

    P7: see module docstring.
    """
    bsz, tgt_len, _ = v.size()

    query_states = self.v_proj(v) * self.scale
    key_states = self._shape(self.l_proj(l), -1, bsz)
    value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
    value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_v_states = value_v_states.view(*proj_shape)
    value_l_states = value_l_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        )

    if self.stable_softmax_2d:
        attn_weights = attn_weights - attn_weights.max()
    if self.clamp_min_for_underflow:
        attn_weights = torch.clamp(attn_weights, min=-50000)
    if self.clamp_max_for_overflow:
        attn_weights = torch.clamp(attn_weights, max=50000)

    attn_weights_T = attn_weights.transpose(1, 2)
    attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
    if self.clamp_min_for_underflow:
        attn_weights_l = torch.clamp(attn_weights_l, min=-50000)
    if self.clamp_max_for_overflow:
        attn_weights_l = torch.clamp(attn_weights_l, max=50000)

    # mask vision for language
    if attention_mask_v is not None:
        attention_mask_v = (
            attention_mask_v[:, None, None, :]
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )
        # P7: see module docstring.
        attn_weights_l.masked_fill_(
            attention_mask_v.bool()
            if attention_mask_v.dtype != torch.bool
            else attention_mask_v,
            MASK_FILL,
        )
    attn_weights_l = attn_weights_l.softmax(dim=-1)

    # mask language for vision
    if attention_mask_l is not None:
        attention_mask_l = (
            attention_mask_l[:, None, None, :]
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
        )
        # P7: see module docstring.
        attn_weights.masked_fill_(
            attention_mask_l.bool()
            if attention_mask_l.dtype != torch.bool
            else attention_mask_l,
            MASK_FILL,
        )
    attn_weights_v = attn_weights.softmax(dim=-1)

    attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
    attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

    attn_output_v = torch.bmm(attn_probs_v, value_l_states)
    attn_output_l = torch.bmm(attn_probs_l, value_v_states)

    if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
        )
    if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
        raise ValueError(
            f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
        )

    attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output_v = attn_output_v.transpose(1, 2)
    attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

    attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
    attn_output_l = attn_output_l.transpose(1, 2)
    attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

    attn_output_v = self.out_v_proj(attn_output_v)
    attn_output_l = self.out_l_proj(attn_output_l)

    return attn_output_v, attn_output_l


# End Qualcomm modification


# -------------------------------------------------------------------------
# P9: Swin window partition/reverse (rank<=5, no rank-6 reshape)
#
# SwinTransformerBlock and BasicLayer are imported unmodified from upstream
# (groundingdino.models.GroundingDINO.backbone.swin_transformer); only their
# forward() is patched here.
# -------------------------------------------------------------------------
def SwinTransformerBlock_forward(
    self: Any, x: torch.Tensor, mask_matrix: torch.Tensor
) -> torch.Tensor:
    """Forward for upstream's SwinTransformerBlock.

    Parameters
    ----------
    self
        The SwinTransformerBlock module instance.
    x
        Input feature, tensor size (B, H*W, C).
    mask_matrix
        Attention mask for cyclic shift.

    Returns
    -------
    x_out : torch.Tensor
        Output feature, tensor size (B, H*W, C).
    """
    B, L, C = x.shape
    H, W = self.H, self.W
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    pad_r = (self.window_size - W % self.window_size) % self.window_size
    pad_b = (self.window_size - H % self.window_size) % self.window_size
    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
    _, Hp, Wp, _ = x.shape

    if self.shift_size > 0:
        shifted_x = torch.roll(
            x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
        )
        attn_mask = mask_matrix
    else:
        shifted_x = x
        attn_mask = None

    # Begin Qualcomm modification
    # ref: groundingdino.models.GroundingDINO.backbone.swin_transformer.window_partition/window_reverse
    # P9: see module docstring.
    x_windows, _ = window_partition_5d(shifted_x, self.window_size)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

    attn_windows = self.attn(x_windows, mask=attn_mask)

    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_unpartition_5d(
        attn_windows, self.window_size, (Hp, Wp), (Hp, Wp)
    )
    # End Qualcomm modification

    if self.shift_size > 0:
        x = torch.roll(
            shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
        )
    else:
        x = shifted_x

    if pad_r > 0 or pad_b > 0:
        x = x[:, :H, :W, :].contiguous()

    x = x.view(B, H * W, C)
    x = shortcut + self.drop_path(x)
    return x + self.drop_path(self.mlp(self.norm2(x)))


def BasicLayer_forward(
    self: Any, x: torch.Tensor, H: int, W: int
) -> tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
    """Forward for upstream's BasicLayer.

    Parameters
    ----------
    self
        The BasicLayer module instance.
    x
        Input feature, tensor size (B, H*W, C).
    H
        Feature map height in tokens.
    W
        Feature map width in tokens.

    Returns
    -------
    x : torch.Tensor
        Output tokens before downsampling, shape [B, H*W, C].
    H : int
        H (unchanged).
    W : int
        W (unchanged).
    x_down : torch.Tensor
        Output tokens after downsampling (or same as `x` if no downsample).
    Wh : int
        Height after downsampling.
    Ww : int
        Width after downsampling.
    """
    Hp = ((H + self.window_size - 1) // self.window_size) * self.window_size
    Wp = ((W + self.window_size - 1) // self.window_size) * self.window_size
    img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
    h_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
    )
    w_slices = (
        slice(0, -self.window_size),
        slice(-self.window_size, -self.shift_size),
        slice(-self.shift_size, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # Begin Qualcomm modification
    # ref: groundingdino.models.GroundingDINO.backbone.swin_transformer.window_partition
    # P9: see module docstring.
    mask_windows, _ = window_partition_5d(img_mask, self.window_size)
    # End Qualcomm modification
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(
        attn_mask == 0, 0.0
    )

    for blk in self.blocks:
        blk.H, blk.W = H, W
        x = blk(x, attn_mask)

    if self.downsample is not None:
        x_down = self.downsample(x, H, W)
        Wh, Ww = (H + 1) // 2, (W + 1) // 2
        return x, H, W, x_down, Wh, Ww
    return x, H, W, x, H, W


# Begin Qualcomm modification
# ref: groundingdino.models.GroundingDINO.transformer.TransformerEncoder.get_reference_points
def _get_reference_points(
    spatial_shapes: list[tuple[int, int]],
    valid_ratios: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """P4: see module docstring."""
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
        ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None] * valid_ratios[:, None]
    # End Qualcomm modification


# -------------------------------------------------------------------------
# P11: TransformerEncoder (float-cast text self-attention mask, no bool ~mask)
#
# TransformerEncoder is imported unmodified from upstream
# (groundingdino.models.GroundingDINO.transformer); only its forward() is
# patched here.
# -------------------------------------------------------------------------
def TransformerEncoder_forward(
    self: Any,
    src: torch.Tensor,
    pos: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    valid_ratios: torch.Tensor,
    key_padding_mask: torch.Tensor,
    memory_text: torch.Tensor,
    text_attention_mask: torch.Tensor,
    pos_text: torch.Tensor | None = None,
    text_self_attention_masks: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward for upstream's TransformerEncoder.

    Parameters
    ----------
    self
        The TransformerEncoder module instance. Must have a
        ``spatial_shapes_list`` attribute set by ``apply_patches()``
        (see module docstring, P4).
    src
        Flattened multi-level image features, shape [bs, sum(H*W), d_model].
    pos
        Positional embeddings for `src`, same shape as `src`.
    spatial_shapes
        Per-level (H, W) shape, shape [num_levels, 2].
    level_start_index
        Flattened start offset of each level within `src`.
    valid_ratios
        Per-level valid (unpadded) fraction, shape [bs, num_levels, 2].
    key_padding_mask
        Padding mask for `src`, shape [bs, sum(H*W)].
    memory_text
        Encoded text token embeddings, shape [bs, num_text, d_model].
    text_attention_mask
        Padding mask for `memory_text`.
    pos_text
        Optional precomputed text positional embeddings.
    text_self_attention_masks
        Optional segment-aware self-attention mask for the text encoder layers.
    position_ids
        Optional per-token position ids used to derive `pos_text`.

    Returns
    -------
    output : torch.Tensor
        Updated image feature memory, same shape as `src`.
    memory_text : torch.Tensor
        Updated text memory, same shape as `memory_text`.
    """
    output = src
    if self.num_layers > 0:
        # P4: see module docstring.
        reference_points = _get_reference_points(
            self.spatial_shapes_list, valid_ratios, device=src.device
        )

    if self.text_layers:
        bs, n_text, _ = memory_text.shape
        if pos_text is None and position_ids is None:
            pos_text = (
                torch.arange(n_text, device=memory_text.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(bs, 1, 1)
            )
            pos_text = get_sine_pos_embed(
                pos_text, num_pos_feats=256, exchange_xy=False
            )
        if position_ids is not None:
            pos_text = get_sine_pos_embed(
                position_ids[..., None], num_pos_feats=256, exchange_xy=False
            )

    for layer_id, layer in enumerate(self.layers):
        if self.fusion_layers:
            output, memory_text = self.fusion_layers[layer_id](
                v=output,
                l=memory_text,
                attention_mask_v=key_padding_mask,
                attention_mask_l=text_attention_mask,
            )

        if self.text_layers:
            # Begin Qualcomm modification
            # ref: groundingdino.models.GroundingDINO.transformer.TransformerEncoder.forward
            # P11: see module docstring.
            src_mask = (
                (~text_self_attention_masks.bool()).float() * MASK_FILL
                if text_self_attention_masks is not None
                else None
            )
            # End Qualcomm modification
            memory_text = self.text_layers[layer_id](
                src=memory_text.transpose(0, 1),
                src_mask=src_mask,
                src_key_padding_mask=text_attention_mask,
                pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
            ).transpose(0, 1)

        layer_out = layer(
            src=output,
            pos=pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        output = layer_out

    return output, memory_text


# -------------------------------------------------------------------------
# P12: TransformerDecoder (drop untraceable debug NaN-print block)
#
# TransformerDecoder is imported unmodified from upstream
# (groundingdino.models.GroundingDINO.transformer); only its forward() is
# patched here.
# -------------------------------------------------------------------------
# Begin Qualcomm modification
# ref: groundingdino.models.GroundingDINO.transformer.TransformerDecoder.forward
def TransformerDecoder_forward(
    self: Any,
    tgt: torch.Tensor,
    memory: torch.Tensor,
    tgt_mask: torch.Tensor | None = None,
    memory_mask: torch.Tensor | None = None,
    tgt_key_padding_mask: torch.Tensor | None = None,
    memory_key_padding_mask: torch.Tensor | None = None,
    pos: torch.Tensor | None = None,
    refpoints_unsigmoid: torch.Tensor | None = None,
    level_start_index: torch.Tensor | None = None,
    spatial_shapes: torch.Tensor | None = None,
    valid_ratios: torch.Tensor | None = None,
    memory_text: torch.Tensor | None = None,
    text_attention_mask: torch.Tensor | None = None,
) -> list[list[torch.Tensor]]:
    """Forward for upstream's TransformerDecoder.

    P12: see module docstring.
    """
    assert refpoints_unsigmoid is not None
    assert valid_ratios is not None
    output = tgt

    intermediate = []
    reference_points = refpoints_unsigmoid.sigmoid()
    ref_points = [reference_points]

    for layer_id, layer in enumerate(self.layers):
        if reference_points.shape[-1] == 4:
            reference_points_input = (
                reference_points[:, :, None]
                * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # nq, bs, nlevel, 4
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = (
                reference_points[:, :, None] * valid_ratios[None, :]
            )
        query_sine_embed = gen_sineembed_for_position(
            reference_points_input[:, :, 0, :]
        )  # nq, bs, 256*2

        raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
        pos_scale = self.query_scale(output) if self.query_scale is not None else 1
        query_pos = pos_scale * raw_query_pos

        output = layer(
            tgt=output,
            tgt_query_pos=query_pos,
            tgt_query_sine_embed=query_sine_embed,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_reference_points=reference_points_input,
            memory_text=memory_text,
            text_attention_mask=text_attention_mask,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_level_start_index=level_start_index,
            memory_spatial_shapes=spatial_shapes,
            memory_pos=pos,
            self_attn_mask=tgt_mask,
            cross_attn_mask=memory_mask,
        )

        if self.bbox_embed is not None:
            reference_before_sigmoid = _inverse_sigmoid(reference_points)
            delta_unsig = self.bbox_embed[layer_id](output)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()

            reference_points = new_reference_points.detach()
            ref_points.append(new_reference_points)

        intermediate.append(self.norm(output))

    return [
        [itm_out.transpose(0, 1) for itm_out in intermediate],
        [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
    ]
    # End Qualcomm modification


# -------------------------------------------------------------------------
# P10: MSDeformAttn (rank-5 sampling_locations, merged nLxnP dim)
#
# DeformableTransformerEncoderLayer/DeformableTransformerDecoderLayer are
# imported unmodified from upstream (groundingdino.models.GroundingDINO.
# transformer); only the module-level MSDeformAttn name they construct is
# substituted, below in apply_patches().
# -------------------------------------------------------------------------
# Begin Qualcomm modification
# ref: groundingdino.models.GroundingDINO.ms_deform_attn.MSDeformAttn
class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (batch_first=True, P10 rank-5 offsets)."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        batch_first: bool = True,
    ) -> None:
        """Build the sampling-offset/attention-weight/value/output projections and initialize them.

        Always batch-first; `batch_first` is accepted (and must be True) only
        so this is a drop-in for upstream's DeformableTransformerEncoderLayer/
        DecoderLayer.__init__, which pass batch_first=True explicitly.
        """
        super().__init__()
        if not batch_first:
            raise NotImplementedError(
                "MSDeformAttn (P10) only supports batch_first=True."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim

        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

        # P4: see module docstring.
        self.spatial_shapes_list: list[tuple[int, int]] | None = None

    def _reset_parameters(self) -> None:
        """Zero-init sampling offsets/weights (with a directional bias grid) and Xavier-init the value/output projections."""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-scale deformable attention: predict per-query sampling offsets/weights and sample `value` at those locations.

        P10: see module docstring.
        """
        assert self.spatial_shapes_list is not None, (
            "spatial_shapes_list not set -- apply_patches() must tag this "
            "instance after construction (see module docstring, P4)"
        )
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value_out = self.value_proj(value)
        if key_padding_mask is not None:
            mask_bool = (
                key_padding_mask.bool()
                if key_padding_mask.dtype != torch.bool
                else key_padding_mask
            )
            value_out = value_out.masked_fill(mask_bool[..., None], float(0))
        value_out = value_out.view(bs, num_value, self.num_heads, -1)

        # P10: see module docstring.
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        if reference_points.shape[-1] == 2:
            # offset_normalizer: [nL, 2] → repeat nP times → [nL*nP, 2]
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
            ).repeat_interleave(self.num_points, dim=0)  # [nL*nP, 2]
            # reference_points: [bs, nQ, nL, 2] → expand to [bs, nQ, 1, nL*nP, 2]
            ref = reference_points.repeat_interleave(self.num_points, dim=2).unsqueeze(
                2
            )
            sampling_locations = (
                ref + sampling_offsets / offset_normalizer[None, None, None, :, :]
            )
        elif reference_points.shape[-1] == 4:
            ref_xy = (
                reference_points[..., :2]
                .repeat_interleave(self.num_points, dim=2)
                .unsqueeze(2)
            )
            ref_wh = (
                reference_points[..., 2:]
                .repeat_interleave(self.num_points, dim=2)
                .unsqueeze(2)
            )
            sampling_locations = (
                ref_xy + sampling_offsets / self.num_points * ref_wh * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        output = _multi_scale_deformable_attn_pytorch(
            value_out, self.spatial_shapes_list, sampling_locations, attention_weights
        )
        return self.output_proj(output)


def _multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch multi-scale deformable attention sampling (grid_sample per level).

    Parameters
    ----------
    value
        Flattened multi-level feature values, shape [bs, sum(H*W), num_heads, embed_dims].
    value_spatial_shapes
        (H, W) per feature level, as plain Python ints (P4) -- see module
        docstring for why these can't be read from the ``spatial_shapes``
        tensor.
    sampling_locations
        Normalized sampling coordinates, shape [bs, num_queries, num_heads, num_levels*num_points, 2].
    attention_weights
        Per-point attention weights, shape [bs, num_queries, num_heads, num_levels*num_points].

    Returns
    -------
    torch.Tensor
        Sampled and weighted attention output, shape [bs, num_queries, num_heads*embed_dims].
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_lp, _ = sampling_locations.shape
    num_points = num_lp // len(value_spatial_shapes)
    num_levels = len(value_spatial_shapes)
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(bs * num_heads, embed_dims, H_, W_)
        )
        sl = sampling_grids[:, :, :, level * num_points : (level + 1) * num_points, :]
        sampling_grid_l_ = sl.transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


# End Qualcomm modification


# -------------------------------------------------------------------------
# P13: GroupNorm reimplementation (no native ONNX GroupNormalization op)
# -------------------------------------------------------------------------
# Begin Qualcomm modification
# ref: torch.nn.GroupNorm
class _ExportSafeGroupNorm(nn.Module):
    """Drop-in replacement for nn.GroupNorm(num_groups, num_channels).

    P13: see module docstring.
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
        """Store group/channel counts and allocate per-channel affine params, matching nn.GroupNorm's defaults."""
        super().__init__()
        assert num_channels % num_groups == 0, (
            "num_channels must be divisible by num_groups"
        )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x (shape [N,C,H,W]) per-group over (C/G, H, W), then apply the per-channel affine transform."""
        n, c, h, w = x.shape
        x_grouped = x.reshape(n, self.num_groups, c // self.num_groups * h * w)
        mean = x_grouped.mean(dim=2, keepdim=True)
        var = (x_grouped - mean).pow(2).mean(dim=2, keepdim=True)
        x_grouped = (x_grouped - mean) / torch.sqrt(var + self.eps)
        x = x_grouped.reshape(n, c, h, w)
        return x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)


# End Qualcomm modification


# -------------------------------------------------------------------------
# Weight loading + patch orchestration
# -------------------------------------------------------------------------
def _load_weights(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load checkpoint weights into the shared module graph.

    Remaps upstream GroundingDINO parameter names to the component module
    hierarchy, then calls load_state_dict with strict=True. Packed
    in_proj_weight/bias (nn.MultiheadAttention) and Swin's packed qkv
    weight/bias are loaded as-is, matching the packed shape of the
    upstream modules they're assigned to.
    """
    remapped: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("backbone.0."):
            new_k = "backbone_model." + new_k[len("backbone.0.") :]
        remapped[new_k] = v

    # Drop the two known-benign unexpected keys before strict load:
    # - label_enc.weight: training-only, not used in the inference path.
    # - bert.embeddings.position_ids: registered buffer in older HuggingFace
    #   BERT builds but absent in the pinned checkpoint's BERT.
    _KNOWN_UNEXPECTED = {"label_enc.weight", "bert.embeddings.position_ids"}
    remapped = {k: v for k, v in remapped.items() if k not in _KNOWN_UNEXPECTED}

    model.load_state_dict(remapped, strict=True)


def apply_patches(
    model_instance: Any | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
    spatial_shapes_list: list[tuple[int, int]] | None = None,
) -> None:
    """Apply all QNN patches for the shared GroundingDINO-T module graph.

    Called twice from ``from_pretrained()``, since the patches split into two
    phases that can't run at the same time:
        Called with no arguments, before construction: patches the upstream
          Swin/TransformerEncoder classes in place (SwinTransformerBlock.forward,
          BasicLayer.forward, TransformerEncoder.forward, and substituting our
          WindowAttention at the module-attribute level), since
          SwinTransformerBlock.__init__ looks up WindowAttention via its own
          module's globals and so must see the substitution before
          SwinTransformer(...)/Transformer(...) are constructed.
        Called with model_instance/state_dict/spatial_shapes_list, after
          construction and weight loading: upgrades every nn.Linear ->
          _LinearAsConv1x1 (Conv2d 1x1) for fast NPU Conv kernel instead of
          MatMul, and tags every TransformerEncoder/MSDeformAttn submodule
          with spatial_shapes_list (P4). This copies weights off the
          already-built, already-loaded submodules, so it requires a live
          model instance.

    Parameters
    ----------
    model_instance
        The constructed, weight-loaded module graph. If None, only
        the pre-construction class patches (P8/P9/P11/P12) are applied.
    state_dict
        Checkpoint state dict to load into model_instance. Required when
        model_instance is given.
    spatial_shapes_list
        Per-level (H, W) of the 4 FPN feature maps, e.g. from
        ``_feature_map_sizes(IMG_HEIGHT, IMG_WIDTH)`` (model.py). Required
        when model_instance is given; see module docstring, P4.
    """
    if model_instance is None:
        # P7: see module docstring.
        _UpstreamBiMultiHeadAttention.forward = BiMultiHeadAttention_forward

        # P1: see module docstring.
        cast(Any, _swin_upstream).NestedTensor = NestedTensor

        # P8: see module docstring.
        cast(Any, _swin_upstream).WindowAttention = WindowAttention

        # P9: see module docstring.
        _swin_upstream.SwinTransformerBlock.forward = SwinTransformerBlock_forward
        _swin_upstream.BasicLayer.forward = BasicLayer_forward

        # P10: see module docstring.
        cast(Any, _transformer_upstream).MSDeformAttn = MSDeformAttn

        # P11: see module docstring.
        _UpstreamTransformerEncoder.forward = TransformerEncoder_forward  # type: ignore[assignment]

        # P12: see module docstring.
        _UpstreamTransformerDecoder.forward = TransformerDecoder_forward

        # P2: see module docstring.
        Transformer.get_valid_ratio = lambda self, mask: get_valid_ratio(mask)

        # P4: see module docstring.
        Transformer.forward = Transformer_forward
        return

    assert state_dict is not None, "state_dict is required when model_instance is given"
    assert spatial_shapes_list is not None, (
        "spatial_shapes_list is required when model_instance is given"
    )

    # Load weights
    _load_weights(model_instance, state_dict)

    # P8/P4: see module docstring.
    for m in model_instance.modules():
        if isinstance(m, WindowAttention):
            m._cache_relative_position_bias()
        elif isinstance(m, (_UpstreamTransformerEncoder, MSDeformAttn)):
            m.spatial_shapes_list = spatial_shapes_list

    _upgrade_linears_to_conv1x1(model_instance)
