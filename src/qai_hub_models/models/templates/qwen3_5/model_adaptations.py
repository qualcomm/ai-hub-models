# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Export-friendly forward passes for the Qwen3.5 text decoder.

These functions reuse the weights of the HuggingFace modules but take and
return flat tensors, so the static-shape graph needs no HF Cache objects and
no data-dependent control flow.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

# Private path: scan has no public alias as of torch 2.11.
from torch._higher_order_ops.scan import scan
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
)

# The delta rule is a port of GenAI Lab's exportable_gated_delta_rule
# (qcom-ai-hub/aimet#7841), so AIHM and GenAI Lab export the same kernel.
# The chunk extent is min(seq_len, cap); the triangular solve (band-4 seed plus
# four product-form squarings, A^0..A^15) is exact only for extents <= 64.
DELTA_RULE_CHUNK_CAP = 64
MAX_DELTA_RULE_CHUNK = 64
assert 1 <= DELTA_RULE_CHUNK_CAP <= MAX_DELTA_RULE_CHUNK


def apply_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate the first ``2 * cos.shape[-1]`` channels of x; pass the rest through."""
    half = cos.shape[-1]
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    x_pass = x[..., 2 * half :]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, x_pass), dim=-1)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Scaling by sqrt(d) makes mean(x^2) == sum(x^2), so eps is unchanged and aimet
    # fuses this as an RMSNormalization supergroup instead of quantizing rsqrt/sum.
    # gamma is a [d] tensor, not a scalar: qairt-converter rejects one scalar
    # constant shared as gamma by the query and key RMSNorms.
    d = x.shape[-1]
    x = x * d**0.5
    gamma = torch.full((d,), d**-0.5, dtype=x.dtype, device=x.device)
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * gamma


def _solve_triangular(
    attn: torch.Tensor, chunk_size: int, order: int = 4
) -> torch.Tensor:
    """(I - A)^-1 for strictly lower-triangular A, as ~11 matmuls.

    Order-``order`` Taylor seed masked to a band (exact inside it), refined by
    M0 (I+E0)(I+E0^2)(I+E0^4)(I+E0^8) with E0 = I - (I-A) M0. The mask is what
    keeps float32 stable; unmasked, this cancels catastrophically.
    """
    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    idx = torch.arange(chunk_size, device=attn.device)
    mask_acc = (idx[:, None] - idx[None, :] <= order).to(attn.dtype)

    m_mat = eye - attn
    acc = eye + attn
    power = attn
    for _ in range(2, order + 1):
        power = torch.matmul(power, attn)
        acc = acc + power
    m0 = mask_acc * acc

    e0 = eye - torch.matmul(m_mat, m0)
    e1 = torch.matmul(e0, e0)
    e2 = torch.matmul(e1, e1)
    e3 = torch.matmul(e2, e2)
    return torch.matmul(
        torch.matmul(
            torch.matmul(torch.matmul(m0, eye + e0), eye + e1),
            eye + e2,
        ),
        eye + e3,
    )


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    valid: torch.Tensor,
    chunk_size: int = DELTA_RULE_CHUNK_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked gated delta rule for both prefill and decode.

    The chunk extent is ``min(seq_len, chunk_size)``, kept symbolic with
    ``sym_min`` so one dynamic export hardens to extent 1 at decode and 64 at
    prefill; branching on the length instead would specialize the axis. The
    inter-chunk recurrence is a ``scan``, so the chunk count is its trip count
    and any sequence length works.

    Parameters
    ----------
    query
        (B, L, H, Dk), already L2-normalized.
    key
        (B, L, H, Dk), already L2-normalized.
    value
        (B, L, H, Dv).
    g
        (B, L, H) log decay per token.
    beta
        (B, L, H) write strength per token.
    state
        (B, H, Dk, Dv) recurrent state from the previous call.
    valid
        (B, L) 1.0 for real tokens, 0.0 for padding. Padded positions have
        their key, value and decay zeroed, so they leave the state untouched.
    chunk_size
        Upper bound on the chunk extent, not a fixed width.

    Returns
    -------
    output : torch.Tensor
        (B, L, H, Dv) attention output.
    state : torch.Tensor
        (B, H, Dk, Dv) recurrent state after the last token.
    """
    initial_dtype = query.dtype
    batch_size, seq_len, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    # Batch and heads are merged into one axis so no tensor exceeds rank 4:
    # the HTP backend rejects rank-5 ops (e.g. Neg) when building the context
    # binary. This is a layout change only; the math is aimet#7841's.
    query, key, value = (
        x.transpose(1, 2).reshape(-1, seq_len, x.shape[-1]).to(torch.float32)
        for x in (query, key, value)
    )
    beta, g = (
        x.transpose(1, 2).reshape(-1, seq_len).to(torch.float32) for x in (beta, g)
    )
    state = state.reshape(-1, k_head_dim, v_head_dim)
    chunk_size = cast(int, torch.sym_min(seq_len, chunk_size))

    mask = (
        valid[:, None, :]
        .expand(batch_size, num_heads, seq_len)
        .reshape(-1, seq_len)
        .to(torch.float32)
    )
    key = key * mask.unsqueeze(-1)
    value = value * mask.unsqueeze(-1)
    g = g * mask

    # Unconditional: branching on a symbolic pad would specialize the axis.
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    query = query * (k_head_dim**-0.5)
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # [B*H, n_chunks, chunk, D]
    query, key, value, k_beta, v_beta = (
        x.reshape(x.shape[0], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    )
    g = g.reshape(g.shape[0], -1, chunk_size)

    # Masks come from integer indices, not tril/triu of float ones: those inputs
    # get Q/DQ when quantized, which stops them folding before QNN sees Trilu.
    idx = torch.arange(chunk_size, device=query.device)
    tril_mask = (idx[:, None] >= idx[None, :]).to(torch.float32)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)
    strict_lower_tri = tril_mask - eye

    g_cum = (tril_mask @ g.unsqueeze(-1)).squeeze(-1)
    decay_mask = (
        (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * tril_mask
    ).exp() * tril_mask

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower_tri
    attn = _solve_triangular(attn, chunk_size)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    xs = [x.movedim(1, 0) for x in (query, key, value, k_cumdecay, decay_mask, g_cum)]

    def _chunk_step(
        state: torch.Tensor, xs_i: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_i, k_i, v_i, kc_i, dm_i, g_i = xs_i
        attn_i = (q_i @ k_i.transpose(-1, -2) * dm_i) * tril_mask
        v_new = v_i - kc_i @ state
        o_i = (q_i * g_i.unsqueeze(-1).exp()) @ state + attn_i @ v_new
        next_state = (
            state * g_i[:, -1, None, None].exp()
            + (k_i * (g_i[:, -1, None] - g_i).exp().unsqueeze(-1)).transpose(-1, -2)
            @ v_new
        )
        return next_state, o_i

    state, core_attn_out = scan(_chunk_step, state.to(value), xs)
    core_attn_out = core_attn_out.movedim(0, 1)
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :seq_len]
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, seq_len, v_head_dim)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    state = state.reshape(batch_size, num_heads, k_head_dim, v_head_dim)
    return core_attn_out, state


def _conv_with_state(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    first_valid: torch.Tensor,
    last_valid: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Depthwise causal conv over [conv_state, padded input].

    The real tokens are the contiguous run first_valid..last_valid; padding may
    sit on either side. The first gather moves the cached state to sit directly
    before first_valid, so padding never enters a real token's receptive field.
    The new state is the state_len inputs ending at last_valid.
    """
    state_len = conv_state.shape[-1]
    seq_len = mixed_qkv.shape[-1]
    joined = torch.cat([conv_state, mixed_qkv], dim=-1)
    positions = torch.arange(state_len + seq_len, device=joined.device)
    first_valid = first_valid.to(positions.dtype)
    last_valid = last_valid.to(positions.dtype)
    index = torch.where(
        positions < state_len + first_valid,
        (positions - first_valid).clamp(min=0),
        positions,
    )
    joined = joined.index_select(-1, index)
    out = F.conv1d(joined, weight, groups=joined.shape[1])
    state_index = positions[:state_len] + last_valid + 1
    return F.silu(out[:, :, -seq_len:]), joined.index_select(-1, state_index)


def linear_attention_forward(
    module: Qwen3_5GatedDeltaNet,
    hidden_states: torch.Tensor,
    valid: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gated DeltaNet layer with explicit state I/O.

    Parameters
    ----------
    module
        HF linear-attention module supplying the weights.
    hidden_states
        (B, S, hidden) normalized layer input.
    valid
        (B, S) 1.0 for real tokens, 0.0 for padding. Real tokens must be
        contiguous; padding may be on the left (host) or right (Genie).
    conv_state
        (B, conv_dim, kernel) last conv inputs from the previous call.
    recurrent_state
        (B, H, Dk, Dv) delta-rule state from the previous call.

    Returns
    -------
    output : torch.Tensor
        (B, S, hidden) layer output.
    conv_state : torch.Tensor
        Updated conv state.
    recurrent_state : torch.Tensor
        Updated recurrent state.
    """
    batch_size, seq_len, _ = hidden_states.shape
    # Sums, not min/max: aimet can't handle ReduceMin/Max in either export form.
    first_valid = (valid[0].cumsum(dim=0) == 0).to(valid.dtype).sum(dim=0)
    last_valid = first_valid + valid[0].sum(dim=0) - 1

    mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed_qkv, new_conv_state = _conv_with_state(
        mixed_qkv, conv_state, first_valid, last_valid, module.conv1d.weight
    )
    query, key, value = torch.split(
        mixed_qkv.transpose(1, 2),
        [module.key_dim, module.key_dim, module.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)

    z = module.in_proj_z(hidden_states).reshape(
        batch_size, seq_len, -1, module.head_v_dim
    )
    beta = module.in_proj_b(hidden_states).sigmoid()
    g = -module.A_log.float().exp() * F.softplus(
        module.in_proj_a(hidden_states).float() + module.dt_bias
    )

    repeats = module.num_v_heads // module.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    core_out, new_recurrent_state = gated_delta_rule(
        _l2norm(query), _l2norm(key), value, g, beta, recurrent_state, valid
    )
    core_out = core_out.reshape(-1, module.head_v_dim)
    core_out = module.norm(core_out, z.reshape(-1, module.head_v_dim))
    output = module.out_proj(core_out.reshape(batch_size, seq_len, -1))
    return output, new_conv_state, new_recurrent_state


def full_attention_forward(
    module: Qwen3_5Attention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    past_key: torch.Tensor,
    past_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gated GQA attention with partial RoPE and an external KV cache.

    Parameters
    ----------
    module
        HF full-attention module supplying the weights.
    hidden_states
        (B, S, hidden) normalized layer input.
    attention_mask
        (B, 1, S, past + S) additive mask.
    rope_cos
        (B, 1, S, rotary_dim / 2).
    rope_sin
        (B, 1, S, rotary_dim / 2).
    past_key
        (kv_heads, B, head_dim, past) cached keys.
    past_value
        (kv_heads, B, past, head_dim) cached values.

    Returns
    -------
    output : torch.Tensor
        (B, S, hidden) layer output.
    key : torch.Tensor
        (kv_heads, B, head_dim, S) keys for the new tokens.
    value : torch.Tensor
        (kv_heads, B, S, head_dim) values for the new tokens.
    """
    batch_size, seq_len, _ = hidden_states.shape
    head_dim = module.head_dim

    query, gate = torch.chunk(
        module.q_proj(hidden_states).view(batch_size, seq_len, -1, head_dim * 2),
        2,
        dim=-1,
    )
    gate = gate.reshape(batch_size, seq_len, -1)
    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(
        module.k_proj(hidden_states).view(batch_size, seq_len, -1, head_dim)
    ).transpose(1, 2)
    value = (
        module.v_proj(hidden_states)
        .view(batch_size, seq_len, -1, head_dim)
        .transpose(1, 2)
    )

    query = apply_partial_rope(query, rope_cos, rope_sin)
    key = apply_partial_rope(key, rope_cos, rope_sin)

    all_keys = torch.cat([past_key.permute(1, 0, 3, 2), key], dim=2)
    all_values = torch.cat([past_value.permute(1, 0, 2, 3), value], dim=2)
    groups = module.num_key_value_groups
    all_keys = all_keys.repeat_interleave(groups, dim=1)
    all_values = all_values.repeat_interleave(groups, dim=1)

    scores = (query @ all_keys.transpose(-1, -2)) * module.scaling + attention_mask
    attn = torch.softmax(scores, dim=-1) @ all_values
    attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
    output = module.o_proj(attn * torch.sigmoid(gate))
    return output, key.permute(1, 0, 3, 2), value.permute(1, 0, 2, 3)


def _conv1x1(weight: torch.Tensor) -> nn.Conv2d:
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=False)
    conv.weight.data.copy_(weight[:, :, None, None])
    return conv.to(weight.device)


class SHAFullAttention(nn.Module):
    """Split-head version of Qwen3_5Attention built from its weights.

    Each head gets its own 1x1 Conv2d projections, as in SHAQwen3Attention.
    q_proj interleaves [query | gate] per head, so head h owns rows
    [2h * head_dim, (2h + 2) * head_dim) of it.
    """

    def __init__(self, module: Qwen3_5Attention) -> None:
        super().__init__()
        head_dim = module.head_dim
        num_heads = module.config.num_attention_heads
        num_kv_heads = module.config.num_key_value_heads
        self.num_kv_groups = module.num_key_value_groups
        self.scaling = module.scaling

        q_weight = module.q_proj.weight.view(num_heads, 2, head_dim, -1)
        self.q_proj_sha = nn.ModuleList(
            _conv1x1(q_weight[h, 0]) for h in range(num_heads)
        )
        self.gate_proj_sha = nn.ModuleList(
            _conv1x1(q_weight[h, 1]) for h in range(num_heads)
        )
        k_weight = module.k_proj.weight.view(num_kv_heads, head_dim, -1)
        v_weight = module.v_proj.weight.view(num_kv_heads, head_dim, -1)
        self.k_proj_sha = nn.ModuleList(
            _conv1x1(k_weight[h]) for h in range(num_kv_heads)
        )
        self.v_proj_sha = nn.ModuleList(
            _conv1x1(v_weight[h]) for h in range(num_kv_heads)
        )
        self.o_proj_conv = _conv1x1(module.o_proj.weight)
        # q_norm / k_norm act on head_dim only, so every head shares them.
        self.q_norm = module.q_norm
        self.k_norm = module.k_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same I/O contract as full_attention_forward."""
        # (B, S, hidden) -> (B, hidden, 1, S) so each projection is a 1x1 conv.
        x = hidden_states.unsqueeze(2).transpose(1, 3)

        def project(conv: nn.Module) -> torch.Tensor:
            return conv(x).permute(0, 2, 3, 1)  # (B, 1, S, head_dim)

        keys = [
            apply_partial_rope(self.k_norm(project(k)), rope_cos, rope_sin)
            for k in self.k_proj_sha
        ]
        values = [project(v) for v in self.v_proj_sha]
        all_keys = [
            torch.cat([past_key[h : h + 1].transpose(0, 1), k.transpose(2, 3)], dim=3)
            for h, k in enumerate(keys)
        ]
        all_values = [
            torch.cat([past_value[h : h + 1].transpose(0, 1), v], dim=2)
            for h, v in enumerate(values)
        ]

        outputs = []
        for h, (q_proj, gate_proj) in enumerate(
            zip(self.q_proj_sha, self.gate_proj_sha, strict=True)
        ):
            query = apply_partial_rope(self.q_norm(project(q_proj)), rope_cos, rope_sin)
            kv = h // self.num_kv_groups
            scores = (query * self.scaling) @ all_keys[kv] + attention_mask
            attn = torch.softmax(scores, dim=-1) @ all_values[kv]
            outputs.append(attn * torch.sigmoid(project(gate_proj)))

        attn_output = torch.cat(outputs, dim=3).permute(0, 3, 1, 2)
        output = self.o_proj_conv(attn_output).permute(0, 2, 3, 1).squeeze(1)
        new_key = torch.cat([k.transpose(2, 3) for k in keys], dim=1).transpose(0, 1)
        new_value = torch.cat(values, dim=1).transpose(0, 1)
        return output, new_key, new_value
