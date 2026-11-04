# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import torch
import transformers
from packaging.version import Version
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3ForCausalLM,
    Phi3MLP,
)

from qai_hub_models.models.templates.llm.common import TORCH_SUPPORTS_DYNAMIC_SHAPE
from qai_hub_models.models.templates.llm.model_adaptations import (
    ConvInplaceLinear,
    repeat_kv,
)
from qai_hub_models.models.templates.llm.sha_dynamic_kvcache import (
    SHADynamicCacheNewValueOnly,
)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_single_phi(
    x: torch.Tensor, rope_vals: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    cos, sin = rope_vals
    half = cos.shape[-1]
    rotary_dim = half * 2

    # Split x into rotated and pass-through portions
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]

    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]

    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x2 * cos + x1 * sin

    return torch.cat([x_rotated_1, x_rotated_2, x_pass], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | list[int] | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding to the query and key tensors.
    This is the exact implementation from Phi-3.5, which Phi-4 shares: both are
    `Phi3ForCausalLM` upstream, so these adaptations serve the whole Phi family.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


def QcPhi3_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: list[int] | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Phi3-style rotary position embeddings."""
    return apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)


class Phi35SHAAttention(Phi3Attention):
    """Split-Head Attention version of Phi3Attention (with Convs)"""

    @property
    def hidden_size_(self) -> int:
        if hasattr(self, "hidden_size"):
            return cast(int, self.hidden_size)
        return cast(int, self.config.hidden_size)

    # TODO: Needed?
    @property
    def num_attention_heads_(self) -> int:
        if hasattr(self, "num_heads"):
            return cast(int, self.num_heads)
        return self.config.num_attention_heads

    @property
    def num_key_value_heads_(self) -> int:
        if hasattr(self, "num_key_value_heads"):
            return cast(int, self.num_key_value_heads)
        return cast(int, self.config.num_key_value_heads)

    def prepare_conv(self) -> None:
        if not hasattr(self, "forward_no_conv"):
            qkv_out_dim = (
                self.num_attention_heads_ + 2 * self.num_key_value_heads_
            ) * self.head_dim

            self.qkv_proj_conv = nn.Conv2d(
                self.hidden_size_, qkv_out_dim, 1, bias=self.qkv_proj.bias is not None
            )
            self.o_proj_conv = nn.Conv2d(
                self.num_attention_heads_ * self.head_dim,
                self.hidden_size_,
                1,
                bias=self.o_proj.bias is not None,
            )

            self.qkv_proj_conv.weight.data.copy_(self.qkv_proj.weight[:, :, None, None])
            if self.qkv_proj.bias is not None:
                self.qkv_proj_conv.bias.data.copy_(self.qkv_proj.bias)  # type: ignore[union-attr, unused-ignore]

            self.o_proj_conv.weight.data.copy_(self.o_proj.weight[:, :, None, None])
            if self.o_proj.bias is not None:
                self.o_proj_conv.bias.data.copy_(self.o_proj.bias)  # type: ignore[union-attr, unused-ignore]

            del self.qkv_proj
            del self.o_proj

    def prepare_sha(self) -> None:
        # Ensure conv preparation is done first
        if not (hasattr(self, "qkv_proj_conv") and hasattr(self, "o_proj_conv")):
            raise RuntimeError(
                "The method 'prepare_sha' cannot be run on model without running 'prepare_conv' first."
            )

        if not hasattr(self, "forward_mha"):
            # Create separate projection modules for each head
            self.q_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.hidden_size_,
                        self.head_dim,
                        1,
                        bias=self.qkv_proj_conv.bias is not None,
                    )
                    for _ in range(self.num_attention_heads_)
                ]
            )
            self.k_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.hidden_size_,
                        self.head_dim,
                        1,
                        bias=self.qkv_proj_conv.bias is not None,
                    )
                    for _ in range(self.num_key_value_heads_)
                ]
            )
            self.v_proj_sha = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.hidden_size_,
                        self.head_dim,
                        1,
                        bias=self.qkv_proj_conv.bias is not None,
                    )
                    for _ in range(self.num_key_value_heads_)
                ]
            )

            self.forward_mha = cast(
                Callable[
                    [
                        torch.Tensor,
                        torch.Tensor | None,
                        torch.LongTensor | None,
                        DynamicCache | None,
                        bool,
                        bool,
                        torch.LongTensor | None,
                        Any,
                    ],
                    tuple[
                        torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None
                    ],
                ],
                self.forward,  # type: ignore[has-type, unused-ignore]
            )

            # pyright doesn't like that self.forward_sha doesn't take kwargs
            self.forward = self.forward_sha  # type: ignore[assignment, unused-ignore]  # pyright: ignore[reportAttributeAccessIssue]

        # Split qkv_proj_conv weights into q, k, v
        qkv_weight = self.qkv_proj_conv.weight
        q_size = self.num_attention_heads_ * self.head_dim
        k_size = self.num_key_value_heads_ * self.head_dim

        q_weight = qkv_weight[:q_size]
        k_weight = qkv_weight[q_size : q_size + k_size]
        v_weight = qkv_weight[q_size + k_size :]

        # Copy weights to individual heads
        for i in range(self.num_attention_heads_):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            q_proj = self.q_proj_sha[i]
            assert isinstance(q_proj, (nn.Linear, nn.Conv2d))
            q_proj.weight.data.copy_(q_weight[start_idx:end_idx, :])

        for i in range(self.num_key_value_heads_):
            start_idx = i * self.head_dim
            end_idx = (i + 1) * self.head_dim
            k_proj = self.k_proj_sha[i]
            v_proj = self.v_proj_sha[i]
            assert isinstance(k_proj, (nn.Linear, nn.Conv2d))
            assert isinstance(v_proj, (nn.Linear, nn.Conv2d))
            k_proj.weight.data.copy_(k_weight[start_idx:end_idx, :])
            v_proj.weight.data.copy_(v_weight[start_idx:end_idx, :])

        # Handle biases if present
        if self.qkv_proj_conv.bias is not None:
            q_bias = self.qkv_proj_conv.bias[:q_size]
            k_bias = self.qkv_proj_conv.bias[q_size : q_size + k_size]
            v_bias = self.qkv_proj_conv.bias[q_size + k_size :]

            for i in range(self.num_attention_heads_):
                start_idx = i * self.head_dim
                end_idx = (i + 1) * self.head_dim
                q_proj = self.q_proj_sha[i]
                assert isinstance(q_proj, (nn.Linear, nn.Conv2d))
                if q_proj.bias is not None:
                    q_proj.bias.data.copy_(q_bias[start_idx:end_idx])

            for i in range(self.num_key_value_heads_):
                start_idx = i * self.head_dim
                end_idx = (i + 1) * self.head_dim
                k_proj = self.k_proj_sha[i]
                v_proj = self.v_proj_sha[i]
                assert isinstance(k_proj, (nn.Linear, nn.Conv2d))
                assert isinstance(v_proj, (nn.Linear, nn.Conv2d))
                if k_proj.bias is not None:
                    k_proj.bias.data.copy_(k_bias[start_idx:end_idx])
                if v_proj.bias is not None:
                    v_proj.bias.data.copy_(v_bias[start_idx:end_idx])

        del self.qkv_proj_conv

    def forward_sha(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: SHADynamicCacheNewValueOnly | None = None,  # transformers<4.55
        past_key_values: SHADynamicCacheNewValueOnly
        | None = None,  # transformers>=4.55
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, list[torch.Tensor] | None]
        | tuple[
            torch.Tensor, list[torch.Tensor] | None, SHADynamicCacheNewValueOnly | None
        ]
    ):
        if past_key_values is not None and past_key_value is None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()

        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            hidden_states = hidden_states.unsqueeze(2)
        else:
            hidden_states = torch.reshape(
                hidden_states, (bsz, -1, 1, self.hidden_size_)
            )
        hidden_states = hidden_states.transpose(1, 3)

        query_states = [
            q_proj(hidden_states).permute(0, 2, 3, 1) for q_proj in self.q_proj_sha
        ]
        key_states = [
            k_proj(hidden_states).permute(0, 2, 3, 1) for k_proj in self.k_proj_sha
        ]
        value_states = [
            v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
        ]

        kv_seq_len = value_states[0].shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.value_cache[self.layer_idx][0].shape[-2]  # type: ignore[attr-defined, index, unused-ignore]

        assert position_embeddings is not None
        # Apply rotary embeddings (position_embeddings contains (cos, sin) tuple)
        query_states = [
            _apply_rope_single_phi(q, position_embeddings) for q in query_states
        ]
        key_states = [
            _apply_rope_single_phi(k, position_embeddings) for k in key_states
        ]

        # Handle past key values
        if past_key_value is not None:
            # Get past keys and values
            past_key = past_key_value.key_cache[self.layer_idx]  # type: ignore[attr-defined, index, unused-ignore]
            past_value = past_key_value.value_cache[self.layer_idx]  # type: ignore[attr-defined, index, unused-ignore]

            # Update cache with new values
            transposed_key_states = [
                key_state.transpose(2, 3) for key_state in key_states
            ]
            past_key_value.update(
                transposed_key_states,
                value_states,
                self.layer_idx,  # type: ignore[arg-type, unused-ignore]
                {},
            )

            # Concatenate with past values
            key_states = [
                torch.cat([pk, k.transpose(2, 3)], dim=3)
                for pk, k in zip(past_key, key_states, strict=False)
            ]
            value_states = [
                torch.cat([pv, v], dim=2)
                for pv, v in zip(past_value, value_states, strict=False)
            ]

        # Repeat KV heads if using GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)  # type: ignore[assignment, unused-ignore]
        value_states = repeat_kv(value_states, self.num_key_value_groups)  # type: ignore[assignment, unused-ignore]

        # Compute attention scores for each head
        attn_weights = [
            torch.matmul(q, k) / math.sqrt(self.head_dim)
            for q, k in zip(query_states, key_states, strict=False)
        ]

        # Check attention weights shape
        if attn_weights[0].size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
                f" {attn_weights[0].size()}"
            )

        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = [aw + attention_mask for aw in attn_weights]

        # Softmax
        attn_weights = [
            nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(
                query_states[0].dtype
            )
            for aw in attn_weights
        ]

        # Apply attention dropout if needed
        if hasattr(self, "attention_dropout") and self.training:
            attn_weights = [
                nn.functional.dropout(aw, p=self.attention_dropout, training=True)
                for aw in attn_weights
            ]

        # Apply attention to values
        attn_output = [
            torch.matmul(aw, v)
            for aw, v in zip(attn_weights, value_states, strict=False)
        ]

        # Check output shape
        if attn_output[0].size() != (bsz, 1, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, 1, q_len, self.head_dim)}, but is"
                f" {attn_output[0].size()}"
            )

        # Concatenate heads and apply output projection
        attn_output_concat = torch.cat(
            attn_output, dim=3
        )  # [batch, 1, seq_len, all_head_dim]
        attn_output_concat = attn_output_concat.permute(
            0, 3, 1, 2
        )  # [batch, all_head_dim, 1, seq_len]
        attn_output_concat = self.o_proj_conv(attn_output_concat)
        if TORCH_SUPPORTS_DYNAMIC_SHAPE:
            attn_output_concat = attn_output_concat.transpose(1, 3).squeeze(2)
        else:
            attn_output_concat = attn_output_concat.transpose(1, 3).reshape(
                bsz, q_len, self.hidden_size_
            )

        attn_weights_return = attn_weights if output_attentions else None

        if Version(transformers.__version__) >= Version("4.48.0"):
            return attn_output_concat, attn_weights_return
        return attn_output_concat, attn_weights_return, past_key_value


class QCPhi3MLP(Phi3MLP):
    def prepare_conv(self) -> None:
        gate_up_proj = cast(nn.Linear, self.gate_up_proj)  # type: ignore[has-type, unused-ignore]
        intermediate_size = gate_up_proj.out_features // 2
        has_bias = gate_up_proj.bias is not None

        gate_proj = nn.Linear(
            gate_up_proj.in_features,
            intermediate_size,
            bias=has_bias,
            dtype=gate_up_proj.weight.dtype,
            device=gate_up_proj.weight.device,
        )
        up_proj = nn.Linear(
            gate_up_proj.in_features,
            intermediate_size,
            bias=has_bias,
            dtype=gate_up_proj.weight.dtype,
            device=gate_up_proj.weight.device,
        )
        gate_proj.weight.data.copy_(gate_up_proj.weight[:intermediate_size, :])
        up_proj.weight.data.copy_(gate_up_proj.weight[intermediate_size:, :])
        if has_bias:
            gate_proj.bias.data.copy_(gate_up_proj.bias[:intermediate_size])  # type: ignore[union-attr, unused-ignore]
            up_proj.bias.data.copy_(gate_up_proj.bias[intermediate_size:])  # type: ignore[union-attr, unused-ignore]
        del self.gate_up_proj

        self.gate_proj = ConvInplaceLinear(gate_proj)
        self.up_proj = ConvInplaceLinear(up_proj)
        self.down_proj = ConvInplaceLinear(cast(nn.Linear, self.down_proj))  # type: ignore[has-type, unused-ignore]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  # type: ignore[override, unused-ignore]
        gate = self.gate_proj(hidden_states)  # type: ignore[operator, unused-ignore]
        up_states = self.up_proj(hidden_states)  # type: ignore[operator, unused-ignore]
        return self.down_proj(up_states * self.activation_fn(gate))  # type: ignore[operator, unused-ignore]


class QCPhi3ForCausalLM(Phi3ForCausalLM):
    def prepare_conv(self) -> None:
        self.lm_head = ConvInplaceLinear(cast(nn.Linear, self.lm_head))  # type: ignore[has-type, unused-ignore]
