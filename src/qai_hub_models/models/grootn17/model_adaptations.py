# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import MethodType
from typing import cast

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionAttention,
    Qwen3VLVisionModel,
    repeat_kv,
)

from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.configs.model.gr00t_n1d7 import (
    Gr00tN1d7Config,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.model.gr00t_n1d7.gr00t_n1d7 import (
    Gr00tN1d7ActionHead,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.model.modules.dit import (
    AdaLayerNorm,
    AlternateVLDiT,
    BasicTransformerBlock,
    SelfAttentionTransformer,
)
from qai_hub_models.models.grootn17.external_repos.gr00t.gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificLinear,
    MultiEmbodimentActionEncoder,
    swish,
)


class SelfAttentionTransformerOpt(nn.Module):
    """Forward a padding mask through N1.7's existing projection blocks.

    Sharing the blocks preserves their weights and attention implementation
    without replacing any forward methods on the native policy.
    """

    def __init__(self, transformer: SelfAttentionTransformer) -> None:
        super().__init__()
        self.transformer_blocks = transformer.transformer_blocks
        self.training = transformer.training

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = hidden_states.contiguous()
        all_hidden_states = [hidden_states]
        for block in self.transformer_blocks:
            # BasicTransformerBlock passes additive masks through unchanged.
            # Diffusers expands [B, 1, S, S] over heads before non-causal SDPA.
            hidden_states = block(hidden_states, attention_mask=attention_mask)
            all_hidden_states.append(hidden_states)
        if return_all_hidden_states:
            return hidden_states, all_hidden_states
        return hidden_states


def _additive_and_mask(
    a: torch.Tensor, b: torch.Tensor, mask_min: float = -100.0
) -> torch.Tensor:
    """Float-only equivalent of `(a.bool() & b.bool())`."""
    return (a * b - 1.0) * (-mask_min)


def build_causal_mask(
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    attention_mask: torch.Tensor | None,
    mask_min: float = -100.0,
) -> torch.Tensor:
    """Build the additive 4D causal + padding attention mask for the LLM.
    The exported LLM receives the completed [B, 1, S, S] mask, with masked
    positions set to `mask_min`.
    """
    min_val = float(mask_min)
    causal = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    if attention_mask is not None:
        pad_mask = (1.0 - attention_mask[:, None, None, :]) * min_val
        causal = torch.clamp(causal + pad_mask, min=min_val)

    return causal.to(dtype)


def eager_attention_forward_opt(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized eager attention forward with reordered scaling.
    Reorder masked attention scaling as `(scores + mask / scaling) * scaling`.
    """
    num_key_value_groups = cast(int, module.num_key_value_groups)
    key_states = repeat_kv(key, num_key_value_groups)
    value_states = repeat_kv(value, num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3))
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = (attn_weights + causal_mask * (1.0 / scaling)) * scaling

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@contextmanager
def swap_attr(obj: object, name: str, new_value: object) -> Generator[None, None, None]:
    old_value = getattr(obj, name)
    setattr(obj, name, new_value)
    try:
        yield
    finally:
        setattr(obj, name, old_value)


def bind_method_to_instance(
    model_instance: nn.Module, method_str: str, method_new: object
) -> None:
    """Rebind model_instance.forward so that during this instance's forward call,
    the module-level symbol `method_str` is temporarily `method_new`.
    """
    orig_forward = model_instance.forward

    def wrapped_forward(self: object, *args: object, **kwargs: object) -> object:
        mod = sys.modules[self.__class__.__module__]
        with swap_attr(mod, method_str, method_new):
            return orig_forward(*args, **kwargs)

    model_instance.forward = MethodType(wrapped_forward, model_instance)


@torch.no_grad()
def precompute_mrope_cos_sin(
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    position_ids: torch.Tensor,
    hidden_size: int,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Qwen3-VL mRoPE cos/sin on the host so the device graph holds constants.

    Host-side FP32 computation avoids evaluating rotary angles and trig
    functions at device precision. The resulting constants remove runtime
    Cos/Sin operations and the position_ids input from the text graph.

    Returns (cos, sin), each [1, seq_len, head_dim].
    """
    device = next(
        (b.device for b in rotary_emb.buffers() if b is not None), torch.device("cpu")
    )
    position_ids = position_ids.to(device=device).long()
    seq_len = position_ids.shape[-1]

    dummy_hidden = torch.zeros(
        1, seq_len, hidden_size, dtype=torch.float32, device=device
    )
    cos, sin = rotary_emb(dummy_hidden, position_ids)
    return cos.detach().to(dtype).contiguous(), sin.detach().to(dtype).contiguous()


class Qwen3VLTextBackboneOpt(nn.Module):
    """ONNX-exportable wrapper for the truncated Qwen3-VL text model.

    Uses host-precomputed mRoPE buffers and a host-built 4D attention mask.
    """

    _rope_cos: torch.Tensor
    _rope_sin: torch.Tensor

    def __init__(
        self,
        text_config: Qwen3VLTextConfig,
        src_state_dict: dict,
        position_ids: torch.Tensor,
        *,
        select_layer: int,
    ) -> None:
        super().__init__()
        text_config = copy.deepcopy(text_config)
        text_config.num_hidden_layers = select_layer
        text_config._attn_implementation = "eager"

        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(text_config, i)
                for i in range(text_config.num_hidden_layers)
            ]
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=text_config)

        required = dict(self.named_parameters())
        missing = sorted(required.keys() - src_state_dict.keys())
        mismatches = [
            f"{key}: source {tuple(src_state_dict[key].shape)}, destination {tuple(required[key].shape)}"
            for key in sorted(required.keys() & src_state_dict.keys())
            if src_state_dict[key].shape != required[key].shape
        ]
        errors = []
        if missing:
            errors.append(f"missing required parameters: {missing}")
        if mismatches:
            errors.append(f"shape mismatches: {'; '.join(mismatches)}")
        if errors:
            raise RuntimeError(
                f"Invalid Qwen3VLTextBackboneOpt source weights: {'; '.join(errors)}"
            )

        self.load_state_dict(
            {key: src_state_dict[key] for key in required}, strict=True
        )

        for layer in self.layers:
            bind_method_to_instance(
                cast(Qwen3VLTextDecoderLayer, layer).self_attn,
                "eager_attention_forward",
                eager_attention_forward_opt,
            )

        cos, sin = precompute_mrope_cos_sin(
            self.rotary_emb, position_ids, text_config.hidden_size
        )
        self.register_buffer("_rope_cos", cos)
        self.register_buffer("_rope_sin", sin)

    def _deepstack_add(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Inject deepstack features at visual token positions (ONNX-safe cumsum scatter).

        Replicates Qwen3-VL's visual-token insertion without dynamic boolean
        indexing. The 0/1 float mask is cast to int64 for cumsum/gather indices
        and kept in floating point for the final elementwise multiply.
        """
        B, seq_len, hidden_size = hidden_states.shape
        vis_mask_f = visual_pos_masks.to(hidden_states.dtype)

        vis_count = visual_pos_masks.long().cumsum(dim=1) - 1
        vis_count = vis_count.clamp(min=0)

        num_vis = visual_embeds.shape[0]
        vis_exp = visual_embeds.unsqueeze(0).expand(B, num_vis, hidden_size)

        idx = vis_count.unsqueeze(-1).expand(B, seq_len, hidden_size)
        gathered = vis_exp.gather(1, idx)

        mask_f = vis_mask_f.unsqueeze(-1).to(gathered.dtype)
        delta = gathered * mask_f

        return hidden_states + delta

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_0: torch.Tensor | None = None,
        deepstack_1: torch.Tensor | None = None,
        deepstack_2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs_embeds
            [B, seq_len, hidden_size] merged token and visual embeddings.
        attention_mask
            Already-built 4D additive causal+padding mask, `[B, 1, seq_len,
            seq_len]` (see `build_causal_mask` / `preprocess_llm`) -- built
            host-side, not inside this graph.
        visual_pos_masks
            Optional [B, seq_len] float mask identifying visual token positions.
        deepstack_0
            Optional [num_vis_tokens, hidden_size] features for the first layer.
        deepstack_1
            Optional [num_vis_tokens, hidden_size] features for the second layer.
        deepstack_2
            Optional [num_vis_tokens, hidden_size] features for the third layer.

        Returns
        -------
        torch.Tensor
            [B, seq_len, hidden_size] decoder features before final RMSNorm.
        """
        seq_len = inputs_embeds.shape[1]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        hidden_states = inputs_embeds
        position_embeddings = (self._rope_cos.to(dtype), self._rope_sin.to(dtype))
        cache_position = torch.arange(seq_len, device=device)

        deepstack_list = [
            d.to(dtype)
            for d in (deepstack_0, deepstack_1, deepstack_2)
            if d is not None
        ]

        for layer_idx, layer in enumerate(self.layers):
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            if visual_pos_masks is not None and layer_idx < len(deepstack_list):
                hidden_states = self._deepstack_add(
                    hidden_states, visual_pos_masks, deepstack_list[layer_idx]
                )

        return hidden_states


# ViT
def _apply_rotary_real(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Real-valued rotary embedding (no complex ops), for ONNX export.

    Equivalent to Qwen3-VL's own apply_rotary_pos_emb_vision
    (x*cos + rotate_half(x)*sin); only the unsqueeze axis differs to match the
    [seq, heads, dim] layout used inside the per-image attention chunks below.
    """
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return (x * cos + torch.cat((-x2, x1), dim=-1) * sin).to(orig_dtype)


def _make_qwen3vl_vision_attention_forward_opt(
    attn_module: Qwen3VLVisionAttention, chunk_sizes: list[int]
) -> Callable[..., torch.Tensor]:
    """ONNX-exportable Qwen3VLVisionAttention forward.

    Replaces cu_seqlens-based variable-length attention with per-image chunk
    splitting (grid_thw is baked, so chunk_sizes is a compile-time constant)
    and complex rotary embeddings with real-valued rotate_half.
    """

    def forward(
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = attn_module.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, attn_module.num_heads, -1).permute(1, 0, 2, 3)
        q, k, v = qkv.unbind(0)

        cos, sin = cast(tuple[torch.Tensor, torch.Tensor], position_embeddings)
        q = _apply_rotary_real(q, cos, sin)
        k = _apply_rotary_real(k, cos, sin)

        if len(chunk_sizes) > 1:
            attn_outputs = []
            for q_c, k_c, v_c in zip(
                torch.split(q, chunk_sizes, dim=0),
                torch.split(k, chunk_sizes, dim=0),
                torch.split(v, chunk_sizes, dim=0),
                strict=False,
            ):
                q_c = q_c.transpose(0, 1)
                k_c = k_c.transpose(0, 1)
                v_c = v_c.transpose(0, 1)
                w = F.softmax(
                    (
                        torch.matmul(q_c, k_c.transpose(-2, -1)) * attn_module.scaling
                    ).float(),
                    dim=-1,
                ).to(v_c.dtype)
                attn_outputs.append(torch.matmul(w, v_c).transpose(0, 1))
            attn_output = torch.cat(attn_outputs, dim=0)
        else:
            q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
            w = F.softmax(
                (torch.matmul(q, k.transpose(-2, -1)) * attn_module.scaling).float(),
                dim=-1,
            ).to(v.dtype)
            attn_output = torch.matmul(w, v).transpose(0, 1)

        return attn_module.proj(attn_output.reshape(seq_length, -1).contiguous())

    return forward


class Qwen3VLVisionTransformerOpt(nn.Module):
    """ONNX-exportable Qwen3-VL vision tower wrapper.

    Uses independent block/merger copies, flattened patch weights, and precomputed
    positional/rotary buffers. Fixed attention chunks and real-valued rotary
    operations support static export.
    """

    _grid_thw: torch.Tensor
    _patch_embed_weight: torch.Tensor
    _patch_embed_bias: torch.Tensor
    _pos_embeds: torch.Tensor
    _rot_cos: torch.Tensor
    _rot_sin: torch.Tensor

    def __init__(
        self, vision_model: Qwen3VLVisionModel, grid_thw: torch.Tensor
    ) -> None:
        super().__init__()
        # patch_embed.proj is a Conv3d with kernel_size == stride == the full
        # patch extent, applied to input already reshaped to exactly one
        # spatial position per patch -- mathematically identical to a Linear
        # over the flattened patch, and HTP v73 has no Conv3d op at all
        # ("no properties registered for q::QNN_Conv3d"). Bake the equivalent
        # flattened Linear weight/bias instead of aliasing patch_embed itself.
        proj = vision_model.patch_embed.proj
        self.register_buffer(
            "_patch_embed_weight",
            proj.weight.reshape(proj.weight.shape[0], -1).clone().detach(),
        )
        self.register_buffer(
            "_patch_embed_bias", cast(torch.Tensor, proj.bias).clone().detach()
        )

        # One deepcopy preserves any ties within these subtrees, while giving
        # each wrapper separate modules, parameters, buffers, and configuration.
        # In particular, .to() during serialization cannot move native weights.
        self.blocks, self.merger, self.deepstack_merger_list = copy.deepcopy(
            (
                vision_model.blocks,
                vision_model.merger,
                vision_model.deepstack_merger_list,
            )
        )
        self.deepstack_visual_indexes = list(vision_model.deepstack_visual_indexes)

        with torch.no_grad():
            pos_embeds = vision_model.fast_pos_embed_interpolate(grid_thw)
            rotary = vision_model.rot_pos_emb(grid_thw)
            emb = torch.cat((rotary, rotary), dim=-1)

        self.register_buffer("_pos_embeds", pos_embeds.clone().detach().contiguous())
        self.register_buffer("_rot_cos", emb.cos().clone().detach().contiguous())
        self.register_buffer("_rot_sin", emb.sin().clone().detach().contiguous())
        self.register_buffer("_grid_thw", grid_thw.clone().detach())

        chunk_sizes = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).tolist()
        for block in self.blocks:
            # The closure captures only this wrapper's attention module/grid.
            attn = cast(Qwen3VLVisionAttention, block.attn)
            attn.forward = _make_qwen3vl_vision_attention_forward_opt(
                attn, chunk_sizes=chunk_sizes
            )

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = F.linear(
            pixel_values, self._patch_embed_weight, self._patch_embed_bias
        )
        hidden_states = hidden_states + self._pos_embeds
        position_embeddings = (self._rot_cos, self._rot_sin)

        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states, cu_seqlens=None, position_embeddings=position_embeddings
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_features.append(
                    self.deepstack_merger_list[idx](hidden_states)
                )

        image_embeds = self.merger(hidden_states)
        deepstack = (
            torch.stack(deepstack_features)
            if deepstack_features
            else image_embeds.new_zeros(1, 1, 1)
        )
        return image_embeds, deepstack


def assert_grid_thw_matches(
    baked_grid_thw: torch.Tensor, runtime_grid_thw: torch.Tensor
) -> None:
    """Require grid_thw to match the ViT graph's baked grid, as in TensorRT."""
    if not torch.equal(baked_grid_thw.cpu(), runtime_grid_thw.cpu()):
        raise ValueError(
            f"grid_thw mismatch: ViT graph was baked for {baked_grid_thw.tolist()} "
            f"but received {runtime_grid_thw.tolist()}. A different image "
            "resolution or camera count requires a fresh export."
        )


# DiT / action head
def _bool_mask_to_additive(
    mask: torch.Tensor | None,
    dtype: torch.dtype,
    query_len: int | None = None,
) -> torch.Tensor | None:
    """Convert keep masks to ONNX additive masks, expanding across queries."""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        additive = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        additive = additive.masked_fill(~mask, float("-inf"))
    else:
        additive = mask
    if query_len is not None and additive.dim() == 2:
        additive = additive[:, None, None, :].expand(-1, 1, query_len, -1)
    return additive


class SharedSelfAttentionQKV(nn.Module):
    """Fused self-attention Q/K/V projection with one ONNX initializer."""

    def __init__(self, to_q: nn.Linear, to_k: nn.Linear, to_v: nn.Linear) -> None:
        super().__init__()

        if to_q.in_features != to_k.in_features or to_q.in_features != to_v.in_features:
            raise ValueError("Q/K/V input dimensions must match")
        if (
            to_q.out_features != to_k.out_features
            or to_q.out_features != to_v.out_features
        ):
            raise ValueError("Q/K/V output dimensions must match")

        with torch.no_grad():
            weight_t = torch.cat(
                [
                    to_q.weight.detach().transpose(0, 1),
                    to_k.weight.detach().transpose(0, 1),
                    to_v.weight.detach().transpose(0, 1),
                ],
                dim=1,
            ).contiguous()

        self._onnx_weight_t = nn.Parameter(weight_t, requires_grad=False)

        if to_q.bias is not None:
            bias = torch.cat(
                [
                    to_q.bias.detach(),
                    to_k.bias.detach(),
                    to_v.bias.detach(),
                ],
                dim=0,
            ).contiguous()
            self._onnx_bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("_onnx_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(x, self._onnx_weight_t)
        if self._onnx_bias is not None:
            output = output + self._onnx_bias
        return output


def _prepare_self_attention_qkv_for_onnx_export(attn: Attention) -> None:
    """Replace one self-attention block's Q/K/V with a shared fused projection."""
    if hasattr(attn, "_shared_qkv"):
        return

    attn.add_module(
        "_shared_qkv",
        SharedSelfAttentionQKV(
            cast(nn.Linear, attn.to_q),
            cast(nn.Linear, attn.to_k),
            cast(nn.Linear, attn.to_v),
        ),
    )
    # The fused projection above replaces these Linear modules for export.
    attn.to_q = nn.Identity()  # type: ignore[assignment]
    attn.to_k = nn.Identity()  # type: ignore[assignment]
    attn.to_v = nn.Identity()  # type: ignore[assignment]


class AttnProcessorCrossAttnKV(AttnProcessor2_0):
    """Process split cached cross-attention K/V with broadcastable masks."""

    def __call__(  # type: ignore[override, unused-ignore]
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states_key: torch.Tensor | None = None,
        encoder_hidden_states_value: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if encoder_hidden_states_key is None:
            batch_size = hidden_states.shape[0]
        else:
            batch_size = encoder_hidden_states_key.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        shared_qkv = getattr(attn, "_shared_qkv", None)

        if encoder_hidden_states_key is None and shared_qkv is not None:
            qkv = shared_qkv(hidden_states)
            query, key, value = qkv.chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)

            if encoder_hidden_states_key is None:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
            else:
                key = encoder_hidden_states_key
                value = cast(torch.Tensor, encoder_hidden_states_value)
                attention_mask = encoder_attention_mask

        if encoder_hidden_states_key is None:
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_k is not None:
                key = attn.norm_k(key)
        else:
            head_dim = key.shape[-1]

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def _encode_actions_with_fixed_timestep(
    encoder: MultiEmbodimentActionEncoder,
    actions: torch.Tensor,
    cat_ids: torch.Tensor,
    timestep_encoding: torch.Tensor,
) -> torch.Tensor:
    """Run the N1.7 action MLP with runtime-selected Sin/Cos and exact arithmetic."""
    action_embedding = encoder.W1(actions, cat_ids)
    time_embedding = timestep_encoding.expand(
        actions.shape[0], actions.shape[1], -1
    ).to(dtype=action_embedding.dtype)
    combined = torch.cat((action_embedding, time_embedding), dim=-1)
    return encoder.W3(swish(encoder.W2(combined, cat_ids)), cat_ids)


def _apply_fixed_ada_norm(
    norm1: nn.Module,
    hidden_states: torch.Tensor,
    multiplier: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Apply AdaNorm after its timestep projection has been folded."""
    norm = getattr(norm1, "norm", None)
    if norm is None:
        raise RuntimeError(
            "Expected AdaNorm module with a `norm` submodule while folding "
            f"{type(norm1).__name__}."
        )
    return norm(hidden_states) * multiplier + shift


class BasicTransformerBlockCrossAttnKVOpt(BasicTransformerBlock):
    """
    Transformer block adapted to accept pre-split cross-attention key/value inputs.

    Converts boolean masks to additive masks for ONNX attention locally to the
    adapted model. Cross-attention consumes precomputed K/V; self-attention
    computes its own Q/K/V from the current hidden states.
    """

    def forward(  # type: ignore[override, unused-ignore]
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states_key: torch.Tensor | None = None,
        encoder_hidden_states_value: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        fixed_ada_norm: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.norm_type == "ada_norm":
            if fixed_ada_norm is None:
                norm_hidden_states = self.norm1(hidden_states, temb)
            else:
                multiplier, shift = fixed_ada_norm
                norm_hidden_states = _apply_fixed_ada_norm(
                    self.norm1, hidden_states, multiplier, shift
                )
        else:
            if fixed_ada_norm is not None:
                raise RuntimeError(
                    "Received folded AdaNorm conditioning for a non-AdaNorm "
                    f"block ({type(self.norm1).__name__})."
                )
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        dtype = norm_hidden_states.dtype
        is_cross_attn = encoder_hidden_states_key is not None
        if is_cross_attn:
            query_len = norm_hidden_states.shape[1]
            additive_mask = _bool_mask_to_additive(
                encoder_attention_mask, dtype, query_len=query_len
            )
        else:
            additive_mask = _bool_mask_to_additive(
                attention_mask, dtype, query_len=None
            )

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states_key=encoder_hidden_states_key,
            encoder_hidden_states_value=encoder_hidden_states_value,
            attention_mask=additive_mask,
            encoder_attention_mask=additive_mask,
        )
        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class AlternateVLDiTCrossAttnKV(AlternateVLDiT):
    """
    AlternateVLDiT adapted to accept pre-split cross-attention key/value tensors.

    Replaces the standard `encoder_hidden_states` input with 16 stacked
    `encoder_hidden_states_keys`/`values` (one per cross-attention block).
    """

    def forward(  # type: ignore[override, unused-ignore]
        self,
        hidden_states: torch.Tensor,  # [B, T, D]
        encoder_hidden_states_keys: list[torch.Tensor],
        encoder_hidden_states_values: list[torch.Tensor],
        timestep: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        backbone_attention_mask: torch.Tensor | None = None,
        return_all_hidden_states: bool = False,
        timestep_index: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert image_mask is not None, "Image mask is required"
        assert getattr(self.config, "interleave_self_attention", False), (
            "Interleave self attention must be enabled"
        )

        fixed_temb = None
        if timestep_index is not None:
            if not hasattr(self, "_folded_temb"):
                raise RuntimeError(
                    "Requested folded timestep conditioning before it was prepared."
                )
            fixed_temb = torch.index_select(self._folded_temb, 0, timestep_index)
        temb = self.timestep_encoder(timestep) if fixed_temb is None else fixed_temb
        hidden_states = hidden_states.contiguous()

        image_mask_f = image_mask.to(hidden_states.dtype)
        backbone_mask_f = cast(torch.Tensor, backbone_attention_mask).to(
            hidden_states.dtype
        )
        image_attention_mask = _additive_and_mask(image_mask_f, backbone_mask_f)
        non_image_attention_mask = _additive_and_mask(
            1.0 - image_mask_f, backbone_mask_f
        )

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            fixed_ada_norm = None
            if timestep_index is not None:
                fixed_ada_norm = (
                    torch.index_select(
                        cast(torch.Tensor, block._folded_ada_multiplier),
                        0,
                        timestep_index,
                    ),
                    torch.index_select(
                        cast(torch.Tensor, block._folded_ada_shift), 0, timestep_index
                    ),
                )
            if idx % 2 == 1:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states_key=None,
                    encoder_hidden_states_value=None,
                    encoder_attention_mask=None,
                    temb=temb,
                    fixed_ada_norm=fixed_ada_norm,
                )
            else:
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    curr_mask = non_image_attention_mask
                else:
                    curr_mask = image_attention_mask
                kv_idx = idx // 2
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states_key=encoder_hidden_states_keys[kv_idx],
                    encoder_hidden_states_value=encoder_hidden_states_values[kv_idx],
                    encoder_attention_mask=curr_mask,
                    temb=temb,
                    fixed_ada_norm=fixed_ada_norm,
                )
            all_hidden_states.append(hidden_states)

        conditioning = temb
        if timestep_index is None:
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            multiplier = 1 + scale[:, None]
            shift = shift[:, None]
        else:
            shift = torch.index_select(self._folded_final_shift, 0, timestep_index)
            multiplier = torch.index_select(
                self._folded_final_multiplier, 0, timestep_index
            )
        hidden_states = self.norm_out(hidden_states) * multiplier + shift
        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


def _fixed_embodiment_linear_forward(
    self: CategorySpecificLinear, x: torch.Tensor, cat_ids: torch.Tensor
) -> torch.Tensor:
    """Replacement `CategorySpecificLinear.forward` for a layer whose
    embodiment dim has been collapsed to a single fixed row -- `cat_ids` is
    accepted (call sites still pass it) but ignored.
    """
    return torch.matmul(x, self.W) + self.b


def _collapse_category_specific_linear(
    layer: CategorySpecificLinear, embodiment_id: int
) -> None:
    """Slices a `CategorySpecificLinear`'s per-embodiment `W`/`b` down to the
    single row for `embodiment_id`, replacing the runtime Gather+bmm with a
    plain matmul. Shrinks the layer's weights by `num_categories`x (32x for
    N1.7) and removes the corresponding ONNX Gather node.
    """
    with torch.no_grad():
        W = layer.W[embodiment_id].clone()
        b = layer.b[embodiment_id].clone()
    layer.W = nn.Parameter(W)
    layer.b = nn.Parameter(b)
    layer.num_categories = 1
    layer.forward = MethodType(_fixed_embodiment_linear_forward, layer)


def _linear_with_pretransposed_weight(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Use one pretransposed ONNX initializer across repeated linear calls."""
    output = torch.matmul(x, cast(torch.Tensor, self._onnx_weight_t))
    if self.bias is not None:
        output = output + self.bias
    return output


def _prepare_linear_weight_for_shared_onnx_export(layer: nn.Module) -> None:
    """Replace a loaded linear weight with one pre-transposed export parameter."""
    if hasattr(layer, "_onnx_weight_t"):
        return
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(layer).__name__}.")

    weight = layer.weight
    with torch.no_grad():
        weight_t = weight.detach().transpose(0, 1).contiguous()
    del layer._parameters["weight"]
    layer.register_parameter(
        "_onnx_weight_t",
        nn.Parameter(weight_t, requires_grad=False),
    )
    layer.forward = MethodType(_linear_with_pretransposed_weight, layer)


class ActionHeadDiTOpt(Gr00tN1d7ActionHead):
    """Adapt attention and unroll all denoising steps into one static forward pass.

    Checkpoint loading folds timestep-only paths into non-persistent FP32 tables;
    runtime indexing prevents step-specific weight baking. Inherited `vlln` and
    `vl_self_attention` remain unused because they belong to the VLM projection.
    """

    model: AlternateVLDiT
    num_inference_timesteps: int
    pos_ids: torch.Tensor
    embodiment_id: torch.Tensor
    _folded_action_timestep: torch.Tensor

    def __init__(self, config: Gr00tN1d7Config, embodiment_id: int) -> None:
        super().__init__(config)

        for block in self.model.transformer_blocks:
            attn = cast(BasicTransformerBlock, block).attn1
            # The split-K/V processor replaces Diffusers' default processor type.
            attn.processor = AttnProcessorCrossAttnKV()  # type: ignore[assignment, unused-ignore]
        for block in self.model.transformer_blocks:
            block.forward = MethodType(
                BasicTransformerBlockCrossAttnKVOpt.forward, block
            )
        self.model.forward = MethodType(AlternateVLDiTCrossAttnKV.forward, self.model)

        self.register_buffer(
            "pos_ids", torch.arange(self.action_horizon, dtype=torch.long)
        )
        self.register_buffer(
            "embodiment_id", torch.tensor([int(embodiment_id)], dtype=torch.int32)
        )

        # t_cont uses a half-open interval [0, 1) -- t=0 is pure noise, and the
        # final step evaluates velocity at t=(N-1)/N but never at t=1.0 (clean data).
        self.timesteps_tensor_names = []
        for t in range(self.num_inference_timesteps):
            name = f"timesteps_tensor_{t}"
            t_cont = t / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            self.register_buffer(name, torch.full(size=(1,), fill_value=t_discretized))
            self.timesteps_tensor_names.append(name)

        self._timestep_folding_prepared = False
        self._timestep_weights_loaded = False
        # Hooks also run when this head is loaded through a containing module.
        self.register_load_state_dict_pre_hook(self._before_timestep_weights_load)
        self.register_load_state_dict_post_hook(self._after_timestep_weights_load)

    def _before_timestep_weights_load(
        self,
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        self._timestep_folding_prepared = False
        self._timestep_weights_loaded = False
        self._timestep_load_errors = error_msgs
        # strict=False is used for the wrapper's extra control buffers, but it
        # must never permit folding random/unloaded conditioning weights.
        required = [
            name
            for name, _ in self.named_parameters()
            if name.startswith(("model.timestep_encoder.", "model.proj_out_1."))
            or ".norm1.linear." in name
        ]
        absent = [prefix + name for name in required if prefix + name not in state_dict]
        if absent:
            raise RuntimeError(f"Missing timestep-folding checkpoint weights: {absent}")

    def _after_timestep_weights_load(
        self, module: nn.Module, incompatible_keys: object
    ) -> None:
        if self._timestep_load_errors:
            return  # The loader will report shape/copy errors; do not bake them.
        self._timestep_weights_loaded = True
        self.fold_timestep_dependencies()

    @staticmethod
    def _set_folded_buffer(module: nn.Module, name: str, value: torch.Tensor) -> None:
        if value.dtype != torch.float32:
            raise RuntimeError(f"Folded buffer {name} must be computed in FP32.")
        # Derived buffers move with .to() and are exported, but cannot override
        # freshly loaded weights with stale values from a saved state_dict.
        module.register_buffer(name, value.detach().contiguous(), persistent=False)

    @torch.no_grad()
    def fold_timestep_dependencies(self, *, force: bool = False) -> None:
        """Precompute fixed-timestep conditioning as FP32 buffers after loading weights."""
        if not self._timestep_weights_loaded:
            raise RuntimeError("Load checkpoint weights before folding timesteps.")
        if self._timestep_folding_prepared and not force:
            return
        self._timestep_folding_prepared = False

        parameter_dtypes = {p.dtype for p in self.parameters() if p.numel() > 0}
        if parameter_dtypes != {torch.float32}:
            raise RuntimeError(
                "Timestep folding requires an FP32 action head; found parameter "
                f"dtypes {sorted(str(dtype) for dtype in parameter_dtypes)}."
            )

        for block_index, block in enumerate(self.model.transformer_blocks):
            if block.norm_type != "ada_norm":
                raise RuntimeError(
                    f"Expected AdaNorm in block {block_index}, got {block.norm_type!r}."
                )

        device = next(self.model.timestep_encoder.parameters()).device
        # Folding under an outer autocast context must still evaluate GEMMs in
        # FP32, rather than cast low-precision results back to FP32 afterwards.
        with torch.autocast(device_type=device.type, enabled=False):
            self._compute_timestep_buffers(device)
        self._timestep_folding_prepared = True

    def _compute_timestep_buffers(self, device: torch.device) -> None:
        action_encodings = []
        timestep_embeddings = []
        final_shifts = []
        final_multipliers = []
        ada_multipliers: list[list[torch.Tensor]] = [
            [] for _ in self.model.transformer_blocks
        ]
        ada_shifts: list[list[torch.Tensor]] = [
            [] for _ in self.model.transformer_blocks
        ]
        for stored_timestep in self.timesteps_tensors:
            timestep = stored_timestep.to(device=device)
            action_timestep = timestep.reshape(1, 1).expand(-1, self.action_horizon)
            action_encoding = self.action_encoder.pos_encoding(action_timestep)
            if not isinstance(action_encoding, torch.Tensor):
                raise RuntimeError(  # noqa: TRY004 -- Internal encoder failure, not a caller argument.
                    "The action timestep encoder must return a tensor, got "
                    f"{type(action_encoding).__name__}."
                )
            if action_encoding.shape != (
                1,
                self.action_horizon,
                self.input_embedding_dim,
            ):
                raise RuntimeError(
                    "Expected a singleton-batch action timestep encoding, got "
                    f"shape {tuple(action_encoding.shape)}."
                )
            action_encodings.append(action_encoding)

            temb = self.model.timestep_encoder(timestep)
            if not isinstance(temb, torch.Tensor) or temb.ndim != 2:
                raise RuntimeError(
                    "Expected the DiT timestep encoder to return [batch, dim], "
                    f"got {type(temb).__name__} with shape "
                    f"{getattr(temb, 'shape', None)}."
                )
            timestep_embeddings.append(temb)

            final_conditioning = self.model.proj_out_1(F.silu(temb))
            final_shift, final_scale = final_conditioning.chunk(2, dim=1)
            final_shifts.append(final_shift[:, None])
            final_multipliers.append(1 + final_scale[:, None])

            for block_index, block in enumerate(self.model.transformer_blocks):
                linear = getattr(block.norm1, "linear", None)
                if not isinstance(linear, nn.Linear):
                    raise RuntimeError(  # noqa: TRY004 -- Invalid internal AdaNorm structure.
                        "Expected AdaNorm norm1.linear in block "
                        f"{block_index}, got {type(linear).__name__}."
                    )
                projected = linear(cast(AdaLayerNorm, block.norm1).silu(temb))
                scale, shift = projected.chunk(2, dim=1)
                ada_multipliers[block_index].append(1 + scale[:, None])
                ada_shifts[block_index].append(shift[:, None])

        self._set_folded_buffer(
            self, "_folded_action_timestep", torch.cat(action_encodings, dim=0)
        )
        self._set_folded_buffer(
            cast(nn.Module, self.model),
            "_folded_temb",
            torch.cat(timestep_embeddings, dim=0),
        )
        self._set_folded_buffer(
            cast(nn.Module, self.model),
            "_folded_final_shift",
            torch.cat(final_shifts, dim=0),
        )
        self._set_folded_buffer(
            cast(nn.Module, self.model),
            "_folded_final_multiplier",
            torch.cat(final_multipliers, dim=0),
        )
        for block_index, block in enumerate(self.model.transformer_blocks):
            self._set_folded_buffer(
                block,
                "_folded_ada_multiplier",
                torch.cat(ada_multipliers[block_index], dim=0),
            )
            self._set_folded_buffer(
                block, "_folded_ada_shift", torch.cat(ada_shifts[block_index], dim=0)
            )

    @property
    def timesteps_tensors(self) -> list[torch.Tensor]:
        return [getattr(self, name) for name in self.timesteps_tensor_names]

    def collapse_embodiment_weights(self) -> None:
        """Fix state/action `CategorySpecificLinear` layers to `self.embodiment_id`,
        removing unused weights.
        """
        eid = int(self.embodiment_id.item())
        for layer in (
            self.state_encoder.layer1,
            self.state_encoder.layer2,
            self.action_encoder.W1,
            self.action_encoder.W2,
            self.action_encoder.W3,
            self.action_decoder.layer1,
            self.action_decoder.layer2,
        ):
            _collapse_category_specific_linear(layer, eid)

    def prepare_self_attention_qkv_for_onnx_export(self) -> None:
        """Fuse loaded Q/K/V weights for the 16 odd self-attention blocks."""
        for idx, block in enumerate(self.model.transformer_blocks):
            if idx % 2 == 1:
                _prepare_self_attention_qkv_for_onnx_export(
                    cast(BasicTransformerBlock, block).attn1
                )

    def prepare_shared_weights_for_onnx_export(self) -> None:
        """Make shared DiT linear weights explicit for ONNX after loading and Q/K/V fusion."""
        repeated_linears: list[nn.Module] = []
        for idx, block in enumerate(self.model.transformer_blocks):
            block = cast(BasicTransformerBlock, block)
            repeated_linears.extend(
                (
                    block.attn1.to_out[0],
                    cast(nn.Module, block.ff.net[0].proj),
                    block.ff.net[2],
                )
            )
            if idx % 2 == 0:
                repeated_linears.append(block.attn1.to_q)
        repeated_linears.append(self.model.proj_out_2)

        if len(repeated_linears) != 113:
            raise RuntimeError(
                f"Expected 113 repeated DiT linear projections, got "
                f"{len(repeated_linears)}."
            )
        for layer in repeated_linears:
            _prepare_linear_weight_for_shared_onnx_export(layer)

    def forward(  # type: ignore[override, unused-ignore]
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        vlm_proj_keys: list[torch.Tensor],
        vlm_proj_values: list[torch.Tensor],
        image_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not self._timestep_folding_prepared:
            raise RuntimeError("Load checkpoint weights before running the folded DiT.")
        B = int(state.shape[0])
        assert isinstance(self.embodiment_id, torch.Tensor)
        embodiment_id = self.embodiment_id.expand(B)

        state_features = self.state_encoder(state, embodiment_id)

        for t in range(self.num_inference_timesteps):
            timestep_index = timestep_indices[:, t]
            action_features = _encode_actions_with_fixed_timestep(
                self.action_encoder,
                actions,
                embodiment_id,
                torch.index_select(self._folded_action_timestep, 0, timestep_index),
            )
            if self.config.add_pos_embed:
                pos_embs = self.position_embedding(self.pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # N1.7 has no future_tokens (use_future_tokens: false): state + actions only.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            model_output = cast(nn.Module, self.model)(
                hidden_states=sa_embs,
                encoder_hidden_states_keys=vlm_proj_keys,
                encoder_hidden_states_values=vlm_proj_values,
                timestep=None,
                image_mask=image_mask,
                backbone_attention_mask=backbone_mask,
                timestep_index=timestep_index,
            )

            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            actions = actions + (1.0 / self.num_inference_timesteps) * pred_velocity

        return actions
