# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerMaskPredictor,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention,
)
from transformers.models.swin.modeling_swin import SwinLayer, SwinSelfAttention

from qai_hub_models.models.templates.corevision.window_partitioning import (
    window_partition_5d,
    window_unpartition_5d,
)


def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: Tensor | list[tuple],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """
    Copied from transformers/models/mask2former/modeling_mask2former.py
    Qualcomm modifications added to optimize inference on NPU.
    """
    batch_size, _, num_heads, hidden_dim = value.shape

    # -- Begin Qualcomm Change
    num_queries, num_heads, num_points, _ = sampling_locations.shape
    num_queries //= batch_size
    # -- End Qualcomm Change

    value_list = value.split(
        [height * width for height, width in value_spatial_shapes], dim=1
    )
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    # -- Begin Qualcomm Change
    sampling_grids = sampling_grids.split(num_points // len(value_spatial_shapes), 2)
    # -- End Qualcomm Change

    for level_id, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        value_l_ = (
            value_list[level_id]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch_size * num_heads, hidden_dim, height, width)
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2

        # -- Begin Qualcomm Change
        sampling_grid_l_ = sampling_grids[level_id].transpose(0, 1)
        # -- End Qualcomm Change

        # batch_size*num_heads, hidden_dim, num_queries, num_points
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        # -- Begin Qualcomm Change
        batch_size * num_heads,
        1,
        num_queries,
        num_points,
        # -- End Qualcomm Change
    )
    output = (
        # -- Begin Qualcomm Change
        (torch.concat(sampling_value_list, dim=-1) * attention_weights)
        # -- End Qualcomm Change
        .sum(-1)
        .view(batch_size, num_heads * hidden_dim, num_queries)
    )
    return output.transpose(1, 2).contiguous()


class PatchedMask2FormerPixelDecoderEncoderMultiscaleDeformableAttention(nn.Module):
    """
    Copied from transformers/models/mask2former/modeling_mask2former.py::Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
    Qualcomm modifications added to optimize inference on NPU.
    """

    def __init__(
        self, other: Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention
    ) -> None:
        super().__init__()
        self.im2col_step = other.im2col_step

        self.d_model = other.d_model
        self.n_levels = other.n_levels
        self.n_heads = other.n_heads
        self.n_points = other.n_points

        self.sampling_offsets = other.sampling_offsets
        self.attention_weights = other.attention_weights
        self.value_proj = other.value_proj
        self.output_proj = other.output_proj

    def with_pos_embed(
        self, tensor: torch.Tensor, position_embeddings: Tensor | None
    ) -> torch.Tensor:
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        position_embeddings: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # -- Begin Qualcomm Change
        if spatial_shapes_list is None:
            spatial_shapes_list = []
        assert encoder_hidden_states is not None
        assert reference_points is not None
        # -- End Qualcomm Change

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        total_elements = sum(height * width for height, width in spatial_shapes_list)
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure to align the spatial shapes with the sequence length of the encoder hidden states"
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            # we invert the attention_mask
            value = value.masked_fill(attention_mask[..., None], float(0))
        value = value.view(
            batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )
        sampling_offsets = self.sampling_offsets(hidden_states).view(
            # -- Begin Qualcomm Change
            batch_size * num_queries,
            self.n_heads,
            self.n_levels * self.n_points,
            2,
            # -- End Qualcomm Change
        )
        attention_weights = self.attention_weights(hidden_states).view(
            batch_size, num_queries, self.n_heads, self.n_levels * self.n_points
        )

        # -- Begin Qualcomm Change
        attention_weights = nn.functional.softmax(attention_weights, -1)
        # -- End Qualcomm Change

        # batch_size, num_queries, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(
                [[shape[1], shape[0]] for shape in spatial_shapes_list],
                # -- Begin Qualcomm Change
                dtype=torch.int32,
                # -- End Qualcomm Change
                device=reference_points.device,
            )

            # -- Begin Qualcomm Change
            reference_points = (
                reference_points.reshape(-1, self.n_levels, 1, 2)
                .repeat(1, 1, self.n_points, 1)
                .reshape(-1, 1, self.n_levels * self.n_points, 2)
            )
            sampling_locations = (
                reference_points
                + sampling_offsets
                / offset_normalizer.unsqueeze(-2)
                .repeat(1, self.n_points, 1)
                .reshape(1, 1, -1, 2)
            )
            # -- End Qualcomm Change
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attention(
            value, spatial_shapes_list, sampling_locations, attention_weights
        )
        output = self.output_proj(output)

        return output, attention_weights


class PatchedMask2FormerMaskPredictor(nn.Module):
    """
    Copied from transformers/models/mask2former/modeling_mask2former.py::Mask2FormerMaskPredictor
    Qualcomm modifications added to optimize inference on NPU.
    """

    def __init__(self, other: Mask2FormerMaskPredictor) -> None:
        super().__init__()
        self.hidden_size = other.hidden_size
        self.num_heads = other.num_heads
        self.mask_embedder = other.mask_embedder

    def forward(
        self,
        outputs: torch.Tensor,
        pixel_embeddings: torch.Tensor,
        attention_mask_target_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))

        # Sum up over the channels

        # -- Begin Qualcomm Change
        # Equivalent to einsum('bqc, bchw -> bqhw') but jit friendly
        batch_size, num_queries, num_channels = mask_embeddings.shape
        _, _, height, width = pixel_embeddings.shape
        pixel_embeddings = pixel_embeddings.reshape(batch_size, num_channels, -1)
        outputs_mask = torch.matmul(mask_embeddings, pixel_embeddings).reshape(
            batch_size, num_queries, height, width
        )
        # -- End Qualcomm Change

        attention_mask = nn.functional.interpolate(
            outputs_mask,
            size=attention_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )

        attention_mask = (
            attention_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
        )
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()

        return outputs_mask, attention_mask


class PatchedSwinSelfAttention(nn.Module):
    """
    Copied from transformers/models/swin/modeling_swin.py::SwinSelfAttention
    Qualcomm modifications added to optimize inference on NPU.
    """

    def __init__(self, other: SwinSelfAttention) -> None:
        super().__init__()
        self.num_attention_heads = other.num_attention_heads
        self.attention_head_size = other.attention_head_size
        self.all_head_size = other.all_head_size
        self.window_size = cast(tuple[int, int], other.window_size)
        self.relative_position_bias_table = other.relative_position_bias_table
        self.relative_position_index: torch.Tensor
        self.register_buffer(
            "relative_position_index", other._buffers["relative_position_index"]
        )
        self.query = other.query
        self.key = other.key
        self.value = other.value
        self.dropout = other.dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.FloatTensor | None = None,
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[torch.Tensor]:
        batch_size, dim, _num_channels = hidden_states.shape
        hidden_shape = (batch_size, dim, -1, self.attention_head_size)

        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
            # -- Begin Qualcomm Change
            attention_scores = attention_scores.view(
                -1, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1)
            attention_scores = attention_scores.view(
                -1, self.num_attention_heads, dim, dim
            )
            # -- End Qualcomm Change

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (*context_layer.size()[:-2], self.all_head_size)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return cast(tuple[torch.Tensor], outputs)


class PatchedSwinLayer(nn.Module):
    """
    Copied from transformers/models/swin/modeling_swin.py::SwinLayer
    Qualcomm modifications added to optimize inference on NPU.
    """

    def __init__(self, other: SwinLayer) -> None:
        super().__init__()
        self.chunk_size_feed_forward = other.chunk_size_feed_forward
        self.shift_size: int = cast(int, other.shift_size)
        self.window_size: int = cast(int, other.window_size)
        self.input_resolution = other.input_resolution
        self.layernorm_before = other.layernorm_before
        self.attention = other.attention
        self.drop_path = other.drop_path
        self.layernorm_after = other.layernorm_after
        self.intermediate = other.intermediate
        self.output = other.output

    def get_attn_mask(
        self, height: int, width: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor | None:
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype, device=device)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # -- Begin Qualcomm Change
            mask_windows = window_partition_5d(img_mask, self.window_size)[0]
            # -- End Qualcomm Change
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(
        self, hidden_states: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: tuple[int, int],
        head_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        always_partition: bool | None = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # -- Begin Qualcomm Change
        assert always_partition, "always_partition must be true to use this patch"
        # -- End Qualcomm Change

        height, width = input_dimensions
        batch_size, _, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(
                hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        # -- Begin Qualcomm Change
        hidden_states_windows, padded_hw = window_partition_5d(
            shifted_hidden_states, self.window_size
        )
        assert padded_hw[0] == height_pad
        assert padded_hw[1] == width_pad
        # -- End Qualcomm Change

        hidden_states_windows = hidden_states_windows.view(
            -1, self.window_size * self.window_size, channels
        )
        attn_mask = self.get_attn_mask(
            height_pad,
            width_pad,
            dtype=hidden_states.dtype,
            device=hidden_states_windows.device,
        )

        attention_outputs = self.attention(
            hidden_states_windows,
            attn_mask,
            head_mask,
            output_attentions=output_attentions,
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(
            -1, self.window_size, self.window_size, channels
        )
        # -- Begin Qualcomm Change
        shifted_windows = window_unpartition_5d(
            attention_windows,
            self.window_size,
            padded_hw,
            padded_hw,
        )
        # -- End Qualcomm Change

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(
                shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)

        hidden_states = shortcut + self.drop_path(attention_windows)

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        layer_outputs = (
            (layer_output, attention_outputs[1])
            if output_attentions
            else (layer_output,)
        )
        return cast(tuple[torch.Tensor, torch.Tensor], layer_outputs)
