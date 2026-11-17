# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import (
    WhisperAttention,
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperEncoder,
    WhisperEncoderLayer,
    WhisperModel,
)

from qai_hub_models.utils.model_adapters import Conv2dLinear
from qai_hub_models.utils.torch_typing_helpers import Conv2DWithBias, TypedModuleList


def expand_to_rank4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Expands the tensor to rank 4 by adding two singleton dimensions at the end.

    Parameters
    ----------
    tensor
        The input tensor.

    Returns
    -------
    expanded_tensor : torch.Tensor
        The tensor expanded to rank 4.
    """
    return tensor.unsqueeze(-1).unsqueeze(-1)


class SHAAttention(nn.Module):
    """
    Split-Head Attention with per-head Conv2D projections and a single output
    Conv2D projection. This implementation splits the attention heads into
    separate Conv2D projection layers and applies a single output projection
    after concatenating all heads.
    """

    def __init__(self, origin_attn: WhisperAttention) -> None:
        """
        Initialize SHAAttention by copying weights from the original Attention module.

        Parameters
        ----------
        origin_attn
            The original Whisper attention module.
        """
        super().__init__()
        # Copy the configurations from original attention
        self.embed_dim = origin_attn.embed_dim
        self.num_heads = origin_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.config = origin_attn.config
        assert self.config is not None
        self.max_channel = self.config.max_source_positions

        # Ensure embed_dim is divisible by num_heads
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.is_decoder = origin_attn.is_decoder
        self.is_causal = origin_attn.is_causal

        # Initialize Conv2D layers for key, value, and query projections
        self.q_proj_sha: TypedModuleList[Conv2DWithBias] = TypedModuleList(
            Conv2DWithBias(self.embed_dim, self.head_dim, kernel_size=1)
            for _ in range(self.num_heads)
        )
        self.k_proj_sha: TypedModuleList[nn.Conv2d] = TypedModuleList(
            nn.Conv2d(self.embed_dim, self.head_dim, kernel_size=1, bias=False)
            for _ in range(self.num_heads)
        )
        self.v_proj_sha: TypedModuleList[Conv2DWithBias] = TypedModuleList(
            Conv2DWithBias(self.embed_dim, self.head_dim, kernel_size=1)
            for _ in range(self.num_heads)
        )
        # Copy and convert the weights from original attention for qkv_proj_sha
        for i in range(self.num_heads):
            start, end = i * self.head_dim, (i + 1) * self.head_dim

            # q_proj_sha weights, bias
            q_weight = (
                expand_to_rank4(origin_attn.q_proj.weight.data[start:end, :].clone())
                * self.scaling
            )
            self.q_proj_sha[i].weight.data.copy_(q_weight)
            q_bias = origin_attn.q_proj.bias.data[start:end].clone() * self.scaling
            self.q_proj_sha[i].bias.data.copy_(q_bias)

            # k_proj_sha weights
            k_weight = expand_to_rank4(
                origin_attn.k_proj.weight.data[start:end, :].clone()
            )
            self.k_proj_sha[i].weight.data.copy_(k_weight)

            # v_proj_sha weights, bias
            v_weight = expand_to_rank4(
                origin_attn.v_proj.weight.data[start:end, :].clone()
            )
            self.v_proj_sha[i].weight.data.copy_(v_weight)
            v_bias = origin_attn.v_proj.bias.data[start:end].clone()
            self.v_proj_sha[i].bias.data.copy_(v_bias)

        # Initialize output projection layer
        self.out_proj = Conv2DWithBias(self.embed_dim, self.embed_dim, kernel_size=1)
        # Copy and convert the weights from original attention for out_proj
        out_weight = expand_to_rank4(origin_attn.out_proj.weight.data.clone())
        self.out_proj.weight.data.copy_(out_weight)
        self.out_proj.bias.data.copy_(origin_attn.out_proj.bias.data.clone())

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        Forward-pass routine for SHAAttention.

        Parameters
        ----------
        hidden_states
            The input hidden states.
        past_key_value
            Past key and value states for attention.
        attention_mask
            Attention mask.

        Returns
        -------
        attn_output : torch.Tensor
            The output of the attention mechanism.
        past_key_value : tuple[torch.Tensor, torch.Tensor] | None
            The past key-value states.
        """
        is_cross_attention = self.is_decoder and past_key_value is not None
        past_key_value_rt = None
        bsz, _, tgt_len, _ = hidden_states.size()
        # Rearrange dimensions for Conv2D
        hidden_states = hidden_states.permute(0, 3, 1, 2)

        # Compute query states for each head
        query_states = [
            q_proj(hidden_states).permute(0, 2, 3, 1) for q_proj in self.q_proj_sha
        ]

        if (
            is_cross_attention
            and past_key_value is not None
            and self.is_causal is False
        ):
            assert len(past_key_value) > 1
            key_states = torch.split(past_key_value[0], 1)
            value_states = torch.split(past_key_value[1], 1)
        elif past_key_value is not None:
            assert len(past_key_value) > 1
            key_states = [
                k_proj(hidden_states).permute(0, 2, 1, 3) for k_proj in self.k_proj_sha
            ]
            value_states = [
                v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
            ]
            past_keys = torch.split(past_key_value[0], 1)
            key_states = [
                torch.cat([past_keys[i], key_state], dim=-1)
                for i, key_state in enumerate(key_states)
            ]
            past_values = torch.split(past_key_value[1], 1)
            value_states = [
                torch.cat([past_values[i], value_state], dim=-2)
                for i, value_state in enumerate(value_states)
            ]
        else:
            key_states = [
                k_proj(hidden_states).permute(0, 2, 1, 3) for k_proj in self.k_proj_sha
            ]
            value_states = [
                v_proj(hidden_states).permute(0, 2, 3, 1) for v_proj in self.v_proj_sha
            ]

        if self.is_decoder and self.is_causal is True:
            past_key_value_rt = (
                torch.cat(key_states, dim=0)[:, :, :, 1:].reshape(
                    self.num_heads, bsz, self.head_dim, -1
                ),
                torch.cat(value_states, dim=0)[:, :, 1:, :].reshape(
                    self.num_heads, bsz, -1, self.head_dim
                ),
            )

        src_len = value_states[0].size(-2)
        attn_weights = [
            torch.matmul(query_state, key_states[i])
            for i, query_state in enumerate(query_states)
        ]

        if attention_mask is not None:
            attn_weights = [
                attn_weight.view(bsz, 1, tgt_len, src_len) + attention_mask
                for attn_weight in attn_weights
            ]

        attn_weights = [
            attn_weight.view(bsz, 1, tgt_len, src_len) for attn_weight in attn_weights
        ]

        attn_weights = [
            nn.functional.softmax(attn_weight, dim=-1) for attn_weight in attn_weights
        ]

        attn_output = [
            torch.matmul(attn_weight, value_states[i])
            for i, attn_weight in enumerate(attn_weights)
        ]

        final_attn = torch.cat(attn_output, dim=-1).permute(0, 3, 1, 2)
        final_attn = self.out_proj(final_attn).permute(0, 2, 3, 1)
        return final_attn, past_key_value_rt


class QcWhisperEncoderLayer(nn.Module):
    """
    Optimized Whisper Encoder Layer that replaces linear layers with Conv2D layers.

    Parameters
    ----------
    orig_layer
        The original Whisper encoder layer.
    """

    def __init__(self, orig_layer: WhisperEncoderLayer) -> None:
        super().__init__()

        self.self_attn_layer_norm = orig_layer.self_attn_layer_norm
        self.self_attn = SHAAttention(orig_layer.self_attn)

        self.final_layer_norm = orig_layer.final_layer_norm

        # Replace linear layers with Conv2D layers using Conv2dLinear
        self.fc1 = Conv2dLinear(orig_layer.fc1)
        self.activation_fn = orig_layer.activation_fn
        self.fc2 = Conv2dLinear(orig_layer.fc2)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """
        Forward-pass routine for the optimized Whisper encoder layer.

        Parameters
        ----------
        hidden_states
            The input hidden states.

        Returns
        -------
        hidden_states : torch.Tensor
            The output hidden states.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=None,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states.permute(0, 3, 1, 2)))
        hidden_states = self.fc2(hidden_states).permute(0, 2, 3, 1)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class QcWhisperDecoderLayer(nn.Module):
    """
    Optimized Whisper Decoder Layer that replaces linear layers with Conv2D layers.

    Parameters
    ----------
    orig_layer
        The original Whisper decoder layer.
    """

    def __init__(self, orig_layer: WhisperDecoderLayer) -> None:
        super().__init__()
        self.self_attn_layer_norm = orig_layer.self_attn_layer_norm
        self.self_attn = SHAAttention(orig_layer.self_attn)

        self.encoder_attn_layer_norm = orig_layer.encoder_attn_layer_norm
        self.encoder_attn = SHAAttention(orig_layer.encoder_attn)

        self.final_layer_norm = orig_layer.final_layer_norm
        self.activation_fn = orig_layer.activation_fn

        # Replace linear layers with Conv2D layers using Conv2dLinear
        self.fc1 = Conv2dLinear(orig_layer.fc1)
        self.fc2 = Conv2dLinear(orig_layer.fc2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: torch.Tensor | None = None,
        cross_attn_past_key_value: tuple[torch.Tensor] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Forward-pass routine for the optimized Whisper decoder layer.

        Parameters
        ----------
        hidden_states
            The input hidden states.
        attention_mask
            The attention mask.
        past_key_value
            The past key and value states.
        cross_attn_past_key_value
            The past key and value states for cross-attention.

        Returns
        -------
        output : tuple[torch.Tensor, ...]
            hidden_states
                The output hidden states.
            present_key_value
                The present key and value states (optional).
        """
        # Self-attention block
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # Cross-attention block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        if cross_attn_past_key_value is None:
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=None,
            past_key_value=cross_attn_past_key_value,
        )
        hidden_states = residual + hidden_states

        # Feed-forward block
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states.permute(0, 3, 1, 2)))
        hidden_states = self.fc2(hidden_states).permute(0, 2, 3, 1)
        hidden_states = residual + hidden_states

        return (
            hidden_states,
            present_key_value,
        )


class QcWhisperEncoder(nn.Module):
    """
    Optimized Whisper Encoder with Conv2D layers and attention mechanisms.
    This class modifies the original WhisperEncoder to use Conv2D layers for
    key and value projections in the attention mechanism.
    """

    def __init__(
        self, orig_encoder: WhisperEncoder, qc_decoder: QcWhisperDecoder
    ) -> None:
        """
        Initialize the QcWhisperEncoder by copying weights from the original encoder and the optimized decoder.

        Parameters
        ----------
        orig_encoder
            The original Whisper encoder.
        qc_decoder
            The optimized Whisper decoder.
        """
        super().__init__()
        self.config = orig_encoder.config
        embed_dim = self.config.d_model
        self.num_heads = self.config.decoder_attention_heads
        self.head_dim = self.config.d_model // self.num_heads
        self.num_mel_bins = orig_encoder.num_mel_bins

        # Replace orig_encoder.conv1(Conv1d) with Conv2d
        self.conv1 = Conv2DWithBias(
            self.num_mel_bins, embed_dim, kernel_size=(1, 3), padding=(0, 1)
        )
        # Replace orig_encoder.conv2(Conv1d) with Conv2d
        self.conv2 = Conv2DWithBias(
            embed_dim, embed_dim, kernel_size=(1, 3), stride=2, padding=(0, 1)
        )
        # Copy weights from original encoder
        # Unsqueeze(-2) is used to add a singleton dimension to match the expected input shape of Conv2D layers
        self.conv1.weight.data.copy_(
            orig_encoder.conv1.weight.data.clone().unsqueeze(-2)
        )
        assert orig_encoder.conv1.bias is not None
        self.conv1.bias.data.copy_(orig_encoder.conv1.bias.data.clone())
        self.conv2.weight.data.copy_(
            orig_encoder.conv2.weight.data.clone().unsqueeze(-2)
        )
        assert orig_encoder.conv2.bias is not None
        self.conv2.bias.data.copy_(orig_encoder.conv2.bias.data.clone())

        self.embed_positions = nn.Parameter(
            orig_encoder.embed_positions.weight[: self.config.max_source_positions, :]
        )

        # Initialize encoder layers
        self.layers: TypedModuleList[QcWhisperEncoderLayer] = TypedModuleList(
            [
                QcWhisperEncoderLayer(cast(WhisperEncoderLayer, layer))
                for layer in orig_encoder.layers
            ]
        )
        self.layer_norm = orig_encoder.layer_norm

        # Initialize attention projection layers
        self.encoder_k_proj_sha: TypedModuleList[TypedModuleList[nn.Conv2d]] = (
            TypedModuleList(
                TypedModuleList(
                    nn.Conv2d(embed_dim, self.head_dim, kernel_size=1, bias=False)
                    for _ in range(self.num_heads)
                )
                for _ in range(self.config.decoder_layers)
            )
        )
        self.encoder_v_proj_sha: TypedModuleList[TypedModuleList[Conv2DWithBias]] = (
            TypedModuleList(
                TypedModuleList(
                    Conv2DWithBias(embed_dim, self.head_dim, kernel_size=1)
                    for _ in range(self.num_heads)
                )
                for _ in range(self.config.decoder_layers)
            )
        )

        # Copy cross-attention weights from the optimized decoder to the encoder
        for i in range(qc_decoder.config.decoder_layers):
            for num_head in range(qc_decoder.config.decoder_attention_heads):
                # Get the encoder attention layer from the decoder
                layer = qc_decoder.layers[i].encoder_attn

                # Copy the key projection weights from the decoder to the encoder
                self.encoder_k_proj_sha[i][num_head].weight.data.copy_(
                    layer.k_proj_sha[num_head].weight.data
                )

                # Copy the value projection weights from the decoder to the encoder
                self.encoder_v_proj_sha[i][num_head].weight.data.copy_(
                    layer.v_proj_sha[num_head].weight.data
                )

                # Copy the value projection biases from the decoder to the encoder
                self.encoder_v_proj_sha[i][num_head].bias.data.copy_(
                    layer.v_proj_sha[num_head].bias.data
                )

    def forward(
        self,
        input_features: torch.Tensor,
    ) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...],]:
        """
        Forward-pass routine for the QcWhisperEncoder.

        Parameters
        ----------
        input_features
            The input features for the encoder.

        Returns
        -------
        next_cache : tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...],]
            The output of the encoder, including the next decoder cache.
        """
        input_features = input_features.unsqueeze(0).permute(1, 2, 0, 3)
        input_embeds = nn.functional.gelu(self.conv1(input_features))
        input_embeds = nn.functional.gelu(self.conv2(input_embeds))
        input_embeds = input_embeds.permute(0, 2, 3, 1)
        embed_pos = self.embed_positions
        hidden_states = input_embeds + embed_pos
        for encoder_layer in self.layers:
            layer_output = encoder_layer(hidden_states)
            hidden_states = layer_output[0]
        hidden_states = self.layer_norm(hidden_states)
        # Add encoder attn
        next_cache = []
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        for idx in range(self.config.decoder_layers):
            key_states = [
                k_proj(hidden_states).permute(0, 2, 1, 3).contiguous()
                for k_proj in self.encoder_k_proj_sha[idx]
            ]
            value_states = [
                v_proj(hidden_states).permute(0, 2, 3, 1).contiguous()
                for v_proj in self.encoder_v_proj_sha[idx]
            ]
            past_key_value = (
                torch.cat(key_states, dim=0),
                torch.cat(value_states, dim=0),
            )
            next_cache.append(past_key_value)
        return (tuple(next_cache),)


class QcWhisperDecoder(nn.Module):
    """
    Optimized Whisper Decoder with Conv2D layers and attention mechanisms.
    This class modifies the original WhisperDecoder to use Conv2D layers for
    key and value projections in the attention mechanism.
    """

    def __init__(self, orig_decoder: WhisperDecoder) -> None:
        """
        Initialize the QcWhisperDecoder by copying weights from the original decoder.

        Parameters
        ----------
        orig_decoder
            The original Whisper decoder.
        """
        super().__init__()
        self.config = orig_decoder.config

        self.embed_tokens = orig_decoder.embed_tokens
        self.embed_positions = orig_decoder.embed_positions

        self.layers: TypedModuleList[QcWhisperDecoderLayer] = TypedModuleList(
            [
                QcWhisperDecoderLayer(cast(WhisperDecoderLayer, layer))
                for layer in orig_decoder.layers
            ]
        )

        self.layer_norm = orig_decoder.layer_norm
        self.proj_out = nn.Conv2d(
            self.config.d_model, self.config.vocab_size, kernel_size=1, bias=False
        )
        self.proj_out.weight.data.copy_(
            expand_to_rank4(orig_decoder.embed_tokens.weight.data)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: torch.Tensor,
        cross_attn_past_key_value: torch.Tensor,
        position_ids: torch.BoolTensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Forward-pass routine for the QcWhisperDecoder.

        Parameters
        ----------
        input_ids
            The input IDs for the decoder.
        attention_mask
            The attention mask for the decoder.
        past_key_values
            The past key values for the decoder.
        cross_attn_past_key_value
            The past key values for cross-attention.
        position_ids
            The position IDs for the decoder.

        Returns
        -------
        lm_logits : torch.Tensor
            The output logits of the decoder.
        next_decoder_cache : tuple[torch.Tensor, ...]
            The next cache for the decoder.
        """
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        input_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(
            input_ids,
            past_key_values_length=past_key_values_length,
            position_ids=position_ids,
        )
        input_embeds = input_embeds.unsqueeze(0)
        hidden_states = input_embeds + positions
        next_decoder_cache = []
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            layer_output = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cross_attn_past_key_value=cross_attn_past_key_value[idx],
            )
            hidden_states = layer_output[0]
            next_decoder_cache.append(layer_output[1])
        hidden_states = self.layer_norm(hidden_states)
        lm_logits = self.proj_out(hidden_states.permute(0, 3, 1, 2))
        return lm_logits, tuple(next_decoder_cache)


def replace_component(model: WhisperModel, component_type: str) -> None:
    """
    Replaces components in the Whisper model with their optimized counterparts,
    including converting multi-head attention (MHA) to self-attention (SHA) and linear layers to Conv2D.

    Parameters
    ----------
    model
        The Whisper model to modify.
    component_type
        The type of component to replace ('encoder' or 'decoder').

    Raises
    ------
    ValueError
        If an unknown component_type is provided.
    """
    if component_type == "encoder":
        orig_encoder_module: type[WhisperEncoder] = WhisperEncoder
        qc_encoder_module: type[QcWhisperEncoder] = QcWhisperEncoder
        module = model.get_encoder()
        # Check if the decoder has been updated to QcWhisperDecoder
        if not isinstance(model.get_decoder(), QcWhisperDecoder):
            raise ValueError(
                "Please update the decoder to QcWhisperDecoder before updating the encoder."
            )
    elif component_type == "decoder":
        orig_decoder_module: type[WhisperDecoder] = WhisperDecoder
        qc_decoder_module: type[QcWhisperDecoder] = QcWhisperDecoder
        module = model.get_decoder()
    else:
        raise ValueError(f"Unknown component_type: {component_type}")

    # Replace the top-level module with its optimized counterpart
    for name, submodule in model.named_children():
        if component_type == "encoder" and isinstance(submodule, orig_encoder_module):
            # For encoder, pass both the original encoder and the optimized decoder
            setattr(
                model,
                name,
                qc_encoder_module(module, cast(QcWhisperDecoder, model.get_decoder())),
            )
        elif component_type == "decoder" and isinstance(submodule, orig_decoder_module):
            # For decoder, only pass the original decoder
            setattr(model, name, qc_decoder_module(module))


def monkey_patch_model(model: WhisperModel) -> None:
    """
    Applies a series of modifications to the Whisper model, including replacing components,
    attention modules, and moving cross-key/value pairs.

    Parameters
    ----------
    model
        The Whisper model to modify.
    """
    # Replace decoder components with their optimized counterparts
    replace_component(model, "decoder")

    # Replace encoder components with their optimized counterparts
    replace_component(model, "encoder")
