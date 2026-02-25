# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from typing import TYPE_CHECKING

import torch
from melo import commons
from torch import Tensor, nn

if TYPE_CHECKING:
    import melo.attentions
    import melo.models


class OptimizedTextEncoder(nn.Module):
    """
    An optimized TextEncoder of the original one from
      https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/melo/models.py#L311
    the optimization eliminates the two sqrt in the original code.
    """

    def __init__(self, original_encoder: "melo.models.TextEncoder") -> None:
        super().__init__()
        self.out_channels: int = original_encoder.out_channels
        self.hidden_channels: int = original_encoder.hidden_channels
        self.filter_channels: int = original_encoder.filter_channels
        self.n_heads: int = original_encoder.n_heads
        self.n_layers: int = original_encoder.n_layers
        self.kernel_size: int = original_encoder.kernel_size
        self.p_dropout: float = original_encoder.p_dropout
        self.gin_channels: int = original_encoder.gin_channels

        self.emb: nn.Embedding = original_encoder.emb
        self.tone_emb: nn.Embedding = original_encoder.tone_emb
        self.language_emb: nn.Embedding = original_encoder.language_emb
        self.bert_proj: nn.Conv1d = original_encoder.bert_proj
        self.ja_bert_proj: nn.Conv1d = original_encoder.ja_bert_proj
        self.encoder: melo.attentions.Encoder = original_encoder.encoder
        self.proj: nn.Conv1d = original_encoder.proj

        self.hidden_scale = self.hidden_channels**0.5

    def forward(
        self,
        x: Tensor,
        x_lengths: Tensor,
        tone: Tensor,
        language: Tensor,
        bert: Tensor,
        ja_bert: Tensor,
        g: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Process the phones and tone of the input text, use bert model to tokenize the text

        Parameters
        ----------
        x
            the phones of input text, shape of (1, max_seq_len), i.e., [1, 512]
        x_lengths
            the length of phones, shape of [1]
        tone
            the tone of input text, shape of (1, max_seq_len), i.e., [1, 512]
        language
            phones of each token, shape of (1, max_seq_len), i.e., [1, 512]
        bert
            output feature of bert model, shape of (1, bert_feature_dim, max_seq_len), i.e., [1, 1024, 512]
        ja_bert
            similar to bert, shape of (1, ja_bert_feature_dim, max_seq_len), i.e., [1, 768, 512]
        g
            embedding of speaker ID, shape of (1, speaker_embed_dim, 1)

        Returns
        -------
        x : Tensor
            shape of (1, max_seq_len)
        m : Tensor
            shape of (1, encoder_hidden_dim, max_seq_len)
        logs : Tensor
            shape of (1, encoder_hidden_dim, max_seq_len)
        x_mask : Tensor
            shape of (1, 1, max_seq_len)
        """
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)

        # TODO(17781): Undo clamp after we add a tracing option to set the input value range.
        x = torch.clamp(x, 0, self.emb.num_embeddings - 1)
        tone = torch.clamp(tone, 0, self.tone_emb.num_embeddings - 1)
        language = torch.clamp(language, 0, self.language_emb.num_embeddings - 1)
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
        )
        x = x * self.hidden_scale

        x = x.transpose(1, -1)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask, g=g)

        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class OptimizedDurationPredictor(nn.Module):
    def __init__(self, original_dp: "melo.models.DurationPredictor") -> None:
        super().__init__()

        self.in_channels = original_dp.in_channels
        self.filter_channels = original_dp.filter_channels
        self.kernel_size = original_dp.kernel_size
        self.p_dropout = original_dp.p_dropout
        self.gin_channels = original_dp.gin_channels

        self.conv_1 = original_dp.conv_1
        self.norm_1 = original_dp.norm_1
        self.conv_2 = original_dp.conv_2
        self.norm_2 = original_dp.norm_2
        self.proj = original_dp.proj
        if self.gin_channels != 0:
            self.cond = original_dp.cond
        self.drop = nn.Dropout(original_dp.p_dropout)

    def forward(self, x: Tensor, x_mask: Tensor, g: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x
            the phones of input text, shape of (1, MAX_SEQ_LEN), i.e., [1, 512]
        x_mask
            the length of phones, shape of [1]
        g
            the tone of input text, shape of (1, MAX_SEQ_LEN), i.e., [1, 512]

        Returns
        -------
        Tensor
            shape of (1, MAX_SEQ_LEN), the predicted duration
        """
        x = torch.detach(x)

        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)  # Inplace operation
        x = self.norm_1(x)
        x = self.drop(x)

        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        x = self.proj(x * x_mask)

        return x * x_mask
