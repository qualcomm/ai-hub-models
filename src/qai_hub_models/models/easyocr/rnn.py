# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from typing import cast

import torch
from torch import nn


class UnrolledLSTM(nn.Module):
    """
    A wrapper around nn.LSTM that unrolls the computation to avoid using the
    built-in LSTM forward pass, which may not be well-supported on all target
    runtimes.

    Parameters
    ----------
    lstm
        PyTorch LSTM module (must be single layer, with or without bidirectional).

    Notes
    -----
    PyTorch's LSTM uses a compiled function under the hood that always exports
    to an LSTM op in ONNX. This module avoids that by implementing the LSTM using LSTMCell modules.
    """

    def __init__(self, lstm: nn.LSTM) -> None:
        super().__init__()
        assert lstm.num_layers == 1, "Only single-layer LSTMs are supported"
        assert lstm.batch_first, "Only batch_first=True is supported"
        assert lstm.proj_size == 0, "Projection LSTMs (proj_size > 0) are not supported"

        self.hidden_size = lstm.hidden_size
        self.bidirectional = lstm.bidirectional

        # Create forward cell with shared weights
        self.forward_cell = nn.LSTMCell(
            lstm.input_size, lstm.hidden_size, bias=lstm.bias
        )
        self.forward_cell.weight_ih = cast(nn.Parameter, lstm.weight_ih_l0)
        self.forward_cell.weight_hh = cast(nn.Parameter, lstm.weight_hh_l0)
        if lstm.bias:
            self.forward_cell.bias_ih = cast(nn.Parameter, lstm.bias_ih_l0)
            self.forward_cell.bias_hh = cast(nn.Parameter, lstm.bias_hh_l0)

        # Create backward cell with shared weights
        if lstm.bidirectional:
            self.backward_cell = nn.LSTMCell(
                lstm.input_size, lstm.hidden_size, bias=lstm.bias
            )
            self.backward_cell.weight_ih = cast(nn.Parameter, lstm.weight_ih_l0_reverse)
            self.backward_cell.weight_hh = cast(nn.Parameter, lstm.weight_hh_l0_reverse)
            if lstm.bias:
                self.backward_cell.bias_ih = cast(nn.Parameter, lstm.bias_ih_l0_reverse)
                self.backward_cell.bias_hh = cast(nn.Parameter, lstm.bias_hh_l0_reverse)

    def _run_sequence(
        self, cell: nn.LSTMCell, input_feature: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run an LSTM cell over the input sequence."""
        batch_size = input_feature.shape[0]
        seq_len = input_feature.shape[1]

        h = torch.zeros(
            batch_size,
            self.hidden_size,
            device=input_feature.device,
            dtype=input_feature.dtype,
        )
        c = torch.zeros(
            batch_size,
            self.hidden_size,
            device=input_feature.device,
            dtype=input_feature.dtype,
        )

        time_steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        outputs = []
        for t in time_steps:
            h, c = cell(input_feature[:, t, :], (h, c))
            outputs.append(h)

        if reverse:
            outputs.reverse()

        return torch.stack(outputs, dim=1), h, c

    def forward(
        self, input_feature: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        input_feature
            Input tensor of shape [batch_size, seq_len, input_size].

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape:
            - [batch_size, seq_len, hidden_size] for unidirectional LSTM
            - [batch_size, seq_len, 2 * hidden_size] for bidirectional LSTM
              (forward and backward outputs concatenated along the last dimension)
        hidden : tuple[torch.Tensor, torch.Tensor]
            Tuple of (h_n, c_n) final hidden states.
        """
        forward_output, h_fwd, c_fwd = self._run_sequence(
            self.forward_cell, input_feature
        )

        if not self.bidirectional:
            return forward_output, (h_fwd.unsqueeze(0), c_fwd.unsqueeze(0))

        backward_output, h_bwd, c_bwd = self._run_sequence(
            self.backward_cell, input_feature, reverse=True
        )

        output = torch.cat([forward_output, backward_output], dim=2)
        h_n = torch.stack([h_fwd, h_bwd], dim=0)
        c_n = torch.stack([c_fwd, c_bwd], dim=0)
        return output, (h_n, c_n)
