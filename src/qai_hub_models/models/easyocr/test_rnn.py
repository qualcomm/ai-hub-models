# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import pytest
import torch
from torch import nn

from qai_hub_models.models.easyocr.rnn import UnrolledLSTM


@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("seq_len", [1, 5, 16])
@pytest.mark.parametrize("input_size", [8, 32])
@pytest.mark.parametrize("hidden_size", [16, 64])
def test_unrolled_lstm_matches_native(
    bidirectional: bool, bias: bool, seq_len: int, input_size: int, hidden_size: int
) -> None:
    """Test that UnrolledLSTM produces the same output as native LSTM."""
    batch_size = 2

    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=bias,
        batch_first=True,
        bidirectional=bidirectional,
    )
    lstm.eval()

    unrolled_lstm = UnrolledLSTM(lstm)
    unrolled_lstm.eval()

    input_feature = torch.randn(batch_size, seq_len, input_size)

    with torch.no_grad():
        native_output, (native_h, native_c) = lstm(input_feature)
        unrolled_output, (unrolled_h, unrolled_c) = unrolled_lstm(input_feature)

    assert native_output.shape == unrolled_output.shape
    assert torch.allclose(native_output, unrolled_output, atol=1e-6)
    assert torch.allclose(native_h, unrolled_h, atol=1e-6)
    assert torch.allclose(native_c, unrolled_c, atol=1e-6)
