# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np
import pytest
import torch

from qai_hub_models.utils.compare import (
    compare_psnr,
    compute_mae,
    compute_max_abs_diff,
    compute_mse,
    compute_top_k_accuracy,
)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_compare_psnr(as_tensor: bool) -> None:
    a = np.array([0.999], dtype=np.float32)
    b = np.array([1.0], dtype=np.float32)
    output_a = torch.from_numpy(a) if as_tensor else a
    output_b = torch.from_numpy(b) if as_tensor else b

    # Automatic data-range estimation gives approximately 60 dB.
    compare_psnr(output_a, output_b, psnr_threshold=30)
    with pytest.raises(AssertionError):
        compare_psnr(output_a, output_b, psnr_threshold=70)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_compare_psnr_custom_epsilons(as_tensor: bool) -> None:
    a = np.array([0.9], dtype=np.float32)
    b = np.array([1.0], dtype=np.float32)
    output_a = torch.from_numpy(a) if as_tensor else a
    output_b = torch.from_numpy(b) if as_tensor else b

    # PSNR = 20 * log10((1 + 0.5) / (0.1 + 0.1)), approximately 17.5 dB.
    compare_psnr(output_a, output_b, psnr_threshold=17, eps=0.5, eps2=0.1)
    with pytest.raises(AssertionError):
        compare_psnr(output_a, output_b, psnr_threshold=18, eps=0.5, eps2=0.1)


def test_compute_mse() -> None:
    # Identical arrays should have MSE of 0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert compute_mse(a, b) == 0.0

    # MSE of [1,2,3] vs [2,3,4] = mean([1,1,1]) = 1.0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(compute_mse(a, b), 1.0)


def test_compute_mae() -> None:
    # Identical arrays should have MAE of 0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert compute_mae(a, b) == 0.0

    # MAE of [1,2,3] vs [2,3,4] = mean([1,1,1]) = 1.0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(compute_mae(a, b), 1.0)


def test_compute_max_abs_diff() -> None:
    # Identical arrays should have max abs diff of 0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert compute_max_abs_diff(a, b) == 0.0

    # Max abs diff of [1,2,3] vs [1,2,6] = 3.0
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 6.0])
    np.testing.assert_allclose(compute_max_abs_diff(a, b), 3.0)


def test_compute_top_k_accuracy() -> None:
    expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    k = 3
    result = compute_top_k_accuracy(expected, actual, k)
    np.testing.assert_allclose(1 / 3, result)

    actual = np.array([0.1, 0.2, 0.3, 0.5, 0.4])
    result = compute_top_k_accuracy(expected, actual, k)
    np.testing.assert_allclose(result, 1, atol=1e-3)
