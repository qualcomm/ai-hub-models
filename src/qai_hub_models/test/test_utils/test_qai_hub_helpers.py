# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import qai_hub as hub

from qai_hub_models import Precision
from qai_hub_models.utils.job_options import extract_job_options
from qai_hub_models.utils.qai_hub_helpers import raise_if_fp_is_unsupported


def assert_options_eq(options: str, options_dict: dict[str, str | bool]) -> None:
    assert extract_job_options(MagicMock(spec=hub.Job, options=options)) == options_dict


def test_extract_job_options() -> None:
    assert_options_eq("", {})
    assert_options_eq("--boolean_flag", {"boolean_flag": True})
    assert_options_eq(
        "--blah text --boolean_flag", {"blah": "text", "boolean_flag": True}
    )
    assert_options_eq(
        "--boolean_flag --dict-input='blah=true;x=y'",
        {"dict-input": "blah=true;x=y", "boolean_flag": True},
    )
    assert_options_eq("--dict-input 'blah=true;x=y'", {"dict-input": "blah=true;x=y"})
    assert_options_eq('--dict-input "blah=true;x=y"', {"dict-input": "blah=true;x=y"})
    assert_options_eq('--dict-input="blah=true;x=y"', {"dict-input": "blah=true;x=y"})
    assert_options_eq("--dict-input=blah=true;x=y", {"dict-input": "blah=true;x=y"})


# ---------------------------------------------------------------------------
# raise_if_fp_is_unsupported
# ---------------------------------------------------------------------------


def _fp16_incapable() -> hub.Device:
    return hub.Device(
        name="Google Pixel 3",
        attributes=["os:android", "chipset:qualcomm-snapdragon-845"],
    )


def test_raise_if_fp_is_unsupported_quantized_is_noop() -> None:
    """Quantized precisions never trigger the FP16 check regardless of device."""
    raise_if_fp_is_unsupported(_fp16_incapable(), Precision.w8a8)


def test_raise_if_fp_is_unsupported_rejects_fp_on_no_fp16_device() -> None:
    """FP precision on a fully-resolved fp16-incapable device raises."""
    with pytest.raises(ValueError, match="FP16 support"):
        raise_if_fp_is_unsupported(_fp16_incapable(), Precision.float)
