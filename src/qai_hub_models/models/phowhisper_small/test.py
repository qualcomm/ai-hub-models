# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.hf_whisper.test_utils import (
    run_test_transcribe,
    run_test_wrapper_numerics,
)
from qai_hub_models.models.phowhisper_small.demo import main as demo_main
from qai_hub_models.models.phowhisper_small.model import PhoWhisperSmall


def test_numerics() -> None:
    run_test_wrapper_numerics(PhoWhisperSmall)


def test_transcribe() -> None:
    run_test_transcribe(PhoWhisperSmall)


def test_demo() -> None:
    demo_main(is_test=True)
