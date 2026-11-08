# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared LLM performance collection test.

Replaces the per-model test_llm_perf tests so the whole LLM fleet runs in
one venv with one pytest invocation. The per-model copies were import-heavy
(each pulled the model's source-repo pins through model.py),
which is why CollectLLMPerfTask used to spin up a fresh venv per model.
This file only needs qaihm + qai_hub + qualcomm_device_cloud_sdk +
transformers (tokenizer) — no AIMET, no source-repo deps.
"""

from __future__ import annotations

import importlib
import os

import pytest

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.models.templates.llm import test
from qai_hub_models.models.templates.llm.llm_helpers import log_perf_on_device_result
from qai_hub_models.models.templates.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT


def _llm_model_ids() -> list[str]:
    """All quantized LLMs in the repo (model_type_llm=True and a quantize.py)."""
    out: list[str] = []
    for model_id in MODEL_IDS:
        if not (QAIHM_MODELS_ROOT / model_id / "quantize.py").exists():
            continue
        try:
            info = QAIHMModelManifest.from_model(model_id)
        except Exception:
            continue
        if info.model_type_llm:
            out.append(model_id)
    return out


def _build_params() -> list[tuple[str, Precision, ScorecardDevice]]:
    params: list[tuple[str, Precision, ScorecardDevice]] = []
    for model_id in _llm_model_ids():
        combos = get_llm_perf_parametrization(model_id)
        for precision, device in combos:
            params.append((model_id, precision, device))
    return params


def _param_id(val: object) -> str:
    if isinstance(val, ScorecardDevice):
        return val.name
    return str(val)


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    return LLMPerfConfig.from_environment()


@pytest.mark.llm_perf
@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(
    ("model_id", "precision", "device"),
    _build_params(),
    ids=_param_id,
)
def test_llm_perf(
    model_id: str,
    precision: Precision,
    device: ScorecardDevice,
    llm_perf_config: LLMPerfConfig,
) -> None:
    tps, ttft, prefill_tps = test.run_llm_perf_test(
        model_id=model_id,
        device=device,
        precision=precision,
        output_dir=os.path.join(model_id, test.GENIE_BUNDLES_ROOT),
        qairt_sdk_path=llm_perf_config.qairt_sdk_path,
        skip_perf_update=llm_perf_config.skip_perf_update,
    )
    log_perf_on_device_result(
        model_name=model_id,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=ttft,
    )
