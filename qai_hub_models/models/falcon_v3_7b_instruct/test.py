# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _pytest.capture import CaptureFixture
from transformers import AutoConfig

from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.common import cleanup
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.export import export_model
from qai_hub_models.models.common import Precision, TargetRuntime
from qai_hub_models.models.falcon_v3_7b_instruct import (
    MODEL_ID,
    FP_Model,
    Model,
    PositionProcessor,
    QNN_Model,
)
from qai_hub_models.models.falcon_v3_7b_instruct.demo import falcon_v3_7b_instruct_demo
from qai_hub_models.models.falcon_v3_7b_instruct.export import (
    NUM_LAYERS_PER_SPLIT,
    NUM_SPLITS,
)
from qai_hub_models.models.falcon_v3_7b_instruct.model import (
    DEFAULT_CONTEXT_LENGTH,
    HF_REPO_NAME,
    MODEL_ASSET_VERSION,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_x_elite
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.llm_helpers import (
    create_genie_config,
    log_evaluate_test_result,
    log_perf_on_device_result,
)
from qai_hub_models.utils.testing_export_eval import run_llm_compile

DEFAULT_EVAL_SEQLEN = 2048


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 4096
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    model_list = [f"falcon_v3_7b_instruct_part_{i}_of_5.bin" for i in range(1, 6)]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 4096,
                "n-vocab": 131072,
                "bos-token": 11,
                "eos-token": 11,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 256,
                        "allow-async-init": False,
                        "pos-id-dim": 128,
                        "rope-theta": 1000042,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                },
            },
        }
    }

    assert expected_config == actual_config


@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext", 7.396, 0),
        ("DEFAULT_W4A16", "mmlu", 0.703, 1000),
        ("DEFAULT_UNQUANTIZED", "wikitext", 7.453, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.68, 0),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    cleanup()
    is_unquantized = checkpoint == "DEFAULT_UNQUANTIZED"
    actual_metric, _ = evaluate(
        quantized_model_cls=Model,
        fp_model_cls=FP_Model,
        qnn_model_cls=QNN_Model,
        num_samples=num_samples,
        task=task,
        skip_fp_model_eval=not is_unquantized,
        kwargs=dict(
            checkpoint=checkpoint,
            sequence_length=DEFAULT_EVAL_SEQLEN,
            context_length=DEFAULT_CONTEXT_LENGTH,
        ),
    )
    log_evaluate_test_result(
        model_name=MODEL_ID,
        checkpoint=checkpoint,
        metric=task,
        value=actual_metric,
    )
    np.testing.assert_allclose(actual_metric, expected_metric, rtol=1e-02, atol=1e-02)


@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(checkpoint: CheckpointSpec, capsys: CaptureFixture[str]) -> None:
    cleanup()
    falcon_v3_7b_instruct_demo(
        fp_model_cls=FP_Model,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.skip(
    reason="This test is skipped till we use it to get automatic performance numbers for the LLMs."
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    cleanup()
    run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint="DEFAULT",
            sequence_length=128,
            context_length=DEFAULT_CONTEXT_LENGTH,
            _skip_quantsim_creation=True,
            model_cls=Model,
            model_name=MODEL_ID,
            model_asset_version=MODEL_ASSET_VERSION,
            num_splits=NUM_SPLITS,
            num_layers_per_split=NUM_LAYERS_PER_SPLIT,
            output_dir=test.GENIE_BUNDLES_ROOT,
            fp_model=FP_Model.from_pretrained(
                sequence_length=128, context_length=DEFAULT_CONTEXT_LENGTH
            ),
            position_processor_cls=PositionProcessor,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(test.GENIE_BUNDLES_ROOT)
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    assert (genie_bundle_path / "tokenizer.json").exists()
    assert (genie_bundle_path / "genie_config.json").exists()
    assert (genie_bundle_path / "htp_backend_ext_config.json").exists()


@pytest.mark.skip(
    reason="This test is skipped till we use it to get automatic performance numbers for the LLMs."
)
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not importlib.util.find_spec("qdc_public_api_client"),
    reason="This test can be run on GPU only. Also needs QDC package to run.",
)
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_x_elite),
    ],
)
@pytest.mark.qdc
def test_qdc(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
) -> None:
    cleanup()
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    if scorecard_path.runtime == TargetRuntime.ONNXRUNTIME_GENAI:
        pytest.skip("This test is only valid for Genie runtime.")
    if not (genie_bundle_path / "genie_config.json").exists():
        pytest.fail("The genie bundle does not exist.")
    from qai_hub_models.utils.qdc.genie_jobs import (
        submit_genie_bundle_to_qdc_device,
    )

    qdc_job_name = f"Genie {MODEL_ID} {precision}"
    tps, min_ttft = submit_genie_bundle_to_qdc_device(
        os.environ["QDC_API_TOKEN"],
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=qdc_job_name,
    )
    assert tps is not None and min_ttft is not None, "QDC execution failed."
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        ttft=min_ttft,
    )
    assert tps > 6.0
    assert min_ttft < 250000.0
