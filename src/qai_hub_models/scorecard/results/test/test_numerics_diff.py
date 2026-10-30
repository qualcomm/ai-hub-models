# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import cs_8_gen_3
from qai_hub_models.scorecard.numerics_yaml import QAIHMModelNumerics
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.numerics_diff import NumericsDiff


def test_dump_newly_disabled_json(tmp_path: Path) -> None:
    """dump_newly_disabled_json filters to newly-disabled rows and tags by kind."""
    diff = NumericsDiff()
    diff.device_vs_float_greater_than_enablement_threshold = [
        (
            "fresh_model",
            "coco",
            "mAP",
            cs_8_gen_3,
            Precision.float,
            ScorecardProfilePath.QNN_DLC,
            "0.75 mAP",
            "0.74 mAP",
            "0.60 mAP",
            "-0.150 mAP",
            "0.05 mAP",
            True,
        ),
        (
            "stale_model",
            "coco",
            "mAP",
            cs_8_gen_3,
            Precision.float,
            ScorecardProfilePath.QNN_DLC,
            "0.75 mAP",
            "",
            "0.60 mAP",
            "-0.150 mAP",
            "0.05 mAP",
            False,
        ),
    ]
    diff.benchmark_failures = [
        (
            "fresh_bench",
            "imagenet",
            "top1",
            "device",
            Precision.w8a8,
            ScorecardProfilePath.TFLITE,
            "0.31 top1",
            "0.30 top1",
            "0.70 top1",
            "-0.400 top1",
            "0.05 top1",
            True,
        ),
        (
            "stale_torch",
            "imagenet",
            "top1",
            "torch",
            None,
            None,
            "",
            "0.30 top1",
            "0.70 top1",
            "-0.400 top1",
            "0.05 top1",
            False,
        ),
    ]

    json_path = str(tmp_path / "newly-disabled.json")
    diff.dump_newly_disabled_json(json_path)

    with open(json_path) as f:
        data = json.load(f)

    assert len(data) == 2, "stale (Newly Disabled=False) rows should be filtered out"

    disabled = [r for r in data if r["kind"] == "disabled_configuration"]
    benchmarks = [r for r in data if r["kind"] == "benchmark_failure"]
    assert len(disabled) == 1
    assert len(benchmarks) == 1

    assert disabled[0]["Model ID"] == "fresh_model"
    assert disabled[0]["FP Accuracy"] == "0.75 mAP"
    assert disabled[0]["Newly Disabled"] == "True"
    assert disabled[0]["Sim Accuracy"] == "0.74 mAP"
    assert disabled[0]["Device Accuracy"] == "0.60 mAP"

    assert benchmarks[0]["Model ID"] == "fresh_bench"
    assert benchmarks[0]["Accuracy Type"] == "device"
    assert benchmarks[0]["Device Value"] == "0.30 top1"
    assert benchmarks[0]["Sim Value"] == "0.31 top1"


def test_sim_cell_is_per_precision_and_blank_without_sim() -> None:
    """Sim is looked up by precision, so a quantized row gets its value and float stays blank."""
    diff = NumericsDiff()
    metric = QAIHMModelNumerics.MetricDetails(
        dataset_name="imagenet",
        dataset_link="",
        dataset_split_description="",
        metric_name="top1",
        metric_description="",
        metric_unit="%",
        metric_range=QAIHMModelNumerics.Range(),
        num_partial_samples=100,
        partial_torch_metric=77.6,
    )
    diff.sim_accuracies[("resnet", "imagenet", "top1", Precision.w8a8)] = 77.5

    assert diff._sim_cell("resnet", metric, Precision.w8a8) == "77.5 %"
    # Float has no sim run; a blank cell keeps readers from mistaking it for a measurement.
    assert diff._sim_cell("resnet", metric, Precision.float) == ""
