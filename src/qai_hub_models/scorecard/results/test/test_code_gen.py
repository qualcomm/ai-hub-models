# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.device import cs_8_gen_3
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.results.code_gen import (
    update_code_gen_accuracy_failure_reasons,
)
from qai_hub_models.scorecard.results.numerics_diff import NumericsDiff


def test_accuracy_failure_reasons_read_the_right_columns() -> None:
    # code_gen reads NumericsDiff rows by position; a column added or moved there
    # must not shift which values land in the manifest's disable reasons.
    diff = NumericsDiff()
    diff.device_vs_float_greater_than_enablement_threshold = [
        (
            "resnet18",
            "imagenet",
            "top1",
            cs_8_gen_3,
            Precision.w8a8,
            ScorecardProfilePath.TFLITE,
            "69.8 %",
            "69.5 %",
            "61.2 %",
            "-8.600 %",
            "5.0 %",
            True,
        )
    ]
    diff.benchmark_failures = [
        (
            "resnet18",
            "imagenet",
            "top1",
            "device",
            Precision.w8a8,
            ScorecardProfilePath.QNN_DLC,
            "69.5 %",
            "60.1 %",
            "69.8 %",
            "-9.700 %",
            "5.0 %",
            True,
        )
    ]
    manifest = QAIHMModelManifest.from_model("resnet18")

    update_code_gen_accuracy_failure_reasons("resnet18", manifest, diff)

    reasons = manifest.disabled_paths.data[Precision.w8a8]
    assert reasons[ScorecardProfilePath.TFLITE.runtime].scorecard_accuracy_failure == (
        "Torch and On-device accuracy diff (-8.600 %) above threshold (5.0 %)"
    )
    assert reasons[ScorecardProfilePath.QNN_DLC.runtime].scorecard_accuracy_failure == (
        "Device accuracy (60.1 %) deviates from benchmark (69.8 %) by -9.700 %, "
        "threshold: 5.0 %"
    )
