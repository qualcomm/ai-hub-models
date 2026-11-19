# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from unittest import mock

import pytest

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import cs_x_elite
from qai_hub_models.scripts import run_geniex_bench_benchmarks as mod
from qai_hub_models.utils.llm.geniex.jobs import GenieXBenchMetrics


def _metric(plugin: str) -> GenieXBenchMetrics:
    return GenieXBenchMetrics(
        cell_id=f"m-{plugin}-npu-c512",
        plugin=plugin,
        device_alias="npu",
        context_length=512,
        ttft_ms=456.3,
        prefill_tps=841.4,
        decode_tps=12.2,
        prompt_tokens=384,
        gen_tokens=128,
        geniex_version="v0.11.0",
        qairt_version="2.50",
    )


@pytest.mark.parametrize(
    ("plugin", "requested", "expected"),
    [
        ("qairt", "2.51.0.260929", "2.51.0.260929"),
        ("qairt", None, None),
        ("llama_cpp", "2.51.0.260929", None),
    ],
)
def test_record_outcome_stamps_requested_qairt_version(
    plugin: str, requested: str | None, expected: str | None
) -> None:
    metric = _metric(plugin)
    with (
        mock.patch.object(mod, "update_perf_yaml") as update,
        mock.patch.object(mod, "_save_eval_results"),
    ):
        mod._record_outcome(
            [],
            "qwen3_8b",
            cs_x_elite,
            plugin,
            Precision.w4a16,
            [metric],
            [],
            "success",
            None,
            skip_perf_update=False,
            eval_dataset="",
            qairt_version=requested,
        )
    assert metric.requested_qairt_version == (requested if plugin == "qairt" else None)
    tool_versions = update.call_args.kwargs["tool_versions"]
    assert tool_versions.geniex == "v0.11.0"
    qairt = tool_versions.qairt
    assert (qairt.full_version if qairt else None) == expected
