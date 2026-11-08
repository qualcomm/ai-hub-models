# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
A hybrid LLM's perf.yaml has two writers, and each must only clear its own components.

collect_scorecard_results owns the standalone components it profiled through Workbench;
apply_llm_perf_updates owns the consolidated backbone entry measured end-to-end. Both
run in the same workflow, the LLM one second, and both start by dropping the
(precision, path, device) tuples they intended to measure. Without a component filter
the second writer silently deletes the first's results -- a failure that leaves a
correct-looking perf.yaml with the standalone component simply missing.
"""

from __future__ import annotations

import pytest

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import ScorecardDevice, cs_x_elite
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf

BACKBONE = "Qwen2.5-VL-7B-Instruct"
STANDALONE = "Vision-Encoder"
PRECISION = Precision.w4a16
PATH = ScorecardProfilePath.GENIEX_QAIRT


def _llm_details() -> QAIHMModelPerf.PerformanceDetails:
    return QAIHMModelPerf.PerformanceDetails(
        llm_metrics=[
            QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength(
                context_length=4096,
                tokens_per_second=9.74,
                prefill_tokens_per_second=580.7,
                desired_compute_unit="npu",
            )
        ]
    )


def _workbench_details(job_id: str = "jabcd1234") -> QAIHMModelPerf.PerformanceDetails:
    return QAIHMModelPerf.PerformanceDetails(
        job_id=job_id,
        job_status="Passed",
        inference_time_milliseconds=12.34,
        primary_compute_unit="NPU",
        layer_counts=QAIHMModelPerf.PerformanceDetails.LayerCounts(total=556, npu=556),
    )


def _hybrid_perf(device: ScorecardDevice = cs_x_elite) -> QAIHMModelPerf:
    perf = QAIHMModelPerf()
    perf.precisions[PRECISION] = QAIHMModelPerf.PrecisionDetails(
        components={
            BACKBONE: QAIHMModelPerf.ComponentDetails(
                performance_metrics={device: {PATH: _llm_details()}}
            ),
            STANDALONE: QAIHMModelPerf.ComponentDetails(
                performance_metrics={device: {PATH: _workbench_details()}}
            ),
        }
    )
    return perf


def _scope(
    device: ScorecardDevice = cs_x_elite,
) -> set[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]:
    return {(PRECISION, PATH, device)}


def test_llm_writer_leaves_standalone_component_alone() -> None:
    """The backbone replay must not delete the Workbench entry written moments earlier."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope(), only_components={BACKBONE})

    components = perf.precisions[PRECISION].components
    assert BACKBONE not in components, "the backbone entry should have been cleared"
    assert STANDALONE in components, "the standalone component must survive"
    assert (
        components[STANDALONE].performance_metrics[cs_x_elite][PATH].job_id
        == "jabcd1234"
    )


def test_scorecard_writer_leaves_backbone_alone() -> None:
    """Symmetrically, the Workbench writer must not delete the end-to-end numbers."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope(), only_components={STANDALONE})

    components = perf.precisions[PRECISION].components
    assert STANDALONE not in components
    assert BACKBONE in components
    assert components[BACKBONE].performance_metrics[cs_x_elite][PATH].llm_metrics


def test_unfiltered_drop_still_clears_everything() -> None:
    """Non-hybrid models pass no filter and must keep the original behaviour."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope())
    assert PRECISION not in perf.precisions


def test_filter_naming_a_missing_component_is_a_noop() -> None:
    """A stale label must not silently clear a component it does not name."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope(), only_components={"Audio Encoder"})

    components = perf.precisions[PRECISION].components
    assert set(components) == {BACKBONE, STANDALONE}


@pytest.mark.parametrize("owner", [BACKBONE, STANDALONE])
def test_out_of_scope_devices_are_untouched(owner: str) -> None:
    """Only the (precision, path, device) tuples in scope are cleared."""
    other = ScorecardDevice.get("Samsung Galaxy S25")
    perf = _hybrid_perf()
    for component in (BACKBONE, STANDALONE):
        details = perf.precisions[PRECISION].components[component]
        details.performance_metrics[other] = {PATH: _workbench_details("jother999")}

    perf.drop_entries_in_scope(_scope(cs_x_elite), only_components={owner})

    for component in (BACKBONE, STANDALONE):
        metrics = perf.precisions[PRECISION].components[component].performance_metrics
        assert other in metrics, f"{component} lost its out-of-scope device"
    assert (
        cs_x_elite
        not in perf.precisions[PRECISION].components[owner].performance_metrics
    )
