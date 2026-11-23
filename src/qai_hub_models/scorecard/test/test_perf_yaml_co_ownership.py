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
from qai_hub_models_cli.proto import model_perf_pb2

from qai_hub_models import Precision
from qai_hub_models.scorecard.device import ScorecardDevice, cs_x_elite
from qai_hub_models.scorecard.path_profile import ScorecardInferencePath
from qai_hub_models.scorecard.perf_yaml import (
    QAIHMModelPerf,
)

BACKBONE = "Qwen2.5-VL-7B-Instruct"
STANDALONE = "Vision-Encoder"
PRECISION = Precision.w4a16
PATH = ScorecardInferencePath.GENIEX_QAIRT
MODEL_ID = "qwen2_5_vl_7b_instruct"


def _llm_details() -> model_perf_pb2.ModelPerf.PerformanceDetails:
    return model_perf_pb2.ModelPerf.PerformanceDetails(
        llm_metrics=[
            model_perf_pb2.ModelPerf.LLMPerfMetrics(
                context_length=4096,
                tokens_per_second=9.74,
                prefill_tokens_per_second=580.7,
                desired_compute_unit="npu",
            )
        ]
    )


def _workbench_details(
    job_id: str = "jabcd1234",
) -> model_perf_pb2.ModelPerf.PerformanceDetails:
    return model_perf_pb2.ModelPerf.PerformanceDetails(
        profile_job=model_perf_pb2.ModelPerf.ProfileJob(id=job_id, status="Passed"),
        metrics=model_perf_pb2.ModelPerf.PerfMetrics(
            inference_time_milliseconds=12.34,
            primary_compute_unit="NPU",
            layer_counts=model_perf_pb2.ModelPerf.PerfMetrics.LayerCounts(
                total=556, npu=556
            ),
        ),
    )


def _hybrid_perf(device: ScorecardDevice = cs_x_elite) -> QAIHMModelPerf:
    perf = QAIHMModelPerf(MODEL_ID)
    perf.update_entry(
        QAIHMModelPerf.Key(PRECISION, BACKBONE, device, PATH).stamp(_llm_details())
    )
    perf.update_entry(
        QAIHMModelPerf.Key(PRECISION, STANDALONE, device, PATH).stamp(
            _workbench_details()
        )
    )
    return perf


def _scope(
    device: ScorecardDevice = cs_x_elite,
) -> set[tuple[Precision, ScorecardInferencePath, ScorecardDevice]]:
    return {(PRECISION, PATH, device)}


def test_llm_writer_leaves_standalone_component_alone() -> None:
    """The backbone replay must not delete the Workbench entry written moments earlier."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope(), only_components={BACKBONE})

    assert not perf.has_component(PRECISION, BACKBONE), (
        "the backbone entry should have been cleared"
    )
    assert perf.has_component(PRECISION, STANDALONE), (
        "the standalone component must survive"
    )
    entry = perf.entries.get(
        QAIHMModelPerf.Key(PRECISION, STANDALONE, cs_x_elite, PATH)
    )
    assert entry is not None
    assert entry.profile_job.id == "jabcd1234"


def test_scorecard_writer_leaves_backbone_alone() -> None:
    """Symmetrically, the Workbench writer must not delete the end-to-end numbers."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope(), only_components={STANDALONE})

    assert not perf.has_component(PRECISION, STANDALONE)
    entry = perf.entries.get(QAIHMModelPerf.Key(PRECISION, BACKBONE, cs_x_elite, PATH))
    assert entry is not None
    assert entry.llm_metrics


def test_unfiltered_drop_still_clears_everything() -> None:
    """Non-hybrid models pass no filter and must keep the original behaviour."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope())
    assert not perf.entries


def test_filter_naming_a_missing_component_is_a_noop() -> None:
    """A stale label must not silently clear a component it does not name."""
    perf = _hybrid_perf()
    perf.drop_entries_in_scope(_scope(), only_components={"Audio Encoder"})

    assert {params.component for params in perf.entries} == {BACKBONE, STANDALONE}


@pytest.mark.parametrize("owner", [BACKBONE, STANDALONE])
def test_out_of_scope_devices_are_untouched(owner: str) -> None:
    """Only the (precision, path, device) tuples in scope are cleared."""
    other = ScorecardDevice.get("Samsung Galaxy S25")
    perf = _hybrid_perf()
    for component in (BACKBONE, STANDALONE):
        perf.update_entry(
            QAIHMModelPerf.Key(PRECISION, component, other, PATH).stamp(
                _workbench_details("jother999")
            )
        )

    perf.drop_entries_in_scope(_scope(cs_x_elite), only_components={owner})

    for component in (BACKBONE, STANDALONE):
        assert QAIHMModelPerf.Key(PRECISION, component, other, PATH) in perf.entries, (
            f"{component} lost its out-of-scope device"
        )
    assert QAIHMModelPerf.Key(PRECISION, owner, cs_x_elite, PATH) not in perf.entries
