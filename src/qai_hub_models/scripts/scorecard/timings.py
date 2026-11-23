# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Per-model stage timings for load-balancing scorecard splits.

run_scorecard_single.py appends one row per (model, stage, entry) to stage-timings.csv.
This sums them per model into model-stage-timings.yaml, which
split_torch_models reads. Stages are grouped by the machine phase they run in.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from qai_hub_models.scorecard.artifacts import (
    RUNTIME_STAGE_ACCURACY,
    RUNTIME_STAGE_SUBMIT,
    ScorecardArtifact,
)
from qai_hub_models.utils.base_config import dump_yaml_dict

_PHASE_OF_STAGE = {
    "compile": RUNTIME_STAGE_SUBMIT,
    "inference": RUNTIME_STAGE_SUBMIT,
    "cpu_accuracy": RUNTIME_STAGE_SUBMIT,
    "upload_assets": RUNTIME_STAGE_ACCURACY,
    "device_accuracy": RUNTIME_STAGE_ACCURACY,
}


def build_stage_timings(csv_path: Path) -> dict[str, dict[str, float]]:
    """``{model_id: {phase: seconds}}`` summed from a stage-timings.csv."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) < 6 or row[1] not in _PHASE_OF_STAGE:
                continue
            try:
                seconds = float(row[5])
            except ValueError:
                continue
            totals[row[0]][_PHASE_OF_STAGE[row[1]]] += seconds
    return {
        model_id: {phase: round(s, 1) for phase, s in phases.items()}
        for model_id, phases in sorted(totals.items())
    }


def write_stage_timings(
    csv_path: Path | None = None, output_path: Path | None = None
) -> bool:
    """Replace the checked-in timings YAML with this run's. False if there was
    nothing to write, in which case the existing file is left alone.
    """
    csv_path = csv_path or ScorecardArtifact.STAGE_TIMINGS_CSV.path
    output_path = (
        output_path or ScorecardArtifact.MODEL_RUNTIME_ESTIMATES.intermediates_path
    )
    if not csv_path.exists():
        return False
    timings = build_stage_timings(csv_path)
    if not timings:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml_dict({"models": timings}, output_path)
    return True
