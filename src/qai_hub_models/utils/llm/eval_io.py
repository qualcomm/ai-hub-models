# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Eval-result JSON I/O shared by Genie and GenieX-bench job collection."""

from __future__ import annotations

import json
import logging

from qai_hub_models.datasets.device_eval import (
    DEFAULT_DEVICE_EVAL_DATASET,
    get_device_eval_dataset,
)
from qai_hub_models.scorecard import ScorecardProfilePath

logger = logging.getLogger(__name__)


def save_eval_results_json(results: list[dict], output_path: str) -> None:
    """Save evaluation results to a JSON file, sorted by idx."""
    if not results:
        logger.warning("No results to save.")
        return

    results.sort(key=lambda r: r.get("idx", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Results saved to: %s", output_path)


def save_eval_metadata_json(
    model_id: str,
    chipset: str,
    precision: str,
    output_path: str,
    path: ScorecardProfilePath,
    dataset_name: str = DEFAULT_DEVICE_EVAL_DATASET,
) -> None:
    """Save a sidecar identifying which (model, chipset, precision, path, dataset) an eval JSON belongs to.

    The grader output (``*_eval_grade.json``) carries no model/chipset/precision,
    and the eval filename cannot be parsed unambiguously (model IDs and chipset
    slugs both contain delimiters). collect_llm_accuracy_csv reads this sidecar
    to recover the identity, and skips any grade file that lacks it. ``path`` is
    the scorecard runtime the accuracy row is written under. ``metric_name`` is
    recorded too because grade_responses runs in the grader venv, which cannot
    import the dataset registry.
    """
    metadata = {
        "model_id": model_id,
        "chipset": chipset,
        "precision": precision,
        "path": path.value,
        "dataset_name": dataset_name,
        "metric_name": get_device_eval_dataset(dataset_name).metric_metadata().name,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Eval metadata saved to: %s", output_path)
