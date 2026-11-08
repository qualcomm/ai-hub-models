# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Build a scorecard-format ``accuracy.csv`` from on-device LLM grading output.

Reads ``*_eval_grade.json`` (and ``*_eval.meta.json``) and writes one
accuracy.csv row per file. The resolved manifest baseline is also written back
into each grade file as ``reference_score_pct``, for the CI summary to render.

To add an FP32 baseline for an LLM model, add a ``numerics_benchmark`` block to the
model's ``manifest.yaml``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from qai_hub_models import Precision
from qai_hub_models.datasets.device_eval import (
    DEFAULT_DEVICE_EVAL_DATASET,
    get_device_eval_dataset,
)
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.utils.testing_async_utils import write_accuracy
from qai_hub_models.utils.base_dataset import DatasetMetadata

GRADE_SUFFIX = "_grade.json"
META_SUFFIX = ".meta.json"


def _meta_path_for(grade_path: str) -> str:
    """Map ``..._eval_grade.json`` to its sibling ``..._eval.meta.json``."""
    return grade_path[: -len(GRADE_SUFFIX)] + META_SUFFIX


def _metric_name(dataset_name: str) -> str:
    return get_device_eval_dataset(dataset_name).metric_metadata().name


def _reference_grade(model_id: str, dataset_name: str) -> float | None:
    """FP baseline grade from manifest.yaml's numerics_benchmark, if comparable.

    The baseline must have been measured on the same dataset and metric as this
    run: a Grace1 number beside a Grace2 device score is two different metrics,
    not a float-vs-device delta.
    """
    try:
        sc = QAIHMModelInternalConfig.from_model(model_id)
    except Exception as e:
        print(f"  Could not load internal-config.yaml for {model_id}: {e}")
        return None
    benchmark = sc.manifest.numerics_benchmark
    if benchmark is None:
        return None
    if (
        benchmark.dataset_name == dataset_name
        and benchmark.metric_name == _metric_name(dataset_name)
        and benchmark.unit == "%"
    ):
        return benchmark.value
    print(
        f"  {model_id}: manifest baseline is {benchmark.metric_name} on "
        f"{benchmark.dataset_name}, not {_metric_name(dataset_name)} on "
        f"{dataset_name}; leaving the float reference empty."
    )
    return None


def collect(directory: str) -> int:
    """Write one accuracy.csv row per ``*_eval_grade.json`` in ``directory``.

    Returns the number of rows written.
    """
    grade_paths = sorted(glob.glob(os.path.join(directory, f"*{GRADE_SUFFIX}")))
    if not grade_paths:
        print(f"No *{GRADE_SUFFIX} files found in {directory}; nothing to collect.")
        return 0

    rows_written = 0
    for grade_path in grade_paths:
        meta_path = _meta_path_for(grade_path)
        if not os.path.exists(meta_path):
            print(
                f"Skipping {os.path.basename(grade_path)}: missing identity sidecar "
                f"{os.path.basename(meta_path)}."
            )
            continue

        meta = json.loads(Path(meta_path).read_text())
        grade = json.loads(Path(grade_path).read_text())
        model_id = meta["model_id"]
        chipset = meta["chipset"]
        precision = meta["precision"]
        dataset_name = meta.get("dataset_name", DEFAULT_DEVICE_EVAL_DATASET)
        dataset = get_device_eval_dataset(dataset_name)
        # The sidecar records the scorecard runtime; older/genie sidecars omit
        # it and fall back to GENIE.
        path_value = meta.get("path")
        path = (
            ScorecardProfilePath(path_value)
            if path_value
            else ScorecardProfilePath.GENIE
        )
        score_pct = grade.get("score_pct")
        if score_pct is None:
            print(f"Skipping {os.path.basename(grade_path)}: no score_pct present.")
            continue

        torch_accuracy = _reference_grade(model_id, dataset_name)
        # scripts/ci/grader_summary.py renders this alongside the device score,
        # and cannot resolve it itself: it runs on the system python, no venv.
        if grade.get("reference_score_pct") != torch_accuracy:
            grade["reference_score_pct"] = torch_accuracy
            Path(grade_path).write_text(json.dumps(grade, indent=2))
        write_accuracy(
            model_name=model_id,
            chipset=chipset,
            precision=Precision.parse(precision),
            path=path,
            psnr_values=[],
            torch_accuracy=torch_accuracy,
            device_accuracy=float(score_pct),
            dataset_name=dataset_name,
            dataset_metadata=DatasetMetadata(
                link=dataset.doc_url(), split_description="on-device prompt eval set"
            ),
            metric_metadata=dataset.metric_metadata(),
            num_samples=grade.get("num_items"),
        )
        rows_written += 1
        ref_str = f" (FP32 ref: {torch_accuracy:.1f}%)" if torch_accuracy else ""
        print(
            f"Wrote accuracy row: {model_id} / {chipset} / {precision} "
            f"-> {float(score_pct):.1f}%{ref_str}"
        )

    return rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing *_eval_grade.json and *_eval.meta.json files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory to write accuracy.csv into (sets QAIHM_TEST_ARTIFACTS_DIR). "
        "Defaults to the existing env value or $(cwd)/qaihm_test_artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.artifacts_dir:
        os.environ["QAIHM_TEST_ARTIFACTS_DIR"] = args.artifacts_dir

    rows = collect(args.directory)
    print(f"Collected {rows} accuracy row(s).")


if __name__ == "__main__":
    main()
