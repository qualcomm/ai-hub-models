# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import functools
import math
import os
import traceback
from copy import deepcopy

import pandas as pd
from junitparser.junitparser import Error, Failure, JUnitXml, TestCase, TestSuite

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import DEFAULT_SCORECARD_DEVICE

DEFAULT_CHIPSET = DEFAULT_SCORECARD_DEVICE.chipset
EFFICIENTFORMER_CHIPSET = "qualcomm-sa8295p"

EXPECTED_MODEL_SETS = {
    "yolov8_det": {
        "float": ["onnx", "qnn", "tflite"],
        "w8a8": ["onnx", "qnn", "tflite"],
        "w8a16": ["onnx", "qnn"],
        "w8a8_mixed_int16": ["onnx", "qnn"],
    },
    "mask2former": {
        "float": ["onnx", "qnn"],
    },
    "mediapipe_face": {
        "float": ["onnx", "qnn", "tflite"],
        "w8a8": ["onnx", "qnn", "tflite"],
    },
    "cdcn_torchscript": {"float": ["onnx", "qnn", "tflite"]},
    "efficientformer_onnx": {"float": ["qnn", "tflite"]},
}

# Scored against their eval dataset (Device Accuracy); every other model gets PSNR.
EVAL_DATASET_MODELS = {"yolov8_det", "mask2former"}

SELECTED_DEVICES = {
    ("mask2former", DEFAULT_CHIPSET),
    ("mediapipe_face", DEFAULT_CHIPSET),
    ("mediapipe_face::face_landmark_detector", DEFAULT_CHIPSET),
    ("mediapipe_face::face_detector", DEFAULT_CHIPSET),
    ("yolov8_det", DEFAULT_CHIPSET),
    ("cdcn_torchscript", DEFAULT_CHIPSET),
    ("efficientformer_onnx", EFFICIENTFORMER_CHIPSET),
}

_FACE_SUBGRAPHS = (
    "mediapipe_face::face_landmark_detector",
    "mediapipe_face::face_detector",
)

_ALL_PASS = {"compile": "Passed", "profile": "Passed", "inference": "Passed"}

_RELIABLE_RESULTS_COMBOS: list[tuple[str, str, str, str]] = [
    ("mask2former", "float", "onnx", DEFAULT_CHIPSET),
    ("mask2former", "float", "qnn", DEFAULT_CHIPSET),
    ("mediapipe_face", "float", "onnx", DEFAULT_CHIPSET),
    ("mediapipe_face", "float", "qnn", DEFAULT_CHIPSET),
    ("mediapipe_face", "float", "tflite", DEFAULT_CHIPSET),
    ("mediapipe_face", "w8a8", "onnx", DEFAULT_CHIPSET),
    ("mediapipe_face", "w8a8", "qnn", DEFAULT_CHIPSET),
    ("mediapipe_face", "w8a8", "tflite", DEFAULT_CHIPSET),
    ("yolov8_det", "float", "onnx", DEFAULT_CHIPSET),
    ("yolov8_det", "float", "qnn", DEFAULT_CHIPSET),
    ("yolov8_det", "float", "tflite", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a8", "onnx", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a8", "qnn", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a8", "tflite", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a16", "onnx", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a16", "qnn", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a8_mixed_int16", "onnx", DEFAULT_CHIPSET),
    ("yolov8_det", "w8a8_mixed_int16", "qnn", DEFAULT_CHIPSET),
    ("cdcn_torchscript", "float", "onnx", DEFAULT_CHIPSET),
    ("cdcn_torchscript", "float", "qnn", DEFAULT_CHIPSET),
    ("cdcn_torchscript", "float", "tflite", DEFAULT_CHIPSET),
]

EXPECTED_RESULTS_STAGE_STATUS: dict[tuple[str, str, str, str], dict[str, str]] = {
    combo: dict(_ALL_PASS) for combo in _RELIABLE_RESULTS_COMBOS
}
for sub in _FACE_SUBGRAPHS:
    for precision in ("float", "w8a8"):
        for runtime in ("onnx", "qnn", "tflite"):
            EXPECTED_RESULTS_STAGE_STATUS[
                (sub, precision, runtime, DEFAULT_CHIPSET)
            ] = dict(_ALL_PASS)
for runtime in ("qnn", "tflite"):
    EXPECTED_RESULTS_STAGE_STATUS[
        ("efficientformer_onnx", "float", runtime, EFFICIENTFORMER_CHIPSET)
    ] = {"compile": "Passed", "inference": "Passed"}

EXPECTED_ACCURACY_KEYS: set[tuple[str, str, str]] = {
    ("efficientformer_onnx", "float", "qnn_dlc"),
    ("efficientformer_onnx", "float", "tflite"),
    ("mask2former", "float", "qnn_context_binary"),
    ("mask2former", "float", "precompiled_qnn_onnx"),
    ("mediapipe_face", "float", "onnx"),
    ("mediapipe_face", "float", "qnn_dlc"),
    ("mediapipe_face", "float", "tflite"),
    ("mediapipe_face", "w8a8", "onnx"),
    ("mediapipe_face", "w8a8", "qnn_dlc"),
    ("mediapipe_face", "w8a8", "tflite"),
    ("yolov8_det", "float", "onnx"),
    ("yolov8_det", "float", "qnn_dlc"),
    ("yolov8_det", "float", "tflite"),
    ("yolov8_det", "w8a8", "onnx"),
    ("yolov8_det", "w8a8", "qnn_dlc"),
    ("yolov8_det", "w8a8", "tflite"),
    ("yolov8_det", "w8a16", "onnx"),
    ("yolov8_det", "w8a16", "qnn_dlc"),
    ("yolov8_det", "w8a8_mixed_int16", "onnx"),
    ("yolov8_det", "w8a8_mixed_int16", "qnn_dlc"),
    ("cdcn_torchscript", "float", "onnx"),
    ("cdcn_torchscript", "float", "qnn_dlc"),
    ("cdcn_torchscript", "float", "tflite"),
}


@functools.lru_cache(maxsize=1)
def num_configurations() -> int:
    """Number of (runtime, precision, model) combinations"""
    return sum(
        [
            len(runtime_list)
            for _, model_dict in EXPECTED_MODEL_SETS.items()
            for _, runtime_list in model_dict.items()
        ]
    )


def _status_kind(status: str) -> str:
    if not status or status == "skipped":
        return "skipped"
    if status.startswith("Passed"):
        return "Passed"
    if status.startswith("Failed"):
        return "Failed"
    return status


def _check_expected_stage_statuses(results_df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    by_key = {
        (row.model_id, row.precision, row.runtime, row.chipset): row
        for _, row in results_df.iterrows()
    }
    for key, expected in EXPECTED_RESULTS_STAGE_STATUS.items():
        row = by_key.get(key)
        if row is None:
            errors.append(
                f"Missing expected results row for {key} (cannot check stage statuses)."
            )
            continue
        for stage, expected_status in expected.items():
            actual = _status_kind(getattr(row, f"{stage}_status", ""))
            if actual != expected_status:
                full = getattr(row, f"{stage}_status", "")
                errors.append(
                    f"Stage regression: {key} expected {stage}={expected_status} "
                    f"but got {actual!r} (full status: {full!r})."
                )
    return errors


def validate_results_df(results_df: pd.DataFrame) -> list[str]:
    """
    Checks the aggregated results csv and verifies that it has the expected outputs.
    Returns a list of error strings, if any.
    """
    errors = _check_expected_stage_statuses(results_df)

    results_df = results_df[
        results_df[["model_id", "chipset"]].apply(tuple, axis=1).isin(SELECTED_DEVICES)
    ]
    unfound_model_sets = deepcopy(EXPECTED_MODEL_SETS)
    unfound_model_sets["mediapipe_face::face_landmark_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )
    unfound_model_sets["mediapipe_face::face_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )

    unfound_model_sets["yolov8_det"]["w8a16"].append("tflite")
    unfound_model_sets["yolov8_det"]["w8a8_mixed_int16"].append("tflite")
    unfound_model_sets["mask2former"]["float"].append("tflite")
    for _, row in results_df.iterrows():
        runtime_list = unfound_model_sets.get(row.model_id, {}).get(row.precision, [])
        if row.runtime in runtime_list:
            unfound_model_sets[row.model_id][row.precision].remove(row.runtime)
            if len(unfound_model_sets[row.model_id][row.precision]) == 0:
                unfound_model_sets[row.model_id].pop(row.precision)
            if len(unfound_model_sets[row.model_id]) == 0:
                unfound_model_sets.pop(row.model_id)
        else:
            errors.append(
                f"Unexpected (or duplicate) model configuration in aggregated csv ({row.model_id}, {row.precision}, {row.runtime})."
            )
    if len(unfound_model_sets) > 0:
        errors.append(
            f"Missing some rows in aggregated results csv: {unfound_model_sets}"
        )
    return errors


def validate_scorecard_df(scorecard_df: pd.DataFrame) -> list[str]:
    """
    Checks the performance results csv and verifies that it has the expected outputs.
    Returns a list of error strings, if any.
    """
    scorecard_df = scorecard_df[
        scorecard_df[["model_id", "chipset"]]
        .apply(tuple, axis=1)
        .isin(SELECTED_DEVICES)
    ]
    unfound_model_sets = deepcopy(EXPECTED_MODEL_SETS)
    unfound_model_sets["mediapipe_face::face_landmark_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )
    unfound_model_sets["mediapipe_face::face_detector"] = deepcopy(
        unfound_model_sets["mediapipe_face"]
    )
    unfound_model_sets.pop("mediapipe_face")

    errors = []
    for _, row in scorecard_df.iterrows():
        runtime_list = unfound_model_sets.get(row.model_id, {}).get(row.precision, [])
        if row.runtime in runtime_list:
            unfound_model_sets[row.model_id][row.precision].remove(row.runtime)
            if len(unfound_model_sets[row.model_id][row.precision]) == 0:
                unfound_model_sets[row.model_id].pop(row.precision)
            if len(unfound_model_sets[row.model_id]) == 0:
                unfound_model_sets.pop(row.model_id)
        else:
            errors.append(
                f"Unexpected (or duplicate) model configuration in scorecard csv ({row.model_id}, {row.precision}, {row.runtime})."
            )
    if len(unfound_model_sets) > 0:
        errors.append(
            f"Missing some rows in scorecard results csv: {unfound_model_sets}"
        )
    return errors


def validate_accuracy_df(accuracy_df: pd.DataFrame) -> list[str]:
    """
    Checks the accuracy results csv and verifies that it has the expected outputs.
    Returns a list of error strings, if any.
    """
    errors: list[str] = []
    expected_models = set(EXPECTED_MODEL_SETS.keys())
    accuracy_models = set(accuracy_df.model_id.unique())
    if accuracy_models != expected_models:
        errors.append(
            f"Mismatch between accuracy csv models ({accuracy_models}) and expected models ({expected_models})."
        )

    by_key: dict[tuple[str, str, str], float | None] = {}
    for _, row in accuracy_df.iterrows():
        key = (row.model_id, row.precision, row.runtime)
        column = "Device Accuracy" if row.model_id in EVAL_DATASET_MODELS else "PSNR_0"
        value = row.get(column)
        try:
            by_key[key] = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            by_key[key] = None

    for key in EXPECTED_ACCURACY_KEYS:
        if key not in by_key:
            errors.append(f"Missing expected accuracy row for {key}.")
            continue
        actual = by_key[key]
        if actual is None or math.isnan(actual):
            column = "Device Accuracy" if key[0] in EVAL_DATASET_MODELS else "PSNR_0"
            errors.append(f"Accuracy row {key} has no valid {column} value.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate scorecard integration test outputs"
    )
    parser.add_argument("--junit-xml", help="Path to write JUnit XML results")
    args = parser.parse_args()

    def write_junit_xml(
        result: Failure | Error | None,
    ) -> None:
        if not args.junit_xml:
            return
        suite = TestSuite(name="Scorecard Integration Test")
        tc = TestCase(
            name="validate_scorecard_outputs", classname="scorecard_integration_test"
        )
        if result is not None:
            tc.result = [result]
        suite.add_testcase(tc)

        xml = JUnitXml()
        xml.add_testsuite(suite)
        os.makedirs(os.path.dirname(args.junit_xml) or ".", exist_ok=True)
        xml.write(args.junit_xml)

    try:
        scorecard_df = pd.read_csv(ScorecardArtifact.EXPORT_CSV.path)
        results_df = pd.read_csv(ScorecardArtifact.RESULTS_CSV.path)

        errors = []
        errors.extend(validate_results_df(results_df))
        errors.extend(validate_scorecard_df(scorecard_df))

        if ScorecardArtifact.ACCURACY_CSV.exists():
            accuracy_df = pd.read_csv(ScorecardArtifact.ACCURACY_CSV.path)
            errors.extend(validate_accuracy_df(accuracy_df))
        else:
            errors.append("accuracy.csv not found — accuracy tests may have failed.")

        if errors:
            raise ValueError(  # noqa: TRY301
                "The following errors occurred during validation:\n\n"
                + "\n\n".join(errors)
            )
    except Exception as e:
        err = Error(message=str(e), type_=type(e).__name__)
        err.text = traceback.format_exc()
        write_junit_xml(err)
        raise

    write_junit_xml(None)


if __name__ == "__main__":
    main()
