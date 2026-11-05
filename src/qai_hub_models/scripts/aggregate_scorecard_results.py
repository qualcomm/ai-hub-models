# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import TableauBranchNameEnvvar
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.utils.testing_async_utils import (
    get_accuracy_columns,
    get_accuracy_metadata_columns,
    get_accuracy_numerics_columns,
)

# If a model has multiple components that fail at different point
# in numerical checks, the model overall should report the error
# that is earliest in the stack.
NUMERICS_STATUS_ORDER = {
    "Off-Target Numerical Failure": 0,
    "Inference Failed": 1,
    "Low PSNR": 2,
    "On-Target Numerical Failure": 3,
    "Pass Pending Data": 4,
    "Passed": 5,
}

# Tableau expects certain columns to show up twice in the spreadsheet if they're
# present in both data sources.
DUPLICATE_ACCURACY_PERF_COLUMNS = [
    "branch",
    "chipset",
    "date",
    "precision",
    "runtime",
]


def _nullify_perf_columns(df: pd.DataFrame) -> None:
    """Set all of the columns related to perf to be empty."""
    df.inference_time = np.nan
    df.load_time = np.nan
    df.NPU = np.nan
    df.CPU = np.nan
    df.GPU = np.nan


def _convert_runtime_name(runtime: str) -> str:
    try:
        return ScorecardProfilePath(runtime).spreadsheet_name
    except ValueError:
        return runtime  # retired path from an older scorecard


def prepare_raw_data(scorecard_path: Path, accuracy_path: Path | None) -> pd.DataFrame:
    """Filter data to a single device and join accuracy data into scorecard data."""
    # Tolerate malformed rows (e.g. legacy accuracy.csv with unescaped commas)
    # so we still produce results.csv.
    scorecard_df = pd.read_csv(scorecard_path, on_bad_lines="warn")
    if accuracy_path is not None:
        accuracy_df = pd.read_csv(accuracy_path, on_bad_lines="warn")
    else:
        accuracy_df = pd.DataFrame(columns=get_accuracy_columns())

    for col_name in get_accuracy_metadata_columns():
        # Direction is kept: check_numerics needs it to know which way a metric
        # moving means "worse". get_model_status() builds its output row
        # explicitly, so it never reaches results.csv.
        if col_name == "metric_higher_is_better":
            continue
        accuracy_df = accuracy_df.drop(col_name, axis=1)
    accuracy_df["runtime"] = accuracy_df["runtime"].apply(_convert_runtime_name)

    # For component models, we want accuracy to be reported once for the entire model
    # so we create additional rows with the component_id and perf columns stripped.
    # In a later step, these rows get aggregated into a single one for the whole model
    # with the final accuracy status.
    # The original rows with perf numbers are left as is for the perf charts.
    # Since their model ids have ::, they'll have no join hits with the accuracy table.
    component_df = scorecard_df.loc[scorecard_df.model_id.str.contains("::")].copy()
    _nullify_perf_columns(component_df)
    component_df["model_id"] = component_df["model_id"].apply(
        lambda x: x.split("::")[0]
    )
    scorecard_df = pd.concat([scorecard_df, component_df])

    return scorecard_df.merge(
        accuracy_df, on=["model_id", "runtime", "precision", "chipset"], how="left"
    )


def create_check_numerics_function(
    psnr_threshold: float, accuracy_threshold: float
) -> Callable[[pd.Series], str]:
    """Return a function that returns whether a single row passes numerical checks."""
    psnr_row_headers = [f"PSNR_{i}" for i in range(10)]

    def regression(row: pd.Series, measured: float) -> float:
        """How much worse than torch `measured` is, in the metric's own direction.

        Positive means worse. For lower-is-better metrics (WER, NME, ...) a
        larger score is the regression, so the subtraction flips.
        """
        torch_accuracy = row["Torch Accuracy"]
        if torch_accuracy < 1:
            torch_accuracy *= 100
            measured *= 100
        # Compared as text so np.bool_, "False" and NaN all read correctly. NaN
        # (row predates the column, or no accuracy data) means higher-is-better,
        # which is what every metric was assumed to be.
        if str(row.get("metric_higher_is_better")).strip().lower() == "false":
            return measured - torch_accuracy
        return torch_accuracy - measured

    def check_numerics(row: pd.Series) -> str:
        if (
            not np.isnan(sim_accuracy := row["Sim Accuracy"])
            and regression(row, sim_accuracy) > accuracy_threshold
        ):
            return "Off-Target Numerical Failure"
        if not np.isnan(device_accuracy := row["Device Accuracy"]):
            if regression(row, device_accuracy) > accuracy_threshold:
                return "On-Target Numerical Failure"
            return "Passed"
        if not np.isnan(row["PSNR_0"]):
            psnr_values = row[psnr_row_headers]
            if min(psnr_values[~psnr_values.isna()]) > psnr_threshold:
                return "Pass Pending Data"
            return "Low PSNR"
        # Inference job failed, should be caught in previous bucket
        return "Inference Failed"

    return check_numerics


def create_model_status_function(
    psnr_threshold: float, accuracy_threshold: float
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Determines the final status for each model. Most models will only pass 1 row
    to the below function and determine the status from that row.

    For models with multiple components, the first check that ANY component fails,
    will be what the entire model is marked as.
    """
    check_numerics = create_check_numerics_function(psnr_threshold, accuracy_threshold)
    accuracy_numeric_columns = get_accuracy_numerics_columns()

    def get_model_status(group: pd.DataFrame) -> pd.Series:
        precision = group.iloc[0].precision
        failed_quantize_rows = group[
            ~group["quantize_status"].str.startswith("Passed", na=False)
        ]
        failed_compile_rows = group[
            ~group["compile_status"].str.startswith("Passed", na=False)
        ]
        failed_link_rows = group[
            ~group["link_status"].str.startswith("Passed", na=False)
        ]
        failed_profile_rows = group[
            ~group["profile_status"].str.startswith("Passed", na=False)
        ]
        skipped_inference_rows = group[
            group["inference_status"].str.startswith("Skipped", na=False)
        ]
        group["numerics_status"] = group.apply(check_numerics, axis=1)
        group["numerics_order"] = group.numerics_status.map(NUMERICS_STATUS_ORDER)
        group.sort_values(by="numerics_order")
        numerics_row = group.iloc[0]
        numerics_status = numerics_row.numerics_status

        if len(failed_quantize_rows) > 0 and precision != "float":
            selected_row = failed_quantize_rows.iloc[0]
            final_status = "Quantize Failed"
        elif numerics_status == "Off-Target Numerical Failure":
            selected_row = numerics_row
            final_status = numerics_status
        elif len(failed_compile_rows) > 0:
            selected_row = failed_compile_rows.iloc[0]
            final_status = "Compile Failed"
        elif len(failed_link_rows) > 0:
            selected_row = failed_link_rows.iloc[0]
            final_status = "Link Failed"
        elif len(failed_profile_rows) > 0:
            selected_row = failed_profile_rows.iloc[0]
            final_status = "Profile Failed"
        elif len(skipped_inference_rows) > 0:
            selected_row = skipped_inference_rows.iloc[0]
            final_status = "No Inference Data"
        else:
            selected_row = numerics_row
            final_status = numerics_status
        raw_data = {
            "quantize_status": selected_row.quantize_status,
            "compile_status": selected_row.compile_status,
            # "link_status": selected_row.link_status,
            "profile_status": selected_row.profile_status,
            "inference_status": selected_row.inference_status,
            "final_status": final_status,
            "quantize_url": selected_row.quantize_url,
            "compile_url": selected_row.compile_url,
            # ": selected_row.link_url,
            "profile_url": selected_row.profile_url,
            "inference_url": selected_row.inference_url,
            "tags": selected_row.tags,
            "inference_time": selected_row.inference_time,
            "load_time": selected_row.first_load_time,
            "NPU": selected_row.NPU,
            "GPU": selected_row.GPU,
            "CPU": selected_row.CPU,
            "known_issue": selected_row.known_issue,
            "branch": selected_row.branch_x,
            "date": selected_row.date_x,
            "domain": selected_row.domain,
            "use_case": selected_row.use_case,
            # TODO: Implement this column per-model
            "accuracy_threshold": "",
        }
        for col in accuracy_numeric_columns:
            raw_data[col] = selected_row[col]
        return pd.Series(raw_data)

    return get_model_status


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create table with status of each model on each runtime and print "
        "the aggregated number in each bucket for each runtime. "
        "This will only look at 8 gen 3 for now since that's all we have accuracy "
        "numbers for."
    )
    parser.add_argument(
        "--scorecard-csv",
        type=str,
        help="CSV file with scorecard data for each model",
    )
    parser.add_argument(
        "--accuracy-csv",
        type=str,
        help="CSV file with accuracy data for each model",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=1.0,
        help="Absolute percentage threshold to deem something being low accuracy.",
    )
    parser.add_argument(
        "--psnr-threshold",
        type=float,
        default=20.0,
        help="PSNR threshold to deem something being low accuracy.",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        help="The file to write the results data.",
    )
    TableauBranchNameEnvvar.add_arg(parser)
    return parser


def duplicate_scorecard_columns(source_df: pd.DataFrame) -> None:
    """
    Tableau expects certain columns to show up twice in the spreadsheet if they're
    present in both data sources.
    """
    for col in DUPLICATE_ACCURACY_PERF_COLUMNS:
        source_df[f"{col} (ACCURACY)"] = source_df[col]


def _populate_tflite_compile_failures(scorecard_df: pd.DataFrame) -> pd.DataFrame:
    """
    We skip compilation on tflite for some precisions/runtimes.

    This creates an imbalance in Tableau data where qnn/onnx have more rows than tflite,
    since all these configurations have no data on tflite.

    Here we manually populate Compile Failed on tflite for these models to balance the bar charts.
    """
    qnn_df = scorecard_df[scorecard_df.runtime == "qnn"]
    tflite_df = scorecard_df[scorecard_df.runtime == "tflite"]
    merge_df = qnn_df.merge(
        tflite_df, on=["model_id", "chipset", "precision"], how="left", indicator=True
    )

    # Remove _x suffix from column names that is an artifact of merging
    merge_df.columns = [col.removesuffix("_x") for col in merge_df.columns]

    # All chipsets, precisions that have qnn but not tflite
    subset_df = merge_df[merge_df["_merge"] == "left_only"][qnn_df.columns]
    subset_df.quantize_url = np.nan
    subset_df.compile_url = np.nan
    subset_df.link_url = np.nan
    subset_df.inference_url = np.nan
    subset_df.profile_url = np.nan
    subset_df.profile_status = "Skipped"
    subset_df.quantize_status = "Skipped"
    subset_df.inference_status = "Skipped"
    subset_df.compile_status = "Failed (Not Attempted)"
    subset_df.link_status = "Skipped"
    subset_df.runtime = "tflite"
    _nullify_perf_columns(subset_df)
    for col in get_accuracy_numerics_columns():
        subset_df[col] = np.nan
    return pd.concat([scorecard_df, subset_df], ignore_index=True)


def main() -> None:
    args = get_parser().parse_args()
    scorecard_csv = Path(args.scorecard_csv or ScorecardArtifact.EXPORT_CSV.touch())
    results_csv = Path(args.results_file or ScorecardArtifact.RESULTS_CSV.touch())
    accuracy_csv = (
        Path(args.accuracy_csv or ScorecardArtifact.ACCURACY_CSV.path)
        if ScorecardArtifact.ACCURACY_CSV.exists()
        else None
    )

    scorecard_df = prepare_raw_data(scorecard_csv, accuracy_csv)
    if scorecard_df.empty:
        # A sweep can have no Workbench jobs at all (e.g. scoped to
        # geniex_llamacpp, whose assets are static urls). Leave results.csv empty.
        os.makedirs(results_csv.parent, exist_ok=True)
        results_csv.touch()
        print(f"No scorecard rows to aggregate; left {results_csv} empty.")
        return

    scorecard_df = _populate_tflite_compile_failures(scorecard_df)
    scorecard_df["runtime"] = scorecard_df.runtime.replace(
        "precompiled_qnn_onnx", "onnx"
    )

    model_status_fn = create_model_status_function(
        args.psnr_threshold, args.accuracy_threshold
    )
    results_df = scorecard_df.groupby(
        ["model_id", "runtime", "precision", "chipset"], as_index=False
    ).apply(model_status_fn)
    if args.tableau_branch_name:
        results_df["branch"] = args.tableau_branch_name
    duplicate_scorecard_columns(results_df)

    aggregate_df = results_df.groupby(["runtime", "precision", "final_status"]).agg(
        count=("model_id", "count")
    )
    pd.set_option("display.max_rows", None)
    print(aggregate_df)

    # Tableau doesn't have the final status field, so we need to remove it for now
    # here and re-compute it there from scratch
    results_df = results_df.drop("final_status", axis=1)
    os.makedirs(results_csv.parent, exist_ok=True)
    results_df.to_csv(results_csv, index=False)
    print(f"Full results written to {results_csv}")


if __name__ == "__main__":
    main()
