# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Markdown summary of a scorecard run, for the GitHub step summary and the nightly
issue: per-stage pass / fail counts and times, each failed (model, stage, path,
device) with its reason, and the Workbench job failures per stage.

Reads stage-timings.csv (run_scorecard) and results.csv from the artifacts dir.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scripts.generate_test_summary import (
    clean_message,
    write_to_github_summary,
)

SUMMARY_FILE = "scorecard-summary.md"
FAILED_PREFIX = "failed:"
BLOCKED_PREFIX = "blocked:"
FAILURE_REASON_MAX_LEN = 255
NO_FAILURE_MESSAGE = "(no job failure message)"
JOB_URL_PATTERN = re.compile(r"https?://\S*/jobs/(?P<id>\w+)/?")
FAILURE_PREFIX_PATTERN = re.compile(
    r"^(?:Inference job Job\(.*?\) failed:|(?:\w+ )?Job \w+ failed:|Job failed:)\s*"
)
JOB_ID_PATTERN = re.compile(r"Job (?P<id>j\w+) failed:|job_id=(?P<repr_id>\w+)")
JOB_STATUS_COLUMNS = {
    "Quantize": "quantize_status",
    "Compile": "compile_status",
    "Inference": "inference_status",
}


def stage_stats_table(timings: pd.DataFrame) -> str:
    """One row per stage: counts of passed / failed / blocked entries and total time.

    Blocked entries never ran because an earlier stage failed; that failure is
    listed on its own row.
    """
    rows = []
    for stage, group in timings.groupby("stage", sort=False):
        failed = int(group["status"].str.startswith(FAILED_PREFIX).sum())
        blocked = int(group["status"].str.startswith(BLOCKED_PREFIX).sum())
        rows.append(
            {
                "Status": "❌" if failed else "✅",
                "Stage": stage,
                "Total": len(group),
                "Passed": len(group) - failed - blocked,
                "Failed": failed,
                "Blocked": blocked,
                "Time (min)": round(group["seconds"].sum() / 60.0, 2),
            }
        )
    return tabulate(
        pd.DataFrame(rows), headers="keys", tablefmt="pipe", showindex=False
    )


def split_failure(status: str) -> tuple[str, str]:
    """
    Split a failed stage status into (markdown job link, Hub failure message).

    Reads "Job <id> failed: <message> (<url>)". Older runs also wrote
    "Job failed: <url>" (no message recorded) and
    "Inference job Job(job_id=<id>, ...) failed: <message>" (no URL, so no link).
    The link is empty when the failure was not a job.
    """
    text = status.removeprefix(FAILED_PREFIX).strip()
    url_match = JOB_URL_PATTERN.search(text)
    id_match = JOB_ID_PATTERN.search(text)
    job_id = (
        (id_match.group("id") or id_match.group("repr_id"))
        if id_match
        else (url_match.group("id") if url_match else None)
    )
    if job_id is None:
        return "", text
    url = url_match.group(0) if url_match else ""
    link = f"[{job_id}]({url})" if url else job_id

    message = FAILURE_PREFIX_PATTERN.sub("", text)
    if url:
        message = re.sub(rf"\s*\({re.escape(url)}\)\s*$", "", message)
        message = "" if message.strip() == url else message
    return link, message.strip() or NO_FAILURE_MESSAGE


def failures_table(timings: pd.DataFrame) -> str:
    """
    One row per failed stage entry, with its job link and the Hub's reason.
    Entries that differ only by device collapse into one row showing the device
    count and the first entry's job.
    """
    failed = timings[timings["status"].str.startswith(FAILED_PREFIX)]
    if failed.empty:
        return "No stage failures."
    split = failed["status"].map(split_failure)
    table = pd.DataFrame(
        {
            "Status": "❌",
            "Model": failed["model"],
            "Stage": failed["stage"],
            "Precision": failed["precision"],
            "Path": failed["path"],
            "Device": failed["device"],
            "Job": split.map(lambda s: s[0]),
            "Failure Reason": split.map(
                lambda s: clean_message(s[1], FAILURE_REASON_MAX_LEN)
            ),
        }
    )
    group_cols = ["Model", "Stage", "Precision", "Path", "Failure Reason"]
    grouped = table.groupby(group_cols, sort=False, dropna=False)
    num_devices = grouped["Device"].transform("size")
    table = table[~table.duplicated(group_cols)].copy()
    table["Device"] = [
        f"{n} devices" if n > 1 else device
        for n, device in zip(num_devices[table.index], table["Device"], strict=True)
    ]
    return tabulate(table, headers="keys", tablefmt="pipe", showindex=False)


def skipped_models_table(timings: pd.DataFrame) -> str:
    """Models run_scorecard did not run, with the reason."""
    skipped = timings[timings["stage"] == "skipped"]
    if skipped.empty:
        return ""
    table = pd.DataFrame({"Model": skipped["model"], "Reason": skipped["status"]})
    return tabulate(table, headers="keys", tablefmt="pipe", showindex=False)


def job_failures_table(results: pd.DataFrame) -> str:
    """Workbench job failures per stage, from results.csv."""
    rows = []
    for stage, column in JOB_STATUS_COLUMNS.items():
        if column not in results.columns:
            continue
        statuses = results[column].fillna("").astype(str)
        failed = int(statuses.str.startswith("Failed").sum())
        total = int((statuses != "").sum() - (statuses == "skipped").sum())
        rows.append(
            {
                "Status": "❌" if failed else "✅",
                "Job": stage,
                "Total": total,
                "Failed": failed,
            }
        )
    return tabulate(
        pd.DataFrame(rows), headers="keys", tablefmt="pipe", showindex=False
    )


def build_summary(
    timings_path: Path, results_path: Path, title: str = "Scorecard"
) -> str:
    sections = [f"## {title} Summary\n"]

    if timings_path.exists() and timings_path.stat().st_size > 0:
        timings = pd.read_csv(timings_path, keep_default_na=False)
        timings["seconds"] = pd.to_numeric(timings["seconds"], errors="coerce").fillna(
            0.0
        )
        timings["status"] = timings["status"].astype(str)
        run = timings[timings["stage"] != "skipped"]
        if run.empty:
            sections.append("❌ No stages ran.\n")
        else:
            sections += ["### Stage Results\n", stage_stats_table(run) + "\n"]
            sections += ["### Stage Failures\n", failures_table(run) + "\n"]
        if skipped := skipped_models_table(timings):
            sections += ["### Models Not Run\n", skipped + "\n"]
    else:
        sections.append(f"❌ {timings_path.name} not found: no stage results.\n")

    if results_path.exists() and results_path.stat().st_size > 0:
        results = pd.read_csv(results_path, keep_default_na=False)
        sections += ["### Workbench Job Results\n", job_failures_table(results) + "\n"]
    else:
        sections.append(f"❌ {results_path.name} not found: no job results.\n")

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="Scorecard")
    parser.add_argument(
        "--timings-csv", type=Path, default=ScorecardArtifact.STAGE_TIMINGS_CSV.path
    )
    parser.add_argument(
        "--results-csv", type=Path, default=ScorecardArtifact.RESULTS_CSV.path
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Where to write the summary (default: {SUMMARY_FILE} next to the timings).",
    )
    args = parser.parse_args()

    summary = build_summary(args.timings_csv, args.results_csv, args.title)
    output = args.output or args.timings_csv.parent / SUMMARY_FILE
    os.makedirs(output.parent, exist_ok=True)
    output.write_text(summary)
    write_to_github_summary(summary)


if __name__ == "__main__":
    main()
