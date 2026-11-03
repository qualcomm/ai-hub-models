#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path

from prettytable import PrettyTable

sys.path.insert(0, str(Path(__file__).parent))
from qai_hub import JobType
from utils import (
    DISPLAY_SEPARATOR,
    JOB_STATUS_SUCCESS,
    create_results_table,
    filter_known_failures,
    find_passing_known_failures,
    get_date_str,
    load_client,
    load_yaml_safe,
    log_and_print,
    rerun_regressions,
    save_yaml_results,
    setup_script_logging,
    validate_tag,
)

logger = logging.getLogger(__name__)


def _compile_row(model_name: str, info: dict) -> list:
    return [model_name, info["prod_job_url"], info["dev_job_url"]]


def _infra_failure_row(model_name: str, info: dict) -> list:
    return [
        model_name,
        info["prod_job_url"],
        info["original_job_url"],
        info["dev_job_url"],
    ]


def _compile_status_row(model_name: str, info: dict) -> list:
    return [
        model_name,
        info.get("prod_job_url", "N/A"),
        info.get("dev_job_url", "N/A"),
        info.get("dev_status", "SUBMISSION_FAILED" if info.get("error") else "N/A"),
    ]


def get_tfl_converter_jobs(*result_sets: dict) -> dict:
    tfl_converter_jobs = {}
    for result_set in result_sets:
        for model_name, info in result_set.items():
            if "_TFL_CONVERTER" in model_name:
                tfl_converter_jobs.setdefault(model_name, info)
    return tfl_converter_jobs


def print_tfl_converter_results(tfl_converter_jobs: dict) -> str:
    passed_count = sum(
        1
        for info in tfl_converter_jobs.values()
        if info.get("dev_status") == JOB_STATUS_SUCCESS
    )
    total_count = len(tfl_converter_jobs)
    summary_line = f"TFL Converter = {passed_count} / {total_count}"

    log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
    log_and_print(summary_line, logger)
    log_and_print(DISPLAY_SEPARATOR, logger)

    failing_jobs = {
        model_name: info
        for model_name, info in tfl_converter_jobs.items()
        if info.get("dev_status") != JOB_STATUS_SUCCESS
    }
    if failing_jobs:
        table = create_results_table(
            failing_jobs,
            ["Model", "Prod Job URL", "Dev Job URL", "Dev Status"],
            _compile_status_row,
        )
        log_and_print("\nTFL CONVERTER FAILING JOBS", logger)
        log_and_print(DISPLAY_SEPARATOR, logger)
        for line in str(table).split("\n"):
            log_and_print(line, logger)

    return summary_line


def print_summary(
    regressions: dict,
    progressions: dict,
    failures: dict,
    passed: dict,
    known: dict,
    passing_known: dict,
    infra_failures: dict,
    job_yaml_tag: str,
) -> PrettyTable:
    summary_table = PrettyTable()
    summary_table.field_names = ["Metric", "Count"]
    summary_table.align["Metric"] = "l"
    summary_table.align["Count"] = "r"
    summary_table.add_row(["Passed (Both Prod & Dev)", len(passed)])
    summary_table.add_row(["Regressions", len(regressions)])
    summary_table.add_row(["Infrastructure Failures", len(infra_failures)])
    summary_table.add_row(["Progressions", len(progressions)])
    summary_table.add_row(["Failures (Both Prod & Dev)", len(failures)])
    summary_table.add_row(["Known Failures (Excluded)", len(known)])
    summary_table.add_row(["Known Issues Passing Now", len(passing_known)])

    log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
    log_and_print(f"Summary for {job_yaml_tag}", logger)
    log_and_print(DISPLAY_SEPARATOR, logger)
    for line in str(summary_table).split("\n"):
        log_and_print(line, logger)

    # Print regressions and known failures to console and log
    field_names = ["Model", "Prod Job URL", "Dev Job URL"]
    console_sections = [
        (
            regressions,
            "REGRESSIONS: Prod SUCCESS -> Dev FAILED (failed again on re-run)",
            field_names,
            _compile_row,
        ),
        (
            infra_failures,
            "INFRASTRUCTURE FAILURES: Dev FAILED then PASSED on re-run",
            ["Model", "Prod Job URL", "Failed Dev Job URL", "Re-run Dev Job URL"],
            _infra_failure_row,
        ),
        (
            known,
            "KNOWN FAILURES: Prod SUCCESS -> Dev FAILED (excluded from regressions)",
            field_names,
            _compile_row,
        ),
        (
            passing_known,
            "KNOWN ISSUES PASSING NOW: remove from known_failures.yaml",
            field_names,
            _compile_row,
        ),
    ]
    for data, title, fields, row_fn in console_sections:
        if data:
            table = create_results_table(data, fields, row_fn)
            log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
            log_and_print(title, logger)
            log_and_print(DISPLAY_SEPARATOR, logger)
            for line in str(table).split("\n"):
                log_and_print(line, logger)

    # Log-only sections
    sections = [
        (progressions, "PROGRESSIONS: Prod FAILED -> Dev SUCCESS"),
        (failures, "FAILURES: FAILED in both Prod and Dev"),
    ]

    for data, title in sections:
        if data:
            table = create_results_table(data, field_names, _compile_row)
            logger.info(f"\n{DISPLAY_SEPARATOR}")
            logger.info(title)
            logger.info(DISPLAY_SEPARATOR)
            for line in str(table).split("\n"):
                logger.info(line)

    return summary_table


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print compile nightly results summary"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing result YAML files (default: results)",
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag used for output file identifier (default: current date)",
    )

    args = parser.parse_args()

    job_yaml_tag = validate_tag(args.tag) if args.tag else get_date_str()
    log_file = setup_script_logging(
        args.results_dir, "post-compile-results", args.verbose, job_yaml_tag
    )
    log_and_print(f"Logging to {log_file}", logger)

    try:
        regressions = load_yaml_safe(
            args.results_dir / f"dev-regressions__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        progressions = load_yaml_safe(
            args.results_dir / f"dev-progressions__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        failures = load_yaml_safe(
            args.results_dir / f"failures-dev-and-prod__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        passed = load_yaml_safe(
            args.results_dir / f"passed-dev-and-prod__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        other_changes = load_yaml_safe(
            args.results_dir / f"other-status-changes__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )
        submission_failures = load_yaml_safe(
            args.results_dir
            / f"dev-compile-jobs-submission-failures__{job_yaml_tag}.yaml",
            return_empty_on_not_found=True,
        )

        dev_client = load_client(args.dev_profile)
        regressions, known = filter_known_failures(
            regressions, dev_client, JobType.COMPILE, "dev_job"
        )
        if known:
            log_and_print(f"Excluded {len(known)} known compile failures", logger)

        # Re-run remaining regressions: failures flagged as regressions are
        # sometimes transient infrastructure issues. Anything that passes on
        # re-run is reclassified as an infrastructure failure.
        regressions, infra_failures = rerun_regressions(
            regressions, dev_client, job_id_key="dev_job"
        )
        if infra_failures:
            infra_path = args.results_dir / f"dev-infra-failures__{job_yaml_tag}.yaml"
            save_yaml_results(infra_failures, infra_path)
            log_and_print(f"Saved infrastructure failures: {infra_path}", logger)

        passing = {**passed, **progressions}
        passing_known = find_passing_known_failures(passing, JobType.COMPILE)

        summary_table = print_summary(
            regressions,
            progressions,
            failures,
            passed,
            known,
            passing_known,
            infra_failures,
            job_yaml_tag,
        )
        tfl_converter_summary_line = print_tfl_converter_results(
            get_tfl_converter_jobs(
                regressions,
                progressions,
                failures,
                passed,
                known,
                passing_known,
                infra_failures,
                other_changes,
                submission_failures,
            )
        )

        summary_path = args.results_dir / f"compile-summary__{job_yaml_tag}.txt"
        summary_path.write_text(
            str(summary_table) + f"\n{tfl_converter_summary_line}\n"
        )
        log_and_print(f"Saved summary table: {summary_path}", logger)

        exit_code = 0

        if regressions:
            log_and_print(f"✗ Found {len(regressions)} real regressions.", logger)
            exit_code = 1

        if infra_failures:
            log_and_print(
                f"✗ Found {len(infra_failures)} infrastructure failures "
                "(passed on re-run).",
                logger,
            )
            exit_code = 1

        if passing_known:
            log_and_print(
                f"✗ Found {len(passing_known)} known issues now passing. "
                "Remove them from known_failures.yaml.",
                logger,
            )
            exit_code = 1

        if exit_code == 0:
            log_and_print("✓ No regressions found.", logger)
        return exit_code

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
