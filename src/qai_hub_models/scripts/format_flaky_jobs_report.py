#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
Script to format flaky jobs report for Slack.
This script reads the flaky_jobs.yaml file and formats it as a bullet point list for Slack.
"""

import argparse
from pathlib import Path

from qai_hub_models.scripts.retry_failed_inference_jobs import FlakyJobsYaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format flaky jobs report for Slack")
    parser.add_argument(
        "--flaky-jobs-file",
        type=str,
        required=True,
        help="Path to the flaky_jobs.yaml file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the output file for the formatted report",
    )
    return parser.parse_args()


def format_flaky_jobs_report(flaky_jobs_data: FlakyJobsYaml) -> str:
    """
    Format the flaky jobs data as a device-centric summary for Slack.

    Parameters
    ----------
    flaky_jobs_data
        The flaky jobs data from the YAML file.

    Returns
    -------
    formatted_report : str
        A formatted string with device-focused summary statistics and breakdowns.
    """
    if not flaky_jobs_data.flaky_jobs:
        return "No flaky jobs found."

    if flaky_jobs_data.total_flaky_jobs == 0:
        return "No flaky jobs found."

    # Build the device-centric report
    report_lines = []

    # Add title with emoji and total count
    report_lines.append(
        f":warning: [Scorecard Flaky Jobs] Found and fixed {flaky_jobs_data.total_flaky_jobs} flaky inference jobs in scorecard run."
    )
    report_lines.append("")

    # Summary by device (which devices failed and how often)
    if flaky_jobs_data.by_device:
        report_lines.append("📱 Failures by Device:")
        for device, count in sorted(
            flaky_jobs_data.by_device.items(), key=lambda x: x[1], reverse=True
        ):
            report_lines.append(f"• {device}: {count} failed jobs")
        report_lines.append("")

    # Device + failure reason breakdown
    if flaky_jobs_data.flaky_jobs:
        report_lines.append("🔍 Device Failure Details:")

        # Group by device + failure reason
        device_failure_counts: dict[tuple[str, str], int] = {}
        for job_details in flaky_jobs_data.flaky_jobs.values():
            device = job_details.device
            reason = job_details.failure_reason
            key = (device, reason)
            device_failure_counts[key] = device_failure_counts.get(key, 0) + 1

        # Sort by device name, then by failure count within each device
        sorted_entries = sorted(
            device_failure_counts.items(), key=lambda x: (x[0][0], -x[1])
        )

        current_device = None
        for (device, reason), count in sorted_entries:
            if device != current_device:
                if current_device is not None:
                    report_lines.append("")  # Add spacing between devices
                report_lines.append(f"{device}:")
                current_device = device
            report_lines.append(f"  • {reason} ({count} jobs)")
        report_lines.append("")

    return "\n".join(report_lines)


def main() -> None:
    args = parse_args()
    flaky_jobs_file = Path(args.flaky_jobs_file)
    output_file = Path(args.output_file)

    if not flaky_jobs_file.exists():
        print(f"Flaky jobs file not found at {flaky_jobs_file}")
        return

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read the flaky jobs data from YAML
    flaky_jobs_data = FlakyJobsYaml.from_yaml(flaky_jobs_file)

    # Format the report
    formatted_report = format_flaky_jobs_report(flaky_jobs_data)

    # Write the formatted report to the output file
    with open(output_file, "w") as f:
        f.write(formatted_report)

    print(f"Formatted report written to {output_file}")


if __name__ == "__main__":
    main()
