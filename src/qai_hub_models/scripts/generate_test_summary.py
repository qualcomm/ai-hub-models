#!/usr/bin/env python3

# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import argparse
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
from tabulate import tabulate

PREREQUISITE_SKIP_PATTERN = re.compile(r"Prerequisite \w+ job ")


def clean_message(message: str, max_len: int = 100) -> str:
    """Clean up a failure message for better display."""
    message = re.sub(r"\n", " ", message)
    message = re.sub(r"\s+", " ", message)
    # Remove pipe characters as they break markdown table formatting
    message = message.replace("|", "")
    if len(message) > max_len:
        message = message[: max_len - 3] + "..."
    return message


def extract_file_and_line(
    stack_trace: str | None, message: str
) -> tuple[str | None, str | None]:
    """
    Extract file path and line number from stack trace or message.

    Parameters
    ----------
    stack_trace
        The stack trace text.
    message
        The error message.

    Returns
    -------
    file_path : str | None
        The file path from the stack trace, or None.
    line_number : str | None
        The line number from the stack trace, or None.
    """
    # No stack trace to analyze
    if not stack_trace:
        return None, None

    # Define regex patterns for file paths and line numbers
    # Pattern for file.py:123: format (common at the end of stack traces)
    file_line_pattern = r"([^\s:]+\.py):(\d+):"

    # Pattern for File "file.py", line 123 format (common in Python tracebacks)
    traceback_pattern = r'File\s+"([^"]+)",\s+line\s+(\d+)'

    # Simple approach: Just get the last file path and line number in the stack trace
    # First, check for the file.py:123: pattern which is often at the very end
    file_line_matches = re.findall(file_line_pattern, stack_trace)
    if file_line_matches:
        # Return the last match, which is typically the actual error location
        return file_line_matches[-1]

    # If that didn't work, look for the File "file.py", line 123 pattern
    traceback_matches = re.findall(traceback_pattern, stack_trace)
    if traceback_matches:
        # Return the last match, which is typically the actual error location
        return traceback_matches[-1]

    # If we still couldn't find anything, try to extract from the message
    file_match = re.search(traceback_pattern, message)
    if file_match:
        return file_match.group(1), file_match.group(2)

    return None, None


def extract_relevant_stack_trace(stack_trace: str | None, message: str) -> str | None:
    """
    Extract the most relevant part of a stack trace.

    Extracts the end of the stack trace where file and line number along with
    error is captured. This assumes the error is near the end of the stack trace,
    which is often but not always true, refer to junit xml for ground truth.

    Parameters
    ----------
    stack_trace
        The full stack trace.
    message
        The error message.

    Returns
    -------
    relevant_trace : str | None
        The most relevant part of the stack trace.
    """
    if not stack_trace:
        return None

    stack_lines = stack_trace.split("\n")

    # If the stack trace is very long, try to extract the most relevant part
    if len(stack_lines) > 10:
        # Look for lines that contain the error message
        error_type = message.split(":", 1)[0]
        error_lines = [i for i, line in enumerate(stack_lines) if error_type in line]

        if error_lines:
            # Get more lines before the error and ALL lines after (to capture
            # multi-line error messages like dicts of failed jobs)
            last_error_line = error_lines[-1]
            start_line = max(
                0, last_error_line - 7
            )  # Include 7 lines before instead of 2
            # Include all lines from the error to the end to capture full error message
            return "\n".join(stack_lines[start_line:])
        # If we can't find the error message, use the last several lines
        return "\n".join(stack_lines[-10:])  # Show 10 lines instead of 5
    if len(stack_lines) > 5:
        # For moderately long stack traces, use all lines
        return stack_trace

    # For short stack traces, use the whole thing
    return stack_trace


def collect_test_statistics(root: ET.Element) -> dict[str, int | float]:
    """
    Collect test statistics from the JUnit XML root element.

    Parameters
    ----------
    root
        The root element of the JUnit XML.

    Returns
    -------
    stats : dict[str, int | float]
        Dictionary of test statistics.
    """
    stats: dict[str, int | float] = {
        "total": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "passed": 0,
        "time": 0.0,
    }

    for testsuite in root.findall(".//testsuite"):
        stats["total"] += int(testsuite.get("tests", 0))
        stats["failures"] += int(testsuite.get("failures", 0))
        stats["errors"] += int(testsuite.get("errors", 0))
        stats["skipped"] += int(testsuite.get("skipped", 0))
        time_attr = testsuite.get("time", "0")
        # Convert time to float, defaulting to 0 if not a valid float
        if time_attr and time_attr.replace(".", "", 1).isdigit():
            stats["time"] += float(time_attr)

    stats["passed"] = (
        stats["total"] - stats["failures"] - stats["errors"] - stats["skipped"]
    )

    return stats


def parse_junit_xml(
    xml_path: str,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """
    Parse a JUnit XML file and extract test failures.

    Parameters
    ----------
    xml_path
        Path to the JUnit XML file.

    Returns
    -------
    failures : list[dict[str, Any]]
        List of dictionaries containing failure information.
    stats : dict[str, int | float]
        Dictionary of test statistics.
    """
    if not os.path.exists(xml_path):
        print(f"No test results file found at {xml_path}")
        empty_stats: dict[str, int | float] = {
            "total": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": 0,
            "time": 0.0,
        }
        return [], empty_stats

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find all testsuites
    testsuites = root.findall(".//testsuite")

    # Find all failed testcases (both failures and errors)
    failures = []

    for testsuite in testsuites:
        # Find all testcases
        testcases = testsuite.findall("testcase")

        for testcase in testcases:
            # Check for both failure and error elements
            failure_element = testcase.find("failure")
            error_element = testcase.find("error")

            if failure_element is not None or error_element is not None:
                # Use whichever element is not None
                element = (
                    failure_element if failure_element is not None else error_element
                )

                if element is None:
                    continue

                # Extract basic test information
                classname = testcase.get("classname", "")
                name = testcase.get("name", "Unknown")

                # If classname is empty but name contains dots, it might be a fully qualified name
                # In this case, split it into classname and name
                if not classname and "." in name:
                    parts = name.rsplit(".", 1)
                    if len(parts) == 2:
                        classname = parts[0]
                        name = parts[1]

                message = clean_message(element.get("message", "No message"))

                # Extract stack trace if available
                stack_trace = element.text.strip() if element.text else None

                # Extract file path and line number
                file_path, line_number = extract_file_and_line(stack_trace, message)

                # Extract relevant part of stack trace
                relevant_stack_trace = extract_relevant_stack_trace(
                    stack_trace, message
                )

                # Add failure information to the list
                failures.append(
                    {
                        "Test Class": classname,
                        "Test Name": name,
                        "Failure Reason": message,
                        "File": file_path,
                        "Line": line_number,
                        "Stack Trace": relevant_stack_trace,
                    }
                )

    # Tests skipped because a prerequisite Workbench job failed are real failures.
    blocked = 0
    for testcase in root.findall(".//testcase"):
        skipped_element = testcase.find("skipped")
        if skipped_element is None:
            continue
        raw_message = skipped_element.get("message") or skipped_element.text or ""
        if not PREREQUISITE_SKIP_PATTERN.search(raw_message):
            continue
        blocked += 1
        failures.append(
            {
                "Test Class": testcase.get("classname", ""),
                "Test Name": testcase.get("name", "Unknown"),
                "Failure Reason": clean_message(raw_message),
                "File": None,
                "Line": None,
                "Stack Trace": raw_message,
            }
        )

    stats = collect_test_statistics(root)
    stats["failures"] += blocked
    stats["skipped"] -= blocked

    return failures, stats


def generate_markdown_table(failures: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Generate a Markdown table from a list of test failures using pandas.

    Also creates a pull down stack trace right underneath the table for
    convenience of viewing.

    Parameters
    ----------
    failures
        List of dictionaries containing failure information.

    Returns
    -------
    markdown_table : str
        Markdown formatted table of test failures.
    stack_traces_section : str
        Section containing detailed stack traces.
    """
    if not failures:
        return "No test failures found.", ""

    # Create a DataFrame from the failures
    df = pd.DataFrame(failures)

    # Add a Status column with red X emoji for failures
    df["Status"] = "❌"

    # Select columns to display - include Status as the first column
    display_columns = [
        "Status",
        "Test Class",
        "Test Name",
        "Failure Reason",
        "File",
        "Line",
    ]

    # Ensure File and Line columns exist in the DataFrame
    for col in ["File", "Line"]:
        if col not in df.columns:
            df[col] = None

    # Generate a markdown table using tabulate
    table = tabulate(
        df[display_columns], headers="keys", tablefmt="pipe", showindex=False
    )

    # Generate stack traces section
    stack_traces_section = ""
    for failure in failures:
        if failure.get("Stack Trace"):
            test_name = f"{failure['Test Class']}.{failure['Test Name']}"
            stack_traces_section += f"<details>\n<summary>Stack trace for {test_name}</summary>\n\n```\n{failure['Stack Trace']}\n```\n</details>\n\n"

    return table, stack_traces_section


def generate_combined_stats_table(
    all_stats: list[tuple[str, dict[str, int | float]]],
) -> str:
    """
    Generate a combined summary table for multiple test sections.

    Parameters
    ----------
    all_stats
        List of tuples containing (name, stats) for each test section.

    Returns
    -------
    summary : str
        Markdown formatted combined test statistics table.
    """
    rows: list[dict[str, str | int | float]] = []
    for name, stats in all_stats:
        # Check if results are missing
        if stats.get("missing"):
            rows.append(
                {
                    "Status": "❌",
                    "Test": name,
                    "Total": "Workflow Failure",
                    "Passed": "",
                    "Failed": "",
                    "Errors": "",
                    "Skipped": "",
                    "Time (min)": "",
                }
            )
            continue

        # Convert time from seconds to minutes
        time_minutes = stats["time"] / 60.0

        # Determine status emoji based on failures and errors
        status_emoji = "✅" if stats["failures"] == 0 and stats["errors"] == 0 else "❌"

        rows.append(
            {
                "Status": status_emoji,
                "Test": name,
                "Total": stats["total"],
                "Passed": stats["passed"],
                "Failed": stats["failures"],
                "Errors": stats["errors"],
                "Skipped": stats["skipped"],
                "Time (min)": round(time_minutes, 2),
            }
        )

    stats_df = pd.DataFrame(rows)
    return tabulate(stats_df, headers="keys", tablefmt="pipe", showindex=False)


def generate_workflow_failures_table(
    workflow_failures: list[str], workflow_urls: list[str]
) -> str:
    """
    Generate a table for workflow failures.

    Parameters
    ----------
    workflow_failures
        List of workflow/job names that failed.
    workflow_urls
        List of URLs corresponding to each workflow failure.

    Returns
    -------
    table : str
        Markdown formatted workflow failures table.
    """
    if not workflow_failures:
        return ""

    rows = []
    for i, name in enumerate(workflow_failures):
        url = workflow_urls[i] if i < len(workflow_urls) else ""
        if url:
            rows.append({"Status": "❌", "Workflow": f"[{name}]({url})"})
        else:
            rows.append({"Status": "❌", "Workflow": name})

    df = pd.DataFrame(rows)
    return tabulate(df, headers="keys", tablefmt="pipe", showindex=False)


def write_to_github_summary(content: str) -> None:
    """
    Write content to the GitHub step summary.

    Parameters
    ----------
    content
        Content to write to the summary.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    print(f"GITHUB_STEP_SUMMARY environment variable: {summary_path}")

    if summary_path:
        with open(summary_path, "a") as f:
            f.write(content + "\n")
        print(f"Successfully wrote to {summary_path}")
    else:
        print("GITHUB_STEP_SUMMARY not set, printing to console instead:")
        print(content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate test failure summary from JUnit XML files"
    )
    parser.add_argument(
        "--title",
        "-t",
        required=False,
        default="Test",
        help="Title for the summary (e.g., 'Scorecard', 'py3.10 Test')",
    )
    parser.add_argument(
        "--name",
        "-n",
        action="append",
        required=True,
        help="Name for a test section (can be specified multiple times, must match --junit-xml count)",
    )
    parser.add_argument(
        "--junit-xml",
        "-j",
        action="append",
        required=True,
        help="JUnit XML file (can be specified multiple times, must match --name count)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for summary.md file",
    )
    parser.add_argument(
        "--workflow-failure",
        action="append",
        default=[],
        dest="workflow_failure",
        help="Name of a workflow/job that failed (can be specified multiple times)",
    )
    parser.add_argument(
        "--workflow-failure-url",
        action="append",
        default=[],
        dest="workflow_failure_url",
        help="URL for a workflow failure (must match --workflow-failure count)",
    )
    args = parser.parse_args()

    # Validate that input and name counts match
    if len(args.junit_xml) != len(args.name):
        print(
            f"Error: Number of --junit-xml ({len(args.junit_xml)}) must match number of --name ({len(args.name)})"
        )
        return

    summary_sections = [f"## {args.title} Summary\n"]

    # Collect all stats and failures
    all_stats: list[tuple[str, dict[str, int | float]]] = []
    all_failures: list[tuple[str, list[dict[str, Any]]]] = []

    # Generate workflow failures table if any
    if args.workflow_failure:
        summary_sections.append("### Workflow Failures\n")
        workflow_failures_table = generate_workflow_failures_table(
            args.workflow_failure, args.workflow_failure_url
        )
        summary_sections.append(workflow_failures_table + "\n\n")

    for input_path, name in zip(args.junit_xml, args.name, strict=False):
        if not os.path.isfile(input_path):
            print(f"Warning: Input file does not exist: {input_path}")
            # Add a "missing" entry that shows as a failure
            missing_stats: dict[str, int | float] = {
                "total": 0,
                "failures": 1,  # Count as failure
                "errors": 0,
                "skipped": 0,
                "passed": 0,
                "time": 0.0,
                "missing": True,
            }
            all_stats.append((name, missing_stats))
            continue

        print(f"Processing {input_path} for {name}")
        failures, stats = parse_junit_xml(input_path)
        all_stats.append((name, stats))
        if failures:
            all_failures.append((name, failures))

    # Generate combined stats table
    if all_stats:
        summary_sections.append("### Test Results\n")
        stats_table = generate_combined_stats_table(all_stats)
        summary_sections.append(stats_table + "\n\n")

    # Add failures section if any
    for name, failures in all_failures:
        summary_sections.append(f"### {name} Failures\n")
        table, stack_traces_section = generate_markdown_table(failures)
        summary_sections.append(table + "\n\n")
        if stack_traces_section:
            summary_sections.append("#### Stack Traces\n")
            summary_sections.append(stack_traces_section)

    summary_text = "\n".join(summary_sections)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write summary to file and GitHub step summary
    with open(args.output, "w") as f:
        f.write(summary_text)
    write_to_github_summary(summary_text)


if __name__ == "__main__":
    main()
