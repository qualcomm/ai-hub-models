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


def clean_message(message: str) -> str:
    """Clean up a failure message for better display."""
    message = re.sub(r"\n", " ", message)
    message = re.sub(r"\s+", " ", message)
    if len(message) > 100:
        message = message[:97] + "..."
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
            # Get more lines before and a few lines after the error
            last_error_line = error_lines[-1]
            start_line = max(
                0, last_error_line - 7
            )  # Include 7 lines before instead of 2
            end_line = min(len(stack_lines), last_error_line + 3)
            return "\n".join(stack_lines[start_line:end_line])
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

    # Collect test statistics
    stats = collect_test_statistics(root)

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


def generate_stats_summary(stats: dict[str, int | float]) -> str:
    """
    Generate a summary of test statistics.

    Parameters
    ----------
    stats
        Dictionary containing test statistics.

    Returns
    -------
    summary : str
        Markdown formatted test statistics.
    """
    # Convert time from seconds to minutes
    time_minutes = stats["time"] / 60.0

    # Determine status emoji based on failures and errors
    status_emoji = "✅" if stats["failures"] == 0 and stats["errors"] == 0 else "❌"

    # Create a DataFrame for the statistics
    stats_df = pd.DataFrame(
        [
            {
                "Status": status_emoji,
                "Total": stats["total"],
                "Passed": stats["passed"],
                "Failed": stats["failures"],
                "Errors": stats["errors"],
                "Skipped": stats["skipped"],
                "Time (min)": round(time_minutes, 2),
            }
        ]
    )

    # Generate a markdown table
    return tabulate(stats_df, headers="keys", tablefmt="pipe", showindex=False)


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


def combine_junit_results(
    file_paths: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """
    Combine results from multiple JUnit XML files.

    Parameters
    ----------
    file_paths
        List of file paths to JUnit XML files.

    Returns
    -------
    failures : list[dict[str, Any]]
        Combined list of failures from all test files.
    stats : dict[str, int | float]
        Combined test statistics.
    """
    all_failures = []
    combined_stats: dict[str, int | float] = {
        "total": 0,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
        "passed": 0,
        "time": 0.0,
    }

    for file_path in file_paths:
        failures, stats = parse_junit_xml(file_path)
        all_failures.extend(failures)

        # Combine stats
        combined_stats["total"] += stats["total"]
        combined_stats["failures"] += stats["failures"]
        combined_stats["errors"] += stats["errors"]
        combined_stats["skipped"] += stats["skipped"]
        combined_stats["passed"] += stats["passed"]
        combined_stats["time"] += stats["time"]

    return all_failures, combined_stats


def write_combined_xml(file_paths: list[str], output_path: str) -> None:
    """
    Combine multiple JUnit XML files into a single file.

    Parameters
    ----------
    file_paths
        List of paths to JUnit XML files to combine.
    output_path
        Path to write the combined JUnit XML file.
    """
    if not file_paths:
        print("No input files to combine")
        return

    # Collect all testsuites from all files
    all_testsuites: list[ET.Element] = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_time = 0.0

    for xml_file in file_paths:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Handle both <testsuites> and <testsuite> roots
            if root.tag == "testsuites":
                for testsuite in root.findall("testsuite"):
                    all_testsuites.append(testsuite)
                    total_tests += int(testsuite.get("tests", 0))
                    total_failures += int(testsuite.get("failures", 0))
                    total_errors += int(testsuite.get("errors", 0))
                    total_skipped += int(testsuite.get("skipped", 0))
                    total_time += float(testsuite.get("time", 0))
            elif root.tag == "testsuite":
                all_testsuites.append(root)
                total_tests += int(root.get("tests", 0))
                total_failures += int(root.get("failures", 0))
                total_errors += int(root.get("errors", 0))
                total_skipped += int(root.get("skipped", 0))
                total_time += float(root.get("time", 0))
            else:
                print(f"Warning: Unknown root element '{root.tag}' in {xml_file}")

        except ET.ParseError as e:
            print(f"Warning: Failed to parse {xml_file}: {e}")
        except Exception as e:
            print(f"Warning: Error processing {xml_file}: {e}")

    if not all_testsuites:
        print("No test suites found in any input files")
        return

    # Create combined root element
    combined_root = ET.Element("testsuites")
    combined_root.set("tests", str(total_tests))
    combined_root.set("failures", str(total_failures))
    combined_root.set("errors", str(total_errors))
    combined_root.set("skipped", str(total_skipped))
    combined_root.set("time", f"{total_time:.3f}")

    # Add all testsuites to combined root
    for testsuite in all_testsuites:
        combined_root.append(testsuite)

    # Write combined XML
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tree = ET.ElementTree(combined_root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)

    print(f"Combined {len(file_paths)} files into {output_path}")
    print(f"  Total tests: {total_tests}")
    print(f"  Failures: {total_failures}")
    print(f"  Errors: {total_errors}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Time: {total_time:.3f}s")


def process_test_results(
    test_type: str,
    failures: list[dict[str, Any]],
    stats: dict[str, int | float],
    summary_sections: list[str],
) -> None:
    """
    Process test results and add them to the summary sections.

    Parameters
    ----------
    test_type
        Type of tests (e.g., "QAIHM", "Model").
    failures
        List of test failures.
    stats
        Dictionary of test statistics.
    summary_sections
        List to append summary sections to.
    """
    summary_sections.append(f"### {test_type} Tests\n")

    # Add test statistics
    summary_sections.append("#### Test Statistics\n")
    stats_table = generate_stats_summary(stats)
    summary_sections.append(stats_table + "\n\n")

    # Add failures if any
    if failures:
        summary_sections.append("#### Test Failures\n")
        table, stack_traces_section = generate_markdown_table(failures)
        summary_sections.append(table + "\n\n")

        # Add stack traces section if there are any stack traces
        if stack_traces_section:
            summary_sections.append("#### Stack Traces\n")
            summary_sections.append(stack_traces_section)
    elif stats["failures"] > 0 or stats["errors"] > 0:
        # If there are failures or errors in the stats but no failure details
        summary_sections.append("#### Test Failures\n")
        summary_sections.append(
            f"⚠️ Tests failed: {stats['failures']} failures and {stats['errors']} errors detected.\n\n"
        )
    else:
        summary_sections.append("All tests passed! 🎉\n\n")


def find_xml_files_recursive(directory: str, pattern: str = "*.xml") -> list[str]:
    """
    Recursively find all XML files in a directory.

    Parameters
    ----------
    directory
        Directory to search in.
    pattern
        Glob pattern to match (default: "*.xml").

    Returns
    -------
    file_paths : list[str]
        List of file paths.
    """
    import glob

    return sorted(glob.glob(os.path.join(directory, "**", pattern), recursive=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine JUnit XML files and generate test failure summary"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing JUnit XML files (searched recursively)",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for combined-junit.xml and summary.md",
    )
    args = parser.parse_args()

    # Find all XML files recursively in input directory
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        return

    all_xml_files = find_xml_files_recursive(args.input)
    if not all_xml_files:
        print(f"No XML files found in {args.input}")
        return

    print(f"Found {len(all_xml_files)} XML files in {args.input}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Write combined XML
    write_combined_xml(all_xml_files, os.path.join(args.output, "combined-junit.xml"))

    # Generate summary
    failures, stats = combine_junit_results(all_xml_files)
    summary_sections = ["## Test Failure Summary\n"]
    process_test_results("Combined", failures, stats, summary_sections)
    summary_text = "\n".join(summary_sections)

    # Write summary to file and GitHub step summary
    with open(os.path.join(args.output, "summary.md"), "w") as f:
        f.write(summary_text)
    write_to_github_summary(summary_text)


if __name__ == "__main__":
    main()
