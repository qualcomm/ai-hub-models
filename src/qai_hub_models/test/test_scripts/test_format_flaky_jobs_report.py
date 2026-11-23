# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Unit tests for format_flaky_jobs_report.py script.
- test_format_flaky_jobs_report_with_flaky_jobs: Validates the complete enhanced format including
  summary by failure reason, detailed job listings with model+device+error combinations,
  job counts, and actions taken section
- test_format_flaky_jobs_report_no_flaky_jobs: Ensures proper "No flaky jobs found" message
  when the jobs list is empty
- test_format_flaky_jobs_report_missing_flaky_jobs_key: Handles malformed JSON input gracefully
  when the flaky_jobs key is missing entirely
- test_format_flaky_jobs_report_with_missing_fields: Tests resilience when individual job entries
  have missing model, device, or failure_reason fields (uses defaults like "unknown")
- test_format_flaky_jobs_report_single_job: Verifies all sections are properly formatted even
  with just one flaky job
- test_parse_args: Validates command-line argument parsing for input and output file paths
- test_main_functionality: End-to-end integration test that writes JSON input, runs the formatter,
  and validates the complete enhanced output format
- test_main_with_missing_input_file: Ensures graceful error handling when the input JSON file
  doesn't exist (no crash, no output file created)
- test_main_with_no_flaky_jobs: Tests the complete workflow when input contains empty flaky jobs
  data, ensuring proper "No flaky jobs found" output
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from qai_hub_models.scripts.format_flaky_jobs_report import (
    format_flaky_jobs_report,
    parse_args,
)
from qai_hub_models.scripts.retry_failed_inference_jobs import FlakyJobsYaml


def test_format_flaky_jobs_report_with_flaky_jobs() -> None:
    """Test formatting flaky jobs data with multiple jobs."""
    flaky_jobs_data = FlakyJobsYaml(
        total_flaky_jobs=4,
        by_failure_reason={
            "Job timed out after 8h": 2,
            "Device unavailable": 1,
            "Memory allocation error": 1,
        },
        by_device={
            "samsung_galaxy_s23": 1,
            "snapdragon_8_gen_3": 1,
            "samsung_galaxy_s24": 1,
            "snapdragon_x_elite": 1,
        },
        flaky_jobs={
            "resnet50_w8a8_ONNX-samsung_galaxy_s23_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="resnet50_w8a8_ONNX-samsung_galaxy_s23_profile",
                model="resnet50",
                device="samsung_galaxy_s23",
                original_job_id="job_12345",
                failure_reason="Job timed out after 8h",
            ),
            "resnet50_w8a8_ONNX-snapdragon_8_gen_3_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="resnet50_w8a8_ONNX-snapdragon_8_gen_3_profile",
                model="resnet50",
                device="snapdragon_8_gen_3",
                original_job_id="job_12346",
                failure_reason="Job timed out after 8h",
            ),
            "mobilenet_v2_w8a8_ONNX-samsung_galaxy_s24_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="mobilenet_v2_w8a8_ONNX-samsung_galaxy_s24_profile",
                model="mobilenet_v2",
                device="samsung_galaxy_s24",
                original_job_id="job_67890",
                failure_reason="Device unavailable",
            ),
            "yolo_v5_w8a8_ONNX-snapdragon_x_elite_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="yolo_v5_w8a8_ONNX-snapdragon_x_elite_profile",
                model="yolo_v5",
                device="snapdragon_x_elite",
                original_job_id="job_11111",
                failure_reason="Memory allocation error",
            ),
        },
    )

    result = format_flaky_jobs_report(flaky_jobs_data)

    # Check that it contains the device summary section
    assert "📱 Failures by Device:" in result
    assert "• samsung_galaxy_s23: 1 failed jobs" in result
    assert "• snapdragon_8_gen_3: 1 failed jobs" in result
    assert "• samsung_galaxy_s24: 1 failed jobs" in result
    assert "• snapdragon_x_elite: 1 failed jobs" in result

    # Check that it contains the device failure details section
    assert "🔍 Device Failure Details:" in result
    assert "samsung_galaxy_s23:" in result
    assert "  • Job timed out after 8h (1 jobs)" in result
    assert "snapdragon_8_gen_3:" in result
    assert "samsung_galaxy_s24:" in result
    assert "  • Device unavailable (1 jobs)" in result
    assert "snapdragon_x_elite:" in result
    assert "  • Memory allocation error (1 jobs)" in result

    # Check that it contains the title with count
    assert (
        ":warning: [Scorecard Flaky Jobs] Found and fixed 4 flaky inference jobs in scorecard run."
        in result
    )


def test_format_flaky_jobs_report_no_flaky_jobs() -> None:
    """Test formatting when no flaky jobs are present."""
    flaky_jobs_data = FlakyJobsYaml()

    result = format_flaky_jobs_report(flaky_jobs_data)
    assert result == "No flaky jobs found."


def test_format_flaky_jobs_report_missing_flaky_jobs_key() -> None:
    """Test formatting when flaky_jobs key is missing."""
    flaky_jobs_data = FlakyJobsYaml()

    result = format_flaky_jobs_report(flaky_jobs_data)
    assert result == "No flaky jobs found."


def test_format_flaky_jobs_report_with_missing_fields() -> None:
    """Test formatting with jobs that have missing fields."""
    flaky_jobs_data = FlakyJobsYaml(
        total_flaky_jobs=4,
        by_failure_reason={
            "Job timed out after 8h": 1,
            "Device unavailable": 1,
            "Memory allocation error": 1,
            "Unknown failure": 1,
        },
        by_device={
            "samsung_galaxy_s23": 1,
            "snapdragon_8_gen_3": 1,
            "samsung_galaxy_s24": 1,
            "snapdragon_x_elite": 1,
        },
        flaky_jobs={
            "resnet50_w8a8_ONNX-samsung_galaxy_s23_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="resnet50_w8a8_ONNX-samsung_galaxy_s23_profile",
                model="resnet50",
                device="samsung_galaxy_s23",
                original_job_id="job_12345",
                failure_reason="Job timed out after 8h",
            ),
            "unknown_w8a8_ONNX-snapdragon_8_gen_3_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="unknown_w8a8_ONNX-snapdragon_8_gen_3_profile",
                model="unknown",
                device="snapdragon_8_gen_3",
                original_job_id="job_67890",
                failure_reason="Device unavailable",
            ),
            "yolo_v5_w8a8_ONNX-samsung_galaxy_s24_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="yolo_v5_w8a8_ONNX-samsung_galaxy_s24_profile",
                model="yolo_v5",
                device="samsung_galaxy_s24",
                original_job_id="job_unknown",
                failure_reason="Memory allocation error",
            ),
            "mobilenet_v2_w8a8_ONNX-snapdragon_x_elite_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="mobilenet_v2_w8a8_ONNX-snapdragon_x_elite_profile",
                model="mobilenet_v2",
                device="snapdragon_x_elite",
                original_job_id="job_33333",
                failure_reason="Unknown failure",
            ),
        },
    )

    result = format_flaky_jobs_report(flaky_jobs_data)

    # Check that it contains the device failure details section
    assert "🔍 Device Failure Details:" in result

    # Check that it handles missing fields gracefully in the device-grouped format
    assert "samsung_galaxy_s23:" in result
    assert "  • Job timed out after 8h (1 jobs)" in result
    assert "snapdragon_8_gen_3:" in result
    assert "  • Device unavailable (1 jobs)" in result
    assert "samsung_galaxy_s24:" in result
    assert "  • Memory allocation error (1 jobs)" in result
    assert "snapdragon_x_elite:" in result
    assert "  • Unknown failure (1 jobs)" in result


def test_format_flaky_jobs_report_single_job() -> None:
    """Test formatting with a single flaky job."""
    flaky_jobs_data = FlakyJobsYaml(
        total_flaky_jobs=1,
        by_failure_reason={
            "Job timed out after 8h": 1,
        },
        by_device={
            "samsung_galaxy_s23": 1,
        },
        flaky_jobs={
            "resnet50_w8a8_ONNX-samsung_galaxy_s23_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="resnet50_w8a8_ONNX-samsung_galaxy_s23_profile",
                model="resnet50",
                device="samsung_galaxy_s23",
                original_job_id="job_12345",
                failure_reason="Job timed out after 8h",
            )
        },
    )

    result = format_flaky_jobs_report(flaky_jobs_data)

    # Check that it contains all sections for a single job
    assert "📱 Failures by Device:" in result
    assert "• samsung_galaxy_s23: 1 failed jobs" in result
    assert "🔍 Device Failure Details:" in result
    assert "samsung_galaxy_s23:" in result
    assert "  • Job timed out after 8h (1 jobs)" in result
    assert (
        ":warning: [Scorecard Flaky Jobs] Found and fixed 1 flaky inference jobs in scorecard run."
        in result
    )


def test_parse_args() -> None:
    """Test argument parsing."""
    # Test with required arguments
    with patch.object(
        sys,
        "argv",
        [
            "format_flaky_jobs_report.py",
            "--flaky-jobs-file",
            "input.json",
            "--output-file",
            "output.txt",
        ],
    ):
        args = parse_args()
        assert args.flaky_jobs_file == "input.json"
        assert args.output_file == "output.txt"


def test_main_functionality() -> None:
    """Test the main functionality with temporary files."""
    # Create test data using FlakyJobsYaml
    flaky_jobs_data = FlakyJobsYaml(
        total_flaky_jobs=2,
        by_failure_reason={
            "Job timed out after 8h": 1,
            "Device unavailable": 1,
        },
        by_device={
            "samsung_galaxy_s23": 1,
            "snapdragon_8_gen_3": 1,
        },
        flaky_jobs={
            "resnet50_w8a8_ONNX-samsung_galaxy_s23_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="resnet50_w8a8_ONNX-samsung_galaxy_s23_profile",
                model="resnet50",
                device="samsung_galaxy_s23",
                original_job_id="job_12345",
                failure_reason="Job timed out after 8h",
            ),
            "mobilenet_v2_w8a8_ONNX-snapdragon_8_gen_3_profile": FlakyJobsYaml.FlakyJobsDetails(
                job_key="mobilenet_v2_w8a8_ONNX-snapdragon_8_gen_3_profile",
                model="mobilenet_v2",
                device="snapdragon_8_gen_3",
                original_job_id="job_67890",
                failure_reason="Device unavailable",
            ),
        },
    )

    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "flaky_jobs.yaml"
        output_file = Path(temp_dir) / "formatted_report.txt"

        # Write test data to input file
        flaky_jobs_data.to_yaml(input_file)

        # Mock sys.argv and run main
        with patch.object(
            sys,
            "argv",
            [
                "format_flaky_jobs_report.py",
                "--flaky-jobs-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        ):
            # Import and run main
            from qai_hub_models.scripts.format_flaky_jobs_report import main

            main()

        # Verify output file was created and contains expected content
        assert output_file.exists()
        with open(output_file) as f:
            result = f.read()

        # Check that the device-centric format is present
        assert "📱 Failures by Device:" in result
        assert "• samsung_galaxy_s23: 1 failed jobs" in result
        assert "• snapdragon_8_gen_3: 1 failed jobs" in result
        assert "🔍 Device Failure Details:" in result
        assert "samsung_galaxy_s23:" in result
        assert "  • Job timed out after 8h (1 jobs)" in result
        assert "snapdragon_8_gen_3:" in result
        assert "  • Device unavailable (1 jobs)" in result
        assert (
            ":warning: [Scorecard Flaky Jobs] Found and fixed 2 flaky inference jobs in scorecard run."
            in result
        )


def test_main_with_missing_input_file() -> None:
    """Test main function behavior when input file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "nonexistent.json"
        output_file = Path(temp_dir) / "output.txt"

        # Mock sys.argv
        with patch.object(
            sys,
            "argv",
            [
                "format_flaky_jobs_report.py",
                "--flaky-jobs-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        ):
            # Import and run main - should handle missing file gracefully
            from qai_hub_models.scripts.format_flaky_jobs_report import main

            main()  # Should not raise an exception

        # Output file should not be created when input file is missing
        assert not output_file.exists()


def test_main_with_no_flaky_jobs() -> None:
    """Test main function with empty flaky jobs data."""
    flaky_jobs_data = FlakyJobsYaml()
    expected_output = "No flaky jobs found."

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "empty_flaky_jobs.yaml"
        output_file = Path(temp_dir) / "formatted_report.txt"

        # Write empty test data to input file
        flaky_jobs_data.to_yaml(input_file, write_if_empty=True, delete_if_empty=False)

        # Mock sys.argv and run main
        with patch.object(
            sys,
            "argv",
            [
                "format_flaky_jobs_report.py",
                "--flaky-jobs-file",
                str(input_file),
                "--output-file",
                str(output_file),
            ],
        ):
            from qai_hub_models.scripts.format_flaky_jobs_report import main

            main()

        # Verify output file was created and contains expected content
        assert output_file.exists()
        with open(output_file) as f:
            result = f.read()
        assert result == expected_output
