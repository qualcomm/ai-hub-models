#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

"""
Script to retry failed inference jobs that failed due to known flaky reasons.
This script is intended to be run after the exec_scorecard job in the scorecard.yml workflow
and right before the collection workflows.

The script will:
1. Load the inference jobs YAML file from either:
   - The artifacts directory specified by QAIHM_TEST_ARTIFACTS_DIR (default)
   - A custom directory specified by the --artifacts-dir argument
2. Find jobs with failure reasons that match known flaky reasons
   - By default, only checks jobs that have already completed
   - With --wait-for-jobs, waits for running jobs to complete before checking
3. Clone those jobs to retry them
4. Update the YAML file with the new job IDs
5. Save the updated YAML back to the same file location

The updated YAMLs will be saved to the same location as the original YAMLs.
"""

import argparse
import multiprocessing
import time
from pathlib import Path
from typing import cast

import qai_hub as hub
from pydantic import Field

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.device import ScorecardDevice
from qai_hub_models.scorecard.envvars import ArtifactsDirEnvvar, DeploymentEnvvar
from qai_hub_models.scorecard.results.yaml import InferenceScorecardJobYaml
from qai_hub_models.utils.base_config import BaseQAIHMConfig
from qai_hub_models.utils.hub_clients import (
    get_default_hub_deployment,
    get_scorecard_client_or_raise,
)


class FlakyJobsYaml(BaseQAIHMConfig):
    """Schema for flaky jobs information."""

    class FlakyJobsDetails(BaseQAIHMConfig):
        """Schema for individual flaky job details."""

        job_key: str
        original_job_id: str
        model: str
        device: str
        failure_reason: str
        new_job_id: str | None = None

    # Summary statistics
    total_flaky_jobs: int = 0
    by_failure_reason: dict[str, int] = Field(default_factory=dict)
    by_device: dict[str, int] = Field(default_factory=dict)
    by_model: dict[str, int] = Field(default_factory=dict)

    # Dictionary of flaky job details using typed sub-config, Field used to create new dict for each instance.
    flaky_jobs: dict[str, FlakyJobsDetails] = Field(default_factory=dict)


# List of known flaky failure reasons.
FLAKY_FAILURES = [
    "Failed (Job timed out after 8h)",
    "Failed (Waiting for device timed out after 6h)",
    "Failed (Failed to profile the model: unexpected device error)",
    "Failed (Error uploading to QDC: status code=503.)",
    "Failed (Error uploading artifact to QDC. Response code=500)",
    "Failed (upload to device farm failed)",
]


def is_flaky_failure(failure_reason: str | None) -> bool:
    """
    Check if a failure reason is in the list of known flaky reasons.

    Parameters
    ----------
    failure_reason
        The failure reason string from the job.

    Returns
    -------
    is_flaky : bool
        True if the failure reason is considered flaky, False otherwise.
    """
    if not failure_reason:
        return False

    formatted_reason = f"Failed ({failure_reason})"

    # Check if any known flaky failure reason is in the formatted reason (case insensitive)
    return any(reason.lower() in formatted_reason.lower() for reason in FLAKY_FAILURES)


def get_job_failure_reason(
    hub_client: hub.client.Client, job_id: str, wait_for_jobs: bool = False
) -> str | None:
    """
    Get the failure reason for a job if it failed.

    Parameters
    ----------
    hub_client
        AI Hub Workbench client to use for API calls.
    job_id
        The ID of the job to check.
    wait_for_jobs
        If True, wait for the job to complete before checking its status - useful for testing.

    Returns
    -------
    failure_reason : str | None
        The failure reason if the job failed, None otherwise.
    """
    job = hub_client.get_job(job_id)

    if wait_for_jobs:
        print(f"Waiting for job {job_id} to complete...")
        job_status = (
            job.wait()
        )  # This waits for the job to complete and returns the final status
    else:
        job_status = (
            job.get_status()
        )  # This just gets the current status without waiting

    return job_status.message if job_status.failure else None


def check_job_for_flaky_failure(
    hub_client: hub.client.Client,
    job_key: str,
    job_id: str,
    test_mode: bool = False,
    wait_for_jobs: bool = False,
) -> tuple[str, str, bool]:
    """
    Check if a job has a flaky failure.

    Parameters
    ----------
    hub_client
        AI Hub Workbench client to use for API calls.
    job_key
        The key of the job in the YAML.
    job_id
        The ID of the job to check.
    test_mode
        If True, treat all jobs as flaky regardless of failure status.
    wait_for_jobs
        If True, wait for the job to complete before checking its status - useful for testing.

    Returns
    -------
    job_key : str
        Key identifying the job.
    job_id : str
        ID of the job.
    has_flaky_failure : bool
        Whether the job has a flaky failure.
    """
    # In test mode, treat all jobs as flaky (for efficient testing, choose to run just ONE model in scorecard workflow)
    # Expect to see updated job ID from this script and make sure said ID is used in collect results workflow.
    if test_mode:
        print(f"[TEST MODE] Treating job {job_id} (key: {job_key}) as flaky")
        return (job_key, job_id, True)

    # Get the failure reason if any
    failure_reason = get_job_failure_reason(hub_client, job_id, wait_for_jobs)

    # Check if it's a flaky failure
    if failure_reason and is_flaky_failure(failure_reason):
        formatted_reason = f"Failed ({failure_reason})"
        print(f"Found flaky failure for job {job_id} (key: {job_key})")
        print(f"  Failure reason: {formatted_reason}")
        return (job_key, job_id, True)

    return (job_key, job_id, False)


def process_flaky_job(
    hub_client: hub.client.Client, job_key: str, job_id: str
) -> tuple[str, str, bool]:
    """
    Process a job that has a flaky failure.

    Parameters
    ----------
    hub_client
        AI Hub Workbench client to use for API calls.
    job_key
        The key of the job in the YAML.
    job_id
        The ID of the job to check.

    Returns
    -------
    job_key : str
        Key identifying the job.
    new_job_id : str
        ID of the newly created job.
    success : bool
        Always True to indicate successful job creation.
    """
    # Clone the job
    print(f"Cloning job {job_id}...")
    prev_job = hub_client.get_job(job_id)
    assert isinstance(prev_job, hub.InferenceJob)
    new_job = hub_client.submit_inference_job(
        prev_job.model,
        prev_job.device,
        name=prev_job.name,
        options=prev_job.options,
    )
    print(f"  Submitted new job: {new_job.job_id}")

    return (job_key, new_job.job_id, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry failed inference jobs that failed due to known flaky reasons"
    )
    ArtifactsDirEnvvar.add_arg(parser)
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without actually cloning jobs",
    )
    parser.add_argument(
        "--collect-failure-reasons",
        action="store_true",
        help="Collect and print all failure reasons without retrying jobs",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Testing mode: resubmit all jobs regardless of failure status",
    )
    parser.add_argument(
        "--wait-for-jobs",
        action="store_true",
        help="Wait for jobs to complete before checking their status. This ensures running jobs are properly evaluated once they finish.",
    )
    parser.add_argument(
        "--flaky-jobs-output",
        type=str,
        help="Path to output file where flaky job IDs will be dumped in YAML format",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run
    collect_reasons = args.collect_failure_reasons
    test_mode = args.test_mode
    wait_for_jobs = args.wait_for_jobs
    flaky_jobs_output = args.flaky_jobs_output

    if wait_for_jobs:
        print("\n=== WAITING FOR JOBS TO COMPLETE ===")
        print(
            "This script will wait for each job to complete before checking its status."
        )

    if test_mode:
        print("\n=== RUNNING IN TEST MODE ===")
        print(
            "All jobs will be treated as flaky and resubmitted regardless of failure status"
        )
        print("This is intended for testing the fix_flaky_jobs workflow only\n")

    # Get the inference jobs YAML file path
    inference_jobs_file = ScorecardArtifact.INFERENCE_YAML.path
    if not inference_jobs_file.exists():
        print(f"Inference jobs file not found at {inference_jobs_file}")
        return

    print(f"Loading inference jobs from: {inference_jobs_file}")

    # Load the inference jobs YAML
    inference_yaml = InferenceScorecardJobYaml.from_test_artifacts()

    hub_client = get_scorecard_client_or_raise(args.deployment)
    print(f"Successfully initialized Hub client for deployment: {args.deployment}")

    start_time = time.time()
    print("Creating a pool of 15 worker processes for parallel operations...")
    pool = multiprocessing.Pool(processes=15)

    print("Checking for flaky failures in parallel through inference-jobs.yaml...")

    # Prepare arguments for parallel processing
    args_list = [
        (hub_client, job_key, job_id, test_mode, wait_for_jobs)
        for job_key, job_id in inference_yaml.mapping.items()
    ]

    # Process jobs in parallel
    results = pool.starmap(check_job_for_flaky_failure, args_list)

    # Create typed flaky jobs info structure
    flaky_jobs_info = FlakyJobsYaml()

    # Process results and collect all reasons - print to console or clone jobs and update YAML.
    if collect_reasons:
        print("Collecting all failure reasons in parallel...")

        # Prepare arguments for parallel processing - job IDs and wait_for_jobs flag
        failure_args_list = [
            (hub_client, job_id, wait_for_jobs)
            for job_id in inference_yaml.mapping.values()
        ]

        # Process jobs in parallel to get all failure reasons
        failure_results = pool.starmap(get_job_failure_reason, failure_args_list)

        # Build dictionary of failure reasons to counts
        failure_reasons: dict[str, int] = {}

        # Process results
        for i, failure_reason in enumerate(failure_results):
            if failure_reason:
                job_id = list(inference_yaml.mapping.values())[i]
                job_key = list(inference_yaml.mapping.keys())[i]

                formatted_reason = f"Failed ({failure_reason})"
                # Add to the dictionary or increment the count
                if formatted_reason in failure_reasons:
                    failure_reasons[formatted_reason] += 1
                else:
                    failure_reasons[formatted_reason] = 1

                # Mark if it's a flaky failure
                is_flaky = is_flaky_failure(failure_reason)
                if is_flaky:
                    print(f"Found flaky failure for job {job_id} (key: {job_key})")
                    print(f"  Failure reason: {formatted_reason}")

        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60.0
        print(
            f"\n=== All Failure Reasons (collected in {elapsed_minutes:.2f} minutes) ==="
        )
        for reason, count in sorted(
            failure_reasons.items(), key=lambda x: x[1], reverse=True
        ):
            # Mark flaky failures with an indicator on the console
            is_flaky = any(flaky.lower() in reason.lower() for flaky in FLAKY_FAILURES)
            flaky_indicator = " [FLAKY]" if is_flaky else ""
            print(f"Count: {count}, Reason: {reason}{flaky_indicator}")

        pool.close()
        return
    # Build dictionary of failed job keys
    failed_job_keys = {
        job_key: job_id
        for job_key, job_id, has_flaky_failure in results
        if has_flaky_failure
    }

    # Get known device names for reliable device extraction
    known_device_names = {device.name for device in ScorecardDevice.all_devices()}

    # Process each flaky job to collect information
    for job_key, job_id in failed_job_keys.items():
        failure_reason = (
            "Test mode - treated as flaky"
            if test_mode
            else get_job_failure_reason(hub_client, job_id, False)
        )

        # Job key format: {model_id}_{precision}_{path_name}-{device_name}_{component}
        job_without_device = job_key.split("-")[0] if "-" in job_key else job_key
        parts = job_without_device.split("_")

        # Model name is everything except the last 2 parts (precision and path)
        model_name = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]

        # Extract device name by matching against known devices
        if "-" in job_key:
            device_and_component = job_key.split("-")[1]
            device_name = "unknown"
            # Check if the full string matches a known device (no component)
            if device_and_component in known_device_names:
                device_name = device_and_component
            else:
                # Try to match known device names (longest first to handle overlapping prefixes, e.g. cs_8_elite vs cs_8_elite_gen_5)
                # Break on first longest match.
                for known_device in sorted(known_device_names, key=len, reverse=True):
                    if device_and_component.startswith(known_device + "_"):
                        device_name = known_device
                        break
        else:
            device_name = "unknown"

        flaky_jobs_info.flaky_jobs[job_key] = FlakyJobsYaml.FlakyJobsDetails(
            job_key=job_key,
            original_job_id=job_id,
            model=model_name,
            device=device_name,
            failure_reason=failure_reason or "Unknown failure",
        )

        # Update summary statistics
        flaky_jobs_info.total_flaky_jobs += 1

        # Count by failure reason
        reason = failure_reason or "Unknown failure"
        flaky_jobs_info.by_failure_reason[reason] = (
            flaky_jobs_info.by_failure_reason.get(reason, 0) + 1
        )

        # Count by device
        flaky_jobs_info.by_device[device_name] = (
            flaky_jobs_info.by_device.get(device_name, 0) + 1
        )

        # Count by model
        flaky_jobs_info.by_model[model_name] = (
            flaky_jobs_info.by_model.get(model_name, 0) + 1
        )

    # If no flaky failures found, exit early
    if not failed_job_keys:
        print("No flaky failures found, no changes made")
        pool.close()
        return

    # In dry run mode, just show what would be done
    if dry_run:
        elapsed_time = time.time() - start_time
        elapsed_minutes = elapsed_time / 60.0

        print(f"[DRY RUN] Found {len(failed_job_keys)} jobs with flaky failures")
        print(f"[DRY RUN] Changes would have been saved to: {inference_jobs_file}")

        # Show a summary of changes that would be made
        print("\n=== Changes that would be made to the YAML file ===")
        for job_key, job_id in failed_job_keys.items():
            print(f"  {job_key}: {job_id} -> [would be replaced with new job ID]")

        print(f"\n[DRY RUN] Total processing time: {elapsed_minutes:.2f} minutes")
        pool.close()
        return

    # Process flaky jobs
    print(
        f"\nSubmitting {len(failed_job_keys)} new inference jobs for the failed ones listed above"
    )

    changes_made = False

    # Process flaky jobs using the same pool
    print("Processing flaky jobs in parallel...")

    # Prepare arguments for parallel processing, needed cast to fix mypy
    args_list = cast(
        list,
        [(hub_client, job_key, job_id) for job_key, job_id in failed_job_keys.items()],
    )

    # Process jobs in parallel
    results = pool.starmap(process_flaky_job, [(x[0], x[1], x[2]) for x in args_list])
    pool.close()

    # Process results
    for job_key, new_job_id, _ in results:
        inference_yaml.mapping[job_key] = new_job_id
        # Update flaky jobs info with new job ID
        if job_key in flaky_jobs_info.flaky_jobs:
            flaky_jobs_info.flaky_jobs[job_key].new_job_id = new_job_id
        changes_made = True

    total_elapsed_time = time.time() - start_time
    total_elapsed_minutes = total_elapsed_time / 60.0
    print(f"Total processing time: {total_elapsed_minutes:.2f} minutes")

    # Save the updated YAML if changes were made
    if changes_made:
        print(f"Saving updated inference jobs YAML to: {inference_yaml.path}")
        inference_yaml.to_file()
        print("Inference jobs YAML updated with new job IDs")

    # Write flaky jobs info to output file if specified
    if flaky_jobs_output and flaky_jobs_info.flaky_jobs:
        flaky_jobs_output_path = Path(flaky_jobs_output)
        flaky_jobs_output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving flaky jobs info to: {flaky_jobs_output_path}")
        flaky_jobs_info.to_yaml(flaky_jobs_output_path)
        print(f"Saved {len(flaky_jobs_info.flaky_jobs)} flaky job entries")


if __name__ == "__main__":
    main()
