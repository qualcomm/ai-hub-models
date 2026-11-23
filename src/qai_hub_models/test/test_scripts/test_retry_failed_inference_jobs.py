# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Hub Client Methods: ['config', 'get_dataset', 'get_datasets', 'get_device_attributes', 'get_devices', 'get_frameworks', 'get_job', 'get_job_summaries', 'get_jobs', 'get_model', 'get_models', 'set_verbose', 'submit_compile_and_profile_jobs', 'submit_compile_job', 'submit_inference_job', 'submit_link_job', 'submit_profile_job', 'submit_quantize_job', 'upload_dataset', 'upload_model', 'verbose']
Job Methods: ['date', 'device', 'disable_sharing', 'download_profile', 'download_results', 'get_sharing', 'get_status', 'hub_version', 'job_id', 'job_type', 'model', 'modify_sharing', 'name', 'options', 'set_name', 'shapes', 'url', 'verbose', 'wait']
Job Type: InferenceJob
Job Status Methods: ['State', 'all_running_states', 'code', 'failure', 'finished', 'message', 'pending', 'running', 'state', 'success', 'symbol']
Job Status Attributes: {'state': <State.SUCCESS: 10>, 'message': ''}

Tried to Mock the followin Hub Obj (attributes and methods) accurately...
"""

import sys
from typing import Any
from unittest.mock import patch

from qai_hub.client import Client as HubClient

from qai_hub_models.scripts.retry_failed_inference_jobs import (
    FLAKY_FAILURES,
    check_job_for_flaky_failure,
    get_job_failure_reason,
    is_flaky_failure,
    parse_args,
    process_flaky_job,
)


# Mock classes for testing
class MockJobStatus:
    """Mock job status for testing."""

    def __init__(self, failure: bool, message: str) -> None:
        self.failure = failure
        self.message = message

    def success(self) -> bool:
        return not self.failure

    def finished(self) -> bool:
        return True  # Assuming the job is finished


class MockInferenceJob:
    """Mock inference job for testing."""

    def __init__(
        self,
        job_id: str,
        status: MockJobStatus,
        model: str = "test_model",
        device: str = "test_device",
        options: str = "",
        name: str | None = None,
    ) -> None:
        self.job_id = job_id
        self.status = status
        self.model = model
        self.device = device
        self.options = options
        self.name = name or "test_job"
        self.job_type = "inference"

    def get_status(self) -> MockJobStatus:
        return self.status


class MockHubClient(HubClient):
    """Mock hub client for testing."""

    def __init__(self, jobs: dict[str, MockInferenceJob]) -> None:
        self.jobs = jobs
        self._verbose = False

    def get_job(self, job_id: str) -> MockInferenceJob | None:  # type: ignore[override]
        return self.jobs.get(job_id)

    def submit_inference_job(  # type: ignore[override]
        self,
        model: str,
        device: str,
        inputs: None = None,
        name: str | None = None,
        options: str = "",
        **kwargs: Any,
    ) -> MockInferenceJob:
        job_id = f"new_{len(self.jobs)}"
        job = MockInferenceJob(
            job_id, MockJobStatus(False, ""), model, device, options, name
        )
        self.jobs[job_id] = job
        return job


def test_is_flaky_failure() -> None:
    # Test with known flaky failures
    for flaky_reason in FLAKY_FAILURES:
        # Remove the "Failed (" prefix and ")" suffix
        # Handle cases where the format might be slightly different
        if flaky_reason.startswith("Failed (") and flaky_reason.endswith(")"):
            reason = flaky_reason[8:-1]
        else:
            # Fallback for malformed entries - just remove "Failed (" if present
            reason = flaky_reason.replace("Failed (", "").rstrip(")")

        assert is_flaky_failure(reason), f"Expected {reason} to be detected as flaky"

    # Test with non-flaky failures
    assert not is_flaky_failure("Some other error")

    # Test with None
    assert not is_flaky_failure(None)

    # Test with case variations
    assert is_flaky_failure("job timed out after 8h")  # lowercase
    assert is_flaky_failure("JOB TIMED OUT AFTER 8H")  # uppercase
    assert is_flaky_failure("Error uploading to QDC: status code=503.")
    assert is_flaky_failure("Error uploading artifact to QDC. Response code=500")
    assert is_flaky_failure("upload to device farm failed")


def test_get_job_failure_reason() -> None:
    # Create test job statuses
    failed_status = MockJobStatus(True, "Job timed out after 8h")
    success_status = MockJobStatus(False, "")

    # Create test jobs
    failed_job = MockInferenceJob("failed_job_id", failed_status)
    success_job = MockInferenceJob("success_job_id", success_status)

    # Create a hub client with our test jobs
    hub_client = MockHubClient(
        {"failed_job_id": failed_job, "success_job_id": success_job}
    )

    # Test with a failed job
    assert (
        get_job_failure_reason(hub_client, "failed_job_id") == "Job timed out after 8h"
    )

    # Test with a successful job
    assert get_job_failure_reason(hub_client, "success_job_id") is None


def test_check_job_for_flaky_failure() -> None:
    # Create test job statuses
    failed_status = MockJobStatus(
        True, "Failed to profile the model: unexpected device error"
    )
    non_flaky_status = MockJobStatus(True, "Some other error")
    success_status = MockJobStatus(False, "")

    # Create test jobs
    flaky_job = MockInferenceJob("flaky_job_id", failed_status)
    non_flaky_job = MockInferenceJob("non_flaky_job_id", non_flaky_status)
    success_job = MockInferenceJob("success_job_id", success_status)

    # Create a hub client with our test jobs
    hub_client = MockHubClient(
        {
            "flaky_job_id": flaky_job,
            "non_flaky_job_id": non_flaky_job,
            "success_job_id": success_job,
        }
    )

    # Test with a flaky failure
    job_key, job_id, has_flaky_failure = check_job_for_flaky_failure(
        hub_client, "key1", "flaky_job_id"
    )
    assert job_key == "key1"
    assert job_id == "flaky_job_id"
    assert has_flaky_failure

    # Test with a non-flaky failure
    job_key, job_id, has_flaky_failure = check_job_for_flaky_failure(
        hub_client, "key2", "non_flaky_job_id"
    )
    assert job_key == "key2"
    assert job_id == "non_flaky_job_id"
    assert not has_flaky_failure

    # Test with a successful job
    job_key, job_id, has_flaky_failure = check_job_for_flaky_failure(
        hub_client, "key3", "success_job_id"
    )
    assert job_key == "key3"
    assert job_id == "success_job_id"
    assert not has_flaky_failure


def test_check_job_for_flaky_failure_test_mode() -> None:
    """Test that test_mode=True treats all jobs as flaky regardless of failure status."""
    # Create test job statuses
    failed_status = MockJobStatus(
        True, "Failed to profile the model: unexpected device error"
    )
    non_flaky_status = MockJobStatus(True, "Some other error")
    success_status = MockJobStatus(False, "")

    # Create test jobs
    flaky_job = MockInferenceJob("flaky_job_id", failed_status)
    non_flaky_job = MockInferenceJob("non_flaky_job_id", non_flaky_status)
    success_job = MockInferenceJob("success_job_id", success_status)

    # Create a hub client with our test jobs
    hub_client = MockHubClient(
        {
            "flaky_job_id": flaky_job,
            "non_flaky_job_id": non_flaky_job,
            "success_job_id": success_job,
        }
    )

    # Test with a flaky failure in test mode
    job_key, job_id, has_flaky_failure = check_job_for_flaky_failure(
        hub_client, "key1", "flaky_job_id", test_mode=True
    )
    assert job_key == "key1"
    assert job_id == "flaky_job_id"
    assert has_flaky_failure

    # Test with a non-flaky failure in test mode
    job_key, job_id, has_flaky_failure = check_job_for_flaky_failure(
        hub_client, "key2", "non_flaky_job_id", test_mode=True
    )
    assert job_key == "key2"
    assert job_id == "non_flaky_job_id"
    assert has_flaky_failure  # Should be True in test mode

    # Test with a successful job in test mode
    job_key, job_id, has_flaky_failure = check_job_for_flaky_failure(
        hub_client, "key3", "success_job_id", test_mode=True
    )
    assert job_key == "key3"
    assert job_id == "success_job_id"
    assert has_flaky_failure  # Should be True in test mode


def test_process_flaky_job_core_logic() -> None:
    """The retried job must carry over the original's model, device, and options."""
    job_status = MockJobStatus(True, "Job timed out after 8h")
    job_to_clone = MockInferenceJob(
        "job_to_clone",
        job_status,
        model="test_model",
        device="test_device",
        options="--my_option value1",
        name="original_job",
    )

    hub_client = MockHubClient({"job_to_clone": job_to_clone})

    # The mock job is a stand-in, not a real hub.InferenceJob, so relax the
    # isinstance guard the function uses to narrow the job type.
    with patch(
        "qai_hub_models.scripts.retry_failed_inference_jobs.hub.InferenceJob",
        MockInferenceJob,
    ):
        job_key, new_job_id, success = process_flaky_job(
            hub_client, "key1", "job_to_clone"
        )

    assert (job_key, success) == ("key1", True)
    assert len(hub_client.jobs) == 2

    new_job = hub_client.jobs[new_job_id]
    assert new_job.model == job_to_clone.model
    assert new_job.device == job_to_clone.device
    # Options must land in `options`, not be swallowed by the `name` parameter.
    assert new_job.options == job_to_clone.options
    assert new_job.name == job_to_clone.name


def test_parse_args_test_mode() -> None:
    """Test that the --test-mode flag is properly recognized."""
    # Test without --test-mode
    with patch.object(
        sys, "argv", ["retry_failed_inference_jobs.py", "--artifacts-dir", "test_dir"]
    ):
        args = parse_args()
        assert not args.test_mode

    # Test with --test-mode
    with patch.object(
        sys,
        "argv",
        [
            "retry_failed_inference_jobs.py",
            "--artifacts-dir",
            "test_dir",
            "--test-mode",
        ],
    ):
        args = parse_args()
        assert args.test_mode

    # Test with other flags
    with patch.object(
        sys,
        "argv",
        [
            "retry_failed_inference_jobs.py",
            "--artifacts-dir",
            "test_dir",
            "--dry-run",
            "--test-mode",
        ],
    ):
        args = parse_args()
        assert args.dry_run
        assert args.test_mode

    # Test with --collect-failure-reasons
    with patch.object(
        sys,
        "argv",
        [
            "retry_failed_inference_jobs.py",
            "--artifacts-dir",
            "test_dir",
            "--collect-failure-reasons",
            "--test-mode",
        ],
    ):
        args = parse_args()
        assert args.collect_failure_reasons
        assert args.test_mode
