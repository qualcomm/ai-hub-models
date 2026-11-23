# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from pprint import pformat

import pytest
import qai_hub.public_rest_api as hub_api
from qai_hub.hub import _global_client

from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import DisableWorkbenchJobTimeoutEnvvar
from qai_hub_models.scorecard.execution_helpers import (
    wait_for_prerequisite_job,
)
from qai_hub_models.scorecard.results.yaml import (
    CompileScorecardJobYaml,
    InferenceScorecardJobYaml,
    LinkScorecardJobYaml,
    QuantizeScorecardJobYaml,
)


def _check_single_job(
    name: str,
    job_id: str,
    max_workbench_job_duration_seconds: int | None,
    failed_jobs: dict[str, str],
    timeout_jobs: dict[str, str],
    lock: threading.Lock,
) -> None:
    """Check a single job's status, storing results in the shared dicts."""
    try:
        # We fetch the job_pb here (instead of using hub.get_job) so we don't have to fetch it again later when we check for timeouts.
        # get_job() is pretty slow. Removing the extra fetch saves us 20+ seconds (for context, checking all compile jobs takes about 50 seconds).
        job_pb = _global_client._api_call(hub_api.get_job, job_id=job_id)
        job = _global_client._make_job(job_pb)
        assert job is not None, f"Unable to retrieve job {job_id}"
        status = wait_for_prerequisite_job(job, max_workbench_job_duration_seconds)

        exceeded_timeout = False
        if status.success and max_workbench_job_duration_seconds is not None:
            # job completion_time is not exposed in the client, so use lower level APIs.
            specific_job_pb = job._extract_job_specific_pb(job_pb)
            if not specific_job_pb.HasField("completion_time"):
                # The original job_pb above was fetched before the job completed,
                # so it is lacking a set completion time.
                #
                # Fetch the job proto again so we have the completion time.
                specific_job_pb = job._extract_job_specific_pb(
                    job._owner._api_call(hub_api.get_job, job_id=job_id)
                )

            assert specific_job_pb.HasField("completion_time"), (
                "Unexpected workbench behavior: Completion time is missing in the job protobuf, but the job status is success."
            )
            exceeded_timeout = (
                specific_job_pb.completion_time.seconds
                - specific_job_pb.start_time.seconds
            ) > max_workbench_job_duration_seconds

        if not status.success:
            with lock:
                if not status.finished or (
                    status.failure
                    and status.message is not None
                    and "timed out" in status.message
                ):
                    timeout_jobs[name] = job.url
                else:
                    failed_jobs[name] = job.url
        elif exceeded_timeout:
            with lock:
                timeout_jobs[name] = job.url
    except Exception as exc:
        with lock:
            failed_jobs[name] = f"<exception: {exc}>"


_FAILED_JOBS_JSON = "failed-workbench-jobs.json"


def _write_failed_jobs_json(
    job_type: str,
    failed_jobs: dict[str, str],
    timeout_jobs: dict[str, str],
) -> None:
    """Append failed/timeout job URLs to a JSON file in the test-results directory."""
    junit_xml_path = os.environ.get("QAIHM_JUNIT_XML_PATH")
    if not junit_xml_path:
        return
    output_path = Path(junit_xml_path).parent / _FAILED_JOBS_JSON

    # Load existing entries (other job types may have already written)
    existing: dict[str, str] = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text())

    existing.update({f"{job_type}/{name}": url for name, url in failed_jobs.items()})
    existing.update(
        {f"{job_type}/{name} (timeout)": url for name, url in timeout_jobs.items()}
    )
    output_path.write_text(json.dumps(existing, indent=2))


def _verify_jobs_successful(job_ids: dict[str, str], job_type: str) -> None:
    """
    Verifies that the jobs with the given job ids all succeeded.
    If any jobs fail, raises a ValueError with the failed job urls.

    Uses a thread pool to check results in parallel.
    """
    failed_jobs: dict[str, str] = {}
    timeout_jobs: dict[str, str] = {}
    lock = threading.Lock()

    max_workbench_job_duration_minutes = (
        DisableWorkbenchJobTimeoutEnvvar.max_workbench_job_duration_minutes()
    )
    thread_timeout = (
        (max_workbench_job_duration_minutes + 5) * 60
        if max_workbench_job_duration_minutes is not None
        else None
    )

    max_workbench_job_duration_seconds = (
        (max_workbench_job_duration_minutes * 60)
        if max_workbench_job_duration_minutes is not None
        else None
    )
    with ThreadPoolExecutor(max_workers=min(64, len(job_ids))) as pool:
        futures = [
            pool.submit(
                _check_single_job,
                name,
                job_id,
                max_workbench_job_duration_seconds,
                failed_jobs,
                timeout_jobs,
                lock,
            )
            for name, job_id in job_ids.items()
        ]
        for f in as_completed(futures, timeout=thread_timeout):
            f.result()

    # Write failed/timeout job URLs to JSON for structured downstream consumption
    # (e.g. nightly failure notifications). Written before raising so the file
    # is available even when the test fails.
    _write_failed_jobs_json(job_type, failed_jobs, timeout_jobs)

    error_strs = []
    if failed_jobs:
        error_strs.append(
            f"The following {job_type} jobs failed:\n{pformat(failed_jobs)}"
        )
    if timeout_jobs:
        errmsg = f"The following {job_type} jobs timed out"
        if max_workbench_job_duration_minutes is not None:
            errmsg += f" or took longer than the maximum allowed runtime of {max_workbench_job_duration_minutes} minutes to complete"
        error_strs.append(f"{errmsg}:\n{pformat(timeout_jobs)}")
    if len(error_strs) > 0:
        raise ValueError("\n".join(error_strs))


"""
When testing compilation in CI, synchronously waiting for each job to
finish is too slow. Instead, job ids are written to a file upon submission,
and success is validated all at once in the end using these tests.
"""


@pytest.mark.skipif(
    not ScorecardArtifact.QUANTIZE_YAML.exists(),
    reason="No quantize jobs file found",
)
def test_quantize_jobs_success() -> None:
    job_ids = QuantizeScorecardJobYaml.from_test_artifacts().mapping
    _verify_jobs_successful(job_ids, "quantize")


@pytest.mark.skipif(
    not ScorecardArtifact.COMPILE_YAML.exists(),
    reason="No compile jobs file found",
)
def test_compile_jobs_success() -> None:
    job_ids = CompileScorecardJobYaml.from_test_artifacts().mapping
    _verify_jobs_successful(job_ids, "compile")


@pytest.mark.skipif(
    not ScorecardArtifact.LINK_YAML.exists(),
    reason="No link jobs file found",
)
def test_link_jobs_success() -> None:
    job_ids = LinkScorecardJobYaml.from_test_artifacts().mapping
    _verify_jobs_successful(job_ids, "link")


@pytest.mark.skipif(
    not ScorecardArtifact.INFERENCE_YAML.exists(), reason="No inference jobs found"
)
def test_inference_jobs_success() -> None:
    job_ids = InferenceScorecardJobYaml.from_test_artifacts().mapping
    _verify_jobs_successful(job_ids, "inference")
