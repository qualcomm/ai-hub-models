# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""AWS Device Farm implementation of the DeviceFarm contract."""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import boto3
import requests
from qai_hub_models_cli._internal.aws import (
    github_oidc_available,
    oidc_refreshing_session,
)

from qai_hub_models.utils.devicefarm.devicefarm import (
    DeviceFarm,
    create_zip_from_entries,
)

POLL_INTERVAL = 15
# Run states before a phone is allocated. PREPARING/RUNNING already hold one.
_WAITING_RUN_STATES = frozenset(
    {"PENDING", "PENDING_CONCURRENCY", "PENDING_DEVICE", "PROCESSING", "SCHEDULING"}
)
# AWS Device Farm itself caps a single job at 150 minutes; this is a client
# cap on top of that, generous enough for the perf sweep and the 100-prompt
# accuracy eval.
DEFAULT_JOB_TIMEOUT = 7200
UPLOAD_PROCESSING_TIMEOUT = 300
ARTIFACT_LISTING_MAX_RETRIES = 5
ARTIFACT_LISTING_RETRY_DELAY = 5
# Retry budget for the S3 upload PUT. Observed failure mode: SSLEOFError
# ("EOF occurred in violation of protocol") during upload of the test bundle,
# a transient S3-side connection drop -- see run 35474812911. requests streams
# the file body, so a urllib3 Retry adapter can't safely retry (the file
# handle is already partially consumed); retry manually, reopening the file
# fresh each attempt instead.
# Delay sequence (base=3, doubling, capped at 120s): 3, 6, 12, 24, 48, 96,
# 120, 120, 120 -- ~9 minutes of cumulative backoff before giving up, enough
# to ride out a 5+ minute outage like run 35741077622/35819722217 with margin.
UPLOAD_PUT_MAX_RETRIES = 10
UPLOAD_PUT_RETRY_BACKOFF_BASE = 3
UPLOAD_PUT_RETRY_BACKOFF_CAP = 120
# (connect timeout, read timeout). requests applies the read timeout between
# individual socket reads/writes, not to the whole request, so a large but
# still-progressing upload won't spuriously time out -- only a stalled
# connection that stops moving bytes for this long will.
UPLOAD_PUT_TIMEOUT = (30, 60)
# Same (connect, read) semantics as UPLOAD_PUT_TIMEOUT, for artifact downloads.
DOWNLOAD_TIMEOUT = (30, 60)
# Presigned URLs carry their credentials in the query string. The test spec
# log is archived as a CI artifact, so strip them (e.g. from a failed curl).
_PRESIGNED_QUERY_RE = re.compile(rb"(https?://[^\s?'\"]+)\?[^\s'\"]+")
_RETRYABLE_HTTP_STATUS_CODES = (500, 502, 503, 504)


def _is_retryable_upload_error(err: Exception) -> bool:
    """Transient network/connection errors and 5xx are retryable; other
    HTTP errors (e.g. 403 on an expired upload URL) are not.
    """
    if isinstance(err, requests.exceptions.HTTPError):
        status = err.response.status_code if err.response is not None else None
        return status in _RETRYABLE_HTTP_STATUS_CODES
    return isinstance(
        err, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    )


def _put_file_with_retry(url: str, file_path: str) -> None:
    """PUT ``file_path`` to ``url``, retrying transient network/5xx failures."""
    for attempt in range(UPLOAD_PUT_MAX_RETRIES):
        try:
            with open(file_path, "rb") as f:
                resp = requests.put(
                    url,
                    data=f,
                    headers={"content-type": "application/octet-stream"},
                    timeout=UPLOAD_PUT_TIMEOUT,
                )
            resp.raise_for_status()
            return
        except (requests.exceptions.RequestException, OSError) as err:  # noqa: PERF203
            if not _is_retryable_upload_error(err) or (
                attempt == UPLOAD_PUT_MAX_RETRIES - 1
            ):
                raise
            delay = min(
                UPLOAD_PUT_RETRY_BACKOFF_BASE * (2**attempt),
                UPLOAD_PUT_RETRY_BACKOFF_CAP,
            )
            print(
                f"[AWS Device Farm] upload PUT failed ({type(err).__name__}); "
                f"attempt {attempt + 1}/{UPLOAD_PUT_MAX_RETRIES}, "
                f"retrying in {delay}s.",
                file=sys.stderr,
            )
            time.sleep(delay)


# Populated by the test spec's post_test phase (see aws_test_spec.yaml).
# AWS_logs (rather than QDC's on-device directory name) because QDC's
# server-side harvesting requires that literal name -- see
# devicefarm.device_logs_dir_name -- while AWS has no such requirement and
# does its own adb pull of a name it controls.
HOST_DEVICE_LOGS_SUBDIR = "AWS_logs"
# Archive name for the job's TESTSPEC_OUTPUT artifact (the full test spec
# console log, including pytest output), the AWS analogue of QDC's job logs.
TESTSPEC_LOG_NAME = "test_spec_output.txt"

_DEVICE_SCRIPTS_DIR = Path(__file__).parent / "device_scripts"
_PLACEHOLDER_APK = _DEVICE_SCRIPTS_DIR / "placeholder.apk"
_TEST_SPEC_YAML = _DEVICE_SCRIPTS_DIR / "aws_test_spec.yaml"

# Hub device name -> AWS Device Farm device ARN(s) for that chipset. Several
# ARNs can back one hub device name (e.g. different pre-flashed OS builds of
# the same phone); scheduling filters on all of them and Device Farm picks one.
HUB_DEVICE_TO_AWS_DEVICE_ARNS: dict[str, list[str]] = {
    "Samsung Galaxy S25": [
        "arn:aws:devicefarm:us-west-2::device:69977E819F314BD481F194580E8EFC16",
        "arn:aws:devicefarm:us-west-2::device:092D78E7B4694568812D5FF0711CD346",
    ],
    "Samsung Galaxy S26": [
        "arn:aws:devicefarm:us-west-2::device:C2F2BF5633994E75942BE2EB246C2709",
    ],
}


def get_aws_device_arns(hub_device_name: str) -> list[str]:
    arns = HUB_DEVICE_TO_AWS_DEVICE_ARNS.get(hub_device_name)
    if not arns:
        raise ValueError(
            f"No AWS Device Farm device ARNs configured for {hub_device_name!r}. "
            f"Known devices: {sorted(HUB_DEVICE_TO_AWS_DEVICE_ARNS)}"
        )
    return arns


@dataclass
class AwsDeviceFarmConfig:
    project_arn: str
    region: str = "us-west-2"
    # None lets boto3 fall back to its normal credential chain (e.g. an
    # AWS_PROFILE already set by the CI step, or an assumed role's env vars).
    profile_name: str | None = None
    # When set inside a GitHub Actions job with `id-token: write`, the role is
    # re-assumed via OIDC whenever its credentials near expiry (see _make_session).
    role_arn: str | None = None


def get_aws_devicefarm_config() -> AwsDeviceFarmConfig:
    """Read AWS Device Farm project/profile/region from the environment."""
    project_arn = os.environ.get("AWS_DEVICEFARM_PROJECT_ARN")
    if not project_arn:
        raise ValueError("AWS_DEVICEFARM_PROJECT_ARN is not set.")
    return AwsDeviceFarmConfig(
        project_arn=project_arn,
        region=os.environ.get("AWS_DEVICEFARM_REGION", "us-west-2"),
        profile_name=os.environ.get("AWS_DEVICEFARM_PROFILE"),
        role_arn=os.environ.get("AWS_DEVICEFARM_ROLE_ARN") or None,
    )


def _make_session(config: AwsDeviceFarmConfig) -> boto3.Session:
    """Build a boto3 session for Device Farm.

    A static profile expires mid-collect (ExpiredTokenException, run
    36655411185), so in CI use credentials that re-assume the role via OIDC.
    """
    if not (config.role_arn and github_oidc_available()):
        return boto3.Session(
            profile_name=config.profile_name, region_name=config.region
        )
    return oidc_refreshing_session(config.role_arn, config.region, "qaihm-devicefarm")


@dataclass
class AwsLogFile:
    """Duck-types the ``.filename`` attribute QDC job-log objects expose."""

    filename: str


class AwsDeviceFarm(DeviceFarm):
    """AWS Device Farm implementation of :class:`DeviceFarm`."""

    def __init__(self, config: AwsDeviceFarmConfig | None = None) -> None:
        self.config = config or get_aws_devicefarm_config()
        session = _make_session(self.config)
        self.client = session.client("devicefarm", region_name=self.config.region)
        # Populated on first get_job_log_files() call; download_job_log_files
        # reads from it, matching the (get_job_log_files -> download_*) call
        # order callers already use for QDC.
        self._extracted_log_dir: str | None = None

    # ---- upload -------------------------------------------------------

    def _upload_file(self, file_path: str, upload_type: str) -> str:
        """Create an upload slot, PUT the file, and poll until processed.

        Returns the upload's ARN.
        """
        name = os.path.basename(file_path)
        upload = self.client.create_upload(
            projectArn=self.config.project_arn, name=name, type=upload_type
        )["upload"]
        _put_file_with_retry(upload["url"], file_path)
        return self._wait_for_upload(upload["arn"])

    def _wait_for_upload(
        self, upload_arn: str, timeout: int = UPLOAD_PROCESSING_TIMEOUT
    ) -> str:
        elapsed = 0
        while elapsed < timeout:
            upload = self.client.get_upload(arn=upload_arn)["upload"]
            status = upload["status"]
            if status == "SUCCEEDED":
                return upload_arn
            if status == "FAILED":
                raise RuntimeError(
                    f"AWS Device Farm upload {upload_arn} failed: "
                    f"{upload.get('metadata')}"
                )
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        raise TimeoutError(
            f"Upload {upload_arn} did not finish processing within {timeout}s"
        )

    # ---- submit / poll --------------------------------------------------

    def submit_bundle(
        self,
        hub_device_name: str,
        entries: list[tuple[str, str]],
        entry_script: str | None,
        job_name: str,
        timeout: int = DEFAULT_JOB_TIMEOUT,
        job_timeout: int | None = None,
    ) -> str:
        """
        Zip ``entries``, upload the bundle to AWS Device Farm, and schedule a run for it.

        See :meth:`DeviceFarm.submit_bundle` for the shared contract. AWS
        specifics: AWS Device Farm LLM support is Android-only, so
        ``entry_script`` must be ``None`` -- unlike the QDC backend, there is
        no way to point AWS at a custom entry point; it always runs the
        fixed Appium test spec bundled with this backend. The zipped bundle
        is uploaded as an ``APPIUM_PYTHON_TEST_PACKAGE`` artifact alongside a
        placeholder app and the test spec (see :meth:`_submit_automated_job`),
        and the run is scheduled against every device ARN ``hub_device_name``
        resolves to (see :func:`get_aws_device_arns`).

        Parameters
        ----------
        hub_device_name
            QAI Hub device name; resolved to one or more AWS Device Farm
            device ARNs via :func:`get_aws_device_arns`.
        entries
            ``(source_path, arcname)`` pairs zipped into the uploaded test
            package.
        entry_script
            Must be ``None``; AWS Device Farm always runs the fixed Appium
            test spec, not a caller-supplied entry point.
        job_name
            Run name shown in AWS Device Farm; truncated to 256 characters.
        timeout
            Passed through to ``executionConfiguration.jobTimeoutMinutes``
            (converted to minutes, minimum 1) on the scheduled run. Unlike
            the QDC backend, this bounds job execution time, not a wait for a
            free device slot.
        job_timeout
            If set, overrides ``timeout`` as the run's execution limit.

        Returns
        -------
        job_id : str
            The AWS Device Farm run ARN (not a QDC-style numeric job id).
            Used as the "job id" everywhere else in the LLM perf-collection
            machinery (jobs_file, JobRecord, etc.).

        Raises
        ------
        ValueError
            If ``entry_script`` is not ``None``.
        """
        if entry_script is not None:
            raise ValueError(
                "AWS Device Farm LLM support is Android-only (entry_script must "
                f"be None); got entry_script={entry_script!r} for device "
                f"{hub_device_name!r}."
            )
        device_arns = get_aws_device_arns(hub_device_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test.zip")
            create_zip_from_entries(zip_path, entries)
            test_package_arn = self._upload_file(zip_path, "APPIUM_PYTHON_TEST_PACKAGE")
        return self._submit_automated_job(
            device_arns,
            test_package_arn,
            job_name=job_name,
            timeout=timeout if job_timeout is None else job_timeout,
        )

    def _submit_automated_job(
        self,
        device_arns: list[str],
        test_package_arn: str,
        job_name: str,
        timeout: int = DEFAULT_JOB_TIMEOUT,
    ) -> str:
        """Upload the placeholder app + test spec and schedule a run.

        Returns the run's ARN, used as the "job id" everywhere else in the
        LLM perf-collection machinery (jobs_file, JobRecord, etc.).
        """
        app_arn = self._upload_file(str(_PLACEHOLDER_APK), "ANDROID_APP")
        spec_arn = self._upload_file(str(_TEST_SPEC_YAML), "APPIUM_PYTHON_TEST_SPEC")
        run = self.client.schedule_run(
            projectArn=self.config.project_arn,
            appArn=app_arn,
            name=job_name[:256],
            deviceSelectionConfiguration={
                "filters": [
                    {"attribute": "ARN", "operator": "IN", "values": device_arns}
                ],
                "maxDevices": 1,
            },
            test={
                "type": "APPIUM_PYTHON",
                "testPackageArn": test_package_arn,
                "testSpecArn": spec_arn,
            },
            executionConfiguration={"jobTimeoutMinutes": max(1, timeout // 60)},
        )["run"]
        return run["arn"]

    def status(self, run_arn: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> str:
        """Poll until the run reaches a terminal (COMPLETED) state."""
        elapsed = 0
        while elapsed < timeout:
            run = self.client.get_run(arn=run_arn)["run"]
            if run["status"] == "COMPLETED":
                return str(run["status"])
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
        self.client.stop_run(arn=run_arn)
        raise TimeoutError(f"Run {run_arn} did not complete within {timeout}s")

    def is_done(self, run_arn: str) -> bool:
        return self.client.get_run(arn=run_arn)["run"]["status"] == "COMPLETED"

    def count_waiting_jobs(self, hub_device_name: str) -> int:
        """Runs in this project still waiting for a phone of ``hub_device_name``.

        Device Farm has no per-device queue API, so this pages ``list_runs``
        (newest first) and stops at the first page with no unfinished runs.
        """
        device_arns = set(get_aws_device_arns(hub_device_name))
        waiting = 0
        kwargs: dict[str, str] = {"arn": self.config.project_arn}
        while True:
            page = self.client.list_runs(**kwargs)
            runs = page["runs"]
            for run in runs:
                filters = run.get("deviceSelectionResult", {}).get("filters", [])
                run_arns = {v for f in filters for v in f.get("values", [])}
                if run["status"] in _WAITING_RUN_STATES and run_arns & device_arns:
                    waiting += 1
            token = page.get("nextToken")
            if not token or all(r["status"] == "COMPLETED" for r in runs):
                return waiting
            kwargs["nextToken"] = token

    def result(self, run_arn: str) -> str | None:
        """PASSED/FAILED/ERRORED/STOPPED/SKIPPED/WARNED, or None if absent."""
        run = self.client.get_run(arn=run_arn)["run"]
        return run.get("result")

    def is_successful(self, result: str | None) -> bool:
        return result == "PASSED"

    # ---- log retrieval --------------------------------------------------

    def _job_arn(self, run_arn: str) -> str:
        jobs = self.client.list_jobs(arn=run_arn)["jobs"]
        if not jobs:
            raise RuntimeError(f"No jobs found under run {run_arn}")
        return str(jobs[0]["arn"])

    def _download_artifacts(self, run_arn: str) -> tuple[bytes | None, bytes | None]:
        """Return ``(customer_artifacts_zip, testspec_output)``; either may be None."""
        job_arn = self._job_arn(run_arn)
        artifacts: list[dict] = []
        for attempt in range(ARTIFACT_LISTING_MAX_RETRIES):
            artifacts = self.client.list_artifacts(arn=job_arn, type="FILE")[
                "artifacts"
            ]
            if any(a.get("name") == "Customer Artifacts" for a in artifacts):
                break
            if attempt < ARTIFACT_LISTING_MAX_RETRIES - 1:
                time.sleep(ARTIFACT_LISTING_RETRY_DELAY)

        zip_bytes = None
        zip_url = next(
            (a["url"] for a in artifacts if a.get("name") == "Customer Artifacts"),
            None,
        )
        if zip_url is not None:
            resp = requests.get(zip_url, timeout=DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            zip_bytes = resp.content

        testspec = None
        testspec_url = next(
            (a["url"] for a in artifacts if a.get("type") == "TESTSPEC_OUTPUT"),
            None,
        )
        if testspec_url is not None:
            # Diagnostic only: losing it must not discard the device logs.
            try:
                resp = requests.get(testspec_url, timeout=DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                testspec = _PRESIGNED_QUERY_RE.sub(rb"\1?<redacted>", resp.content)
            except requests.RequestException as err:
                print(
                    f"[AwsDeviceFarm] could not fetch test spec output "
                    f"({type(err).__name__}); continuing without it.",
                    file=sys.stderr,
                )
        return zip_bytes, testspec

    def get_job_log_files(
        self, run_arn: str, wait_for_logs: bool = False
    ) -> list[AwsLogFile]:
        """Download+extract the run's Customer Artifacts zip once, then list
        every file under the pulled ``AWS_logs`` directory, plus the test spec
        output (the job's full host-side console log) as ``TESTSPEC_LOG_NAME``.

        One entry per on-device file (not one entry for the whole zip) so
        callers' per-filename matching (e.g. "genie" in filename,
        "profile*.json") works unmodified.
        """
        zip_bytes, testspec = self._download_artifacts(run_arn)
        extract_root = tempfile.mkdtemp(prefix="aws_devicefarm_artifacts_")
        if zip_bytes is not None:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(extract_root)
        # Device Farm writes the log dir under the literal env var name.
        device_logs_dir = os.path.join(
            extract_root,
            "Host_Machine_Files",
            "$DEVICEFARM_LOG_DIR",
            HOST_DEVICE_LOGS_SUBDIR,
        )
        self._extracted_log_dir = device_logs_dir
        files: list[AwsLogFile] = []
        for root, _, filenames in os.walk(device_logs_dir):
            for fn in filenames:
                rel = os.path.relpath(os.path.join(root, fn), device_logs_dir)
                files.append(AwsLogFile(filename=rel.replace(os.sep, "/")))
        # Callers waiting for parseable logs treat an empty listing as a
        # retryable empty-logs job; the test spec log alone must not mask that.
        if testspec is not None and (files or not wait_for_logs):
            os.makedirs(device_logs_dir, exist_ok=True)
            with open(os.path.join(device_logs_dir, TESTSPEC_LOG_NAME), "wb") as f:
                f.write(testspec)
            files.append(AwsLogFile(filename=TESTSPEC_LOG_NAME))
        return files

    def download_job_log_files(self, filename: str, target_path: str) -> None:
        """Zip-wrap the single already-extracted file at ``target_path``.

        Matches the shape callers expect from QDC's own per-file downloads:
        a single-entry zip containing just that file, which they unpack
        themselves. Raises if the file isn't present -- the base class's
        ``try_download_job_log_files`` turns that into a bool.
        """
        if self._extracted_log_dir is None:
            raise RuntimeError("get_job_log_files must be called before this method")
        abs_path = os.path.join(self._extracted_log_dir, filename)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(abs_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with zipfile.ZipFile(target_path, "w", zipfile.ZIP_STORED) as zf:
            zf.write(abs_path, arcname=os.path.basename(filename))
