# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""QDC (Qualcomm Device Cloud) implementation of the DeviceFarm contract."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

import httpx
from qualcomm_device_cloud_sdk.api import qdc_api
from qualcomm_device_cloud_sdk.models import (
    ArtifactType,
    JobMode,
    JobResult,
    JobState,
    JobSubmissionParameter,
    JobType,
    LogUploadStatus,
    TestFramework,
)
from qualcomm_device_cloud_sdk.models import (
    JobType0 as Job,
)

from qai_hub_models.scorecard.device import (
    ScorecardDevice,
    cs_ventuno_q,
    cs_x_elite,
)
from qai_hub_models.utils.devicefarm.devicefarm import (
    DeviceFarm,
    HubDevicePlatform,
    JobOutcome,
    create_zip_from_entries,
)

# States in which a job is still in progress (not yet terminal)
_RUNNING_STATES = {
    JobState.DISPATCHED,
    JobState.RUNNING,
    JobState.SETUP,
    JobState.SUBMITTED,
}
# Log-upload states that are terminal (no further polling will change them)
_TERMINAL_LOG_UPLOAD_STATES = {
    LogUploadStatus.COMPLETED,
    LogUploadStatus.FAILED,
}
# NOLOGS is also what QDC reports right after a job completes, before the upload
# starts, so it only counts as terminal once it has persisted this long (seconds).
NOLOGS_GRACE_SECONDS = 180

# Map from hub device names to QDC target device names
HUB_DEVICE_TO_QDC_DEVICE_MAP = {
    "Snapdragon X Elite CRD": "SC8380XP",
    "Snapdragon X Plus 8-Core CRD": "X1P42100",
    "Snapdragon X2 Elite CRD": "SC8480XP",
    "Samsung Galaxy S25": "SM8750",
    "Samsung Galaxy S26": "SM8850",
    "SA8295P ADP": "SA8295P",
    "SA7255P ADP": "SA7255P",
    "SA8775P ADP": "QAM8775P",
    "Dragonwing IQ-9075 EVK": "QCS9075M",
    "Arduino VENTUNO Q": "QCS8275_Arduino",
}

# The SDK builds its httpx client with timeout=None, i.e. wait forever; a stalled
# download would then block until the job timeout. read/write are per-chunk and
# generous because log archives reach ~90MB (screen recordings).
QDC_HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=60.0)
JOBS_LIST_PAGE_SIZE = 20
# Default timeout for job status polling (in seconds)
DEFAULT_JOB_TIMEOUT = 21600  # 6 hours

# Log-upload poll budget when salvaging logs from an already-failed job. Short on
# purpose: the job is lost either way, so waiting out DEFAULT_JOB_TIMEOUT is wasted.
FAILED_JOB_LOG_TIMEOUT = 300  # 5 minutes
# Polling interval for job status checks (in seconds)
POLL_INTERVAL = 30
# Backoff schedule for retrying a *failed* QDC call (distinct from the steady
# POLL_INTERVAL cadence used while a job is legitimately still running). Grows
# exponentially from RETRY_BACKOFF_BASE, doubling each attempt, capped at
# RETRY_BACKOFF_MAX. Keeps us from hammering a rate-limited (429) or overloaded
# (5xx) endpoint on tight upload/download loops that have no outer throttle.
RETRY_BACKOFF_BASE = 5
RETRY_BACKOFF_MAX = 300
# Default seconds QDC lets a submitted job run on the device.
QDC_DEFAULT_JOB_RUN_TIMEOUT = 600
# QDC submission fail if name exceeds this
QDC_JOB_NAME_LIMIT = 32
# Number of consecutive errors tolerated when polling status. With BASE=5 and
# MAX=300, the schedule (5, 10, 20, 40, 80, 160, 300, 300, 300, 300) sums to
# ~30 minutes — long enough to absorb a ~30-min QDC API outage while still
# probing quickly during the first few seconds.
STATUS_POLL_MAX_RETRIES = 10
# Retry budget when the SDK discarded the status code (_opaque_sdk_parse_error).
# Shorter than STATUS_POLL_MAX_RETRIES: a transient 502 is indistinguishable from
# a permanent 404, so (5, 10, 20, 40) absorbs a blip without a 30-min burn.
OPAQUE_ERROR_MAX_RETRIES = 5
# Retry budget for submit_job. Larger than STATUS_POLL_MAX_RETRIES because a
# whole batch is now submitted at once, so QDC's per-user pending-job cap (a
# retryable 400) is the only thing throttling us, and draining it waits on real
# device runs rather than an API blip. With MAX=300 this retries for ~4 hours.
SUBMIT_MAX_RETRIES = 50
# Log files dropped from every listing: QDC attaches a ~90MB screen recording to
# each job, which no metric/eval parser reads but dominates a collection's bytes.
_UNPARSED_LOG_SUFFIXES = (".mp4",)
# Number of times to re-list job log files when the listing returns empty.
# QDC exposes log-upload-status and the file listing as two independently
# eventually-consistent signals: get_job_log_upload_status can report
# "Completed" seconds before get_job_log_files has indexed the uploaded files,
# so a successful job can momentarily list zero files (observed in the GPU
# nightly, breaking metric extraction for a job that actually succeeded). With
# BASE=5/MAX=300 the backoff schedule (5, 10, 20, 40, 80, 160, 300, 300, 300)
# sums to ~20 minutes.
# This is a client-side workaround; QDC-5565 tracks fixing it server-side so
# the file listing is consistent once log upload reports complete.
# https://jira-dc.qualcomm.com/jira/browse/QDC-5565
LOG_LISTING_MAX_RETRIES = 10
# HTTP status codes that the QDC SDK can surface transiently on status polling.
# The SDK raises a bare Exception with the code embedded in the message (e.g.
# "failed with status code 403 and message: Invalid Credentials"), so we match
# on the message. 401/403 have been observed intermittently on otherwise-valid
# credentials (e.g. a 401 on get_job_log_files minutes after the same client's
# submit/status/result/log_upload_status calls all succeeded, on a job that ran
# fine on-device — a transient token-refresh blip, not a real auth failure);
# 429/5xx are the usual rate-limit / server-side blips. 400 is NOT included
# globally: it is a permanent client error for every other QDC call (malformed
# job ID, bad params), and is only retryable for submit_job (per-user
# pending-job cap rejection) — that callsite passes it explicitly via
# ``extra_retryable_codes``.
_RETRYABLE_STATUS_CODES = (401, 403, 429, 500, 502, 503, 504)

# Devices with a dedicated (non-shared) QDC pool, which allow more concurrent
# jobs and require a separate API key (QDC_PRIVATE_API_KEY).
DEDICATED_POOL_DEVICES: frozenset[ScorecardDevice] = frozenset(
    {cs_x_elite, cs_ventuno_q}
)

# Return type for the generic retry wrapper.
CallableRetT = TypeVar("CallableRetT")


def get_qdc_api_token(device: ScorecardDevice) -> str:
    """QDC_PRIVATE_API_KEY for the dedicated pool, QDC_API_TOKEN otherwise."""
    if device in DEDICATED_POOL_DEVICES:
        token = os.environ.get("QDC_PRIVATE_API_KEY")
        if not token:
            raise ValueError(
                f"QDC_PRIVATE_API_KEY is not set; required for {device.name} "
                "(dedicated QDC pool)."
            )
        return token
    token = os.environ.get("QDC_API_TOKEN")
    if not token:
        raise ValueError("QDC_API_TOKEN is not set.")
    return token


def _matched_retryable_status_code(
    err: Exception, extra_retryable_codes: tuple[int, ...] = ()
) -> int | None:
    """Return the retryable HTTP status code embedded in err, else None.

    The QDC SDK raises a bare ``Exception`` whose message embeds the code (e.g.
    "failed with status code 403 and message: Invalid Credentials"). We return
    only the matched code so callers can log it WITHOUT echoing the rest of the
    message, which may contain server-reflected secrets or credential fragments.

    ``extra_retryable_codes`` is matched in addition to the global set; use it
    for codes that are only retryable at a specific callsite.
    """
    message = str(err)
    for code in _RETRYABLE_STATUS_CODES + extra_retryable_codes:
        if f"status code {code}" in message:
            return code
    return None


# Note: We only have to do this hack because the QDC API re-throws as bare
# exceptions (still true as of SDK 0.4.1 -- ``qdc_api.try_call`` does
# ``raise Exception(msg) from e``). We have asked them not to do this.
# https://jira-dc.qualcomm.com/jira/browse/QDC-5475
def _unwrap_causes(err: BaseException) -> Iterator[BaseException]:
    """Yield ``err`` and every exception in its __cause__/__context__ chain."""
    seen: set[int] = set()
    cur: BaseException | None = err
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def _transient_network_error_name(err: Exception) -> str | None:
    """Return the underlying transient-network error's type name, else None.

    The QDC SDK's ``try_call`` flattens the real failure into a bare
    ``Exception`` (``raise Exception(msg) from e``), so the network type is only
    visible by walking the cause chain. DNS failures, for instance, surface as
    ``socket.gaierror`` ("Temporary failure in name resolution") — an
    ``OSError`` subclass — wrapped under httpx/httpcore ConnectError. We return
    only the type name so callers can log it WITHOUT echoing the message, which
    may carry server-reflected secrets or credential fragments.

    httpx.TransportError and its subclasses (ConnectError, ReadError, *Timeout,
    RemoteProtocolError, etc.) are all transport-layer errors safe to retry.
    HTTPStatusError is a sibling, not a TransportError — non-retryable status
    codes are still filtered by _matched_retryable_status_code.
    """
    for cause in _unwrap_causes(err):
        if isinstance(cause, (OSError, ConnectionError, TimeoutError)):
            return type(cause).__name__
        if isinstance(cause, httpx.TransportError):
            return type(cause).__name__
    return None


def _opaque_sdk_parse_error(err: Exception) -> str | None:
    """Return the decode error's type name when the SDK lost the status code.

    ``try_call`` builds its error message by ``json.loads``-ing the body of any
    non-200, outside its own try/except, so a gateway answering with an empty or
    HTML body raises a bare ``JSONDecodeError``/``UnicodeDecodeError`` carrying
    neither the status code nor a network cause. That decode failure is then the
    only evidence the call failed server-side (observed on ``/jobs/downloadLogs``,
    which stalled ~60s then answered non-JSON). Everything ``_call_with_retry``
    wraps is idempotent, so retrying an unattributable failure is safe.

    We return only the type name, matching the redaction policy of the siblings.
    """
    for cause in _unwrap_causes(err):
        if isinstance(cause, (json.JSONDecodeError, UnicodeDecodeError)):
            return type(cause).__name__
    return None


def _backoff_seconds(attempt: int) -> int:
    """Exponential backoff (capped) for the given zero-based retry attempt."""
    return min(RETRY_BACKOFF_BASE * (2**attempt), RETRY_BACKOFF_MAX)


def _call_with_retry(
    func: Callable[[], CallableRetT],
    description: str,
    extra_retryable_codes: tuple[int, ...] = (),
    max_retries: int = STATUS_POLL_MAX_RETRIES,
) -> CallableRetT:
    """Call ``func()``, retrying through transient QDC errors.

    Covers transient network blips (DNS resolution failures, dropped
    connections), transient HTTP status errors the QDC SDK raises as a bare
    Exception (the global ``_RETRYABLE_STATUS_CODES`` plus any callsite-specific
    codes passed via ``extra_retryable_codes``), and failures whose status code
    the SDK discarded while decoding the error body (see
    ``_opaque_sdk_parse_error``, held to the shorter
    ``OPAQUE_ERROR_MAX_RETRIES``). A genuinely fatal error still surfaces once
    the budget is spent. Delays between attempts use capped exponential backoff
    (see ``_backoff_seconds``) so we don't hammer a rate-limited or overloaded
    endpoint.

    Used to wrap QDC SDK calls that talk to the network and are safe to repeat:
    status polling (``get_jobs_job_id``), log retrieval/download
    (``get_jobs_presigned_download_logs``), artifact upload
    (``upload_file_v2``, whose only commit point is the final ``end_upload_v2``
    and which returns the uuid of the successful attempt), and job submission
    (``submit_job``).

    Parameters
    ----------
    func
        Zero-argument callable performing the QDC SDK call.
    description
        Short human-readable label for the call, used in retry log lines.
    extra_retryable_codes
        Additional HTTP status codes that are retryable for this callsite
        only (matched in addition to the global ``_RETRYABLE_STATUS_CODES``).
    max_retries
        Attempt budget for retryable failures. Defaults to
        ``STATUS_POLL_MAX_RETRIES``; ``submit_job`` passes the longer
        ``SUBMIT_MAX_RETRIES``.

    Returns
    -------
    CallableRetT
        The return value of ``func()`` on the first successful attempt.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as err:  # noqa: PERF203
            # The QDC SDK flattens the real failure into a bare Exception, so we
            # must inspect both the cause chain (transient network errors like a
            # DNS gaierror) and the message (embedded retryable status codes).
            net_err = _transient_network_error_name(err)
            code = _matched_retryable_status_code(err, extra_retryable_codes)
            parse_err = (
                _opaque_sdk_parse_error(err)
                if net_err is None and code is None
                else None
            )
            budget = (
                min(OPAQUE_ERROR_MAX_RETRIES, max_retries)
                if parse_err is not None
                else max_retries
            )
            if (net_err is None and code is None and parse_err is None) or (
                attempt == budget - 1
            ):
                raise
            delay = _backoff_seconds(attempt)
            # Log only the matched type/status code, never the raw message,
            # which the SDK may populate with credential fragments or secrets.
            if net_err is not None:
                reason = f"transient network error ({net_err})"
            elif code is not None:
                reason = f"transient status code {code}"
            else:
                reason = f"undecodable error body ({parse_err}), status unknown"
            print(
                f"[QDC retry] {description} failed with {reason}; attempt "
                f"{attempt + 1}/{budget}, retrying in {delay}s."
            )
            time.sleep(delay)
    raise AssertionError("unreachable")  # loop either returns or raises


class QDCDevice(HubDevicePlatform):
    """Extends :class:`HubDevicePlatform` with QDC-specific device properties."""

    @property
    def qdc_name(self) -> str:
        """QDC target device name corresponding to the hub device name."""
        return HUB_DEVICE_TO_QDC_DEVICE_MAP[self.device.name]

    @property
    def test_framework(self) -> TestFramework:
        """QDC test framework appropriate for this device's platform."""
        if self.is_windows_platform:
            return TestFramework.POWERSHELL
        if self.is_iot_platform:
            return TestFramework.BASH
        return TestFramework.APPIUM


class QDCDeviceFarm(DeviceFarm):
    """QDC (Qualcomm Device Cloud) implementation of :class:`DeviceFarm`."""

    def __init__(
        self,
        *,
        api_key: str,
        app_name_header: str = "QDCDeviceFarmJobApp",
    ) -> None:
        """
        Parameters
        ----------
        api_key
            API key for QDC authentication.
        app_name_header
            Application name header for QDC API client.
        """
        self.client = qdc_api.get_public_api_client_using_api_key(
            api_key_header=api_key,
            app_name_header=app_name_header,
            on_behalf_of_header="ai_hub_models",
            client_type_header="Python",
        ).with_timeout(QDC_HTTP_TIMEOUT)

    def get_job(self, job_id: str) -> Job:
        """Fetch full job details from QDC, retrying transient errors.

        Parameters
        ----------
        job_id
            ID of the job to retrieve.

        Returns
        -------
        job : Job
            Job object returned by ``get_job_by_id``.

        Raises
        ------
        ValueError
            If QDC returns no job for ``job_id``.
        """
        job = _call_with_retry(
            lambda: qdc_api.get_job_by_id(self.client, job_id),
            f"get_job_by_id({job_id})",
        )
        if job is None:
            raise ValueError(f"Failure in `get_job_by_id`. No job found for {job_id}.")
        return job

    def status(self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> str:
        """
        Poll and return the terminal status for a job (e.g., Completed/Canceled).

        Parameters
        ----------
        job_id
            ID of the job to monitor.
        timeout
            Maximum time to wait for job completion in seconds.
            Defaults to DEFAULT_JOB_TIMEOUT (6 hours).

        Returns
        -------
        job_status : str
            Final status of the job.

        Raises
        ------
        TimeoutError
            If job does not complete within the timeout period.
        """
        job_status: JobState | None = None
        elapsed = 0
        while elapsed < timeout:
            job_status = _call_with_retry(
                lambda: qdc_api.get_job_status(self.client, job_id),
                f"get_job_status({job_id})",
            )
            if job_status not in _RUNNING_STATES:
                time.sleep(POLL_INTERVAL)
                return job_status.value
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

        job_status = _call_with_retry(
            lambda: qdc_api.get_job_status(self.client, job_id),
            f"get_job_status({job_id})",
        )
        if job_status not in _RUNNING_STATES:
            return job_status.value
        qdc_api.abort_job(self.client, job_id)
        raise TimeoutError(
            f"Job {job_id} did not complete within {timeout} seconds. "
            f"Last status: {job_status}"
        )

    def is_done(self, job_id: str) -> bool:
        job_status = _call_with_retry(
            lambda: qdc_api.get_job_status(self.client, job_id),
            f"get_job_status({job_id})",
        )
        return job_status not in _RUNNING_STATES

    def _unfinished_jobs(self) -> list[Job]:
        # Newest first; the account's unfinished jobs always fit in one page.
        page = _call_with_retry(
            lambda: qdc_api.get_jobs_list(self.client, 0, JOBS_LIST_PAGE_SIZE),
            "get_jobs_list",
        )
        if page is None:
            raise ValueError("Failure in `get_jobs_list`.")
        return [j for j in (page.data or []) if j and j.state in _RUNNING_STATES]

    def count_waiting_jobs(self, hub_device_name: str) -> int:
        chipset = HUB_DEVICE_TO_QDC_DEVICE_MAP[hub_device_name]
        return sum(
            1
            for j in self._unfinished_jobs()
            if j.state == JobState.SUBMITTED
            and any(t and t.chipset == chipset for t in (j.targets or []))
        )

    def count_unfinished_jobs(self) -> int:
        return len(self._unfinished_jobs())

    def result(self, job_id: str) -> str | None:
        """Return the terminal result of a job (e.g. "Successful"/"Unsuccessful").

        ``status()`` reports only the lifecycle ``state`` ("Completed"); a job can
        reach "Completed" yet still have failed device-side execution, which shows
        up in the separate ``result`` field.

        Parameters
        ----------
        job_id
            ID of the job to query.

        Returns
        -------
        result : str | None
            The job's terminal result (e.g. "Successful"/"Unsuccessful"), or None
            if the field is absent.
        """
        result = getattr(self.get_job(job_id), "result", None)
        return result.value if isinstance(result, JobResult) else None

    def is_successful(self, result: str | None) -> bool:
        # A QDC job can report no ``result`` field at all and still have
        # succeeded on-device; only an explicit non-"Successful" value fails it.
        return result is None or result == JobResult.SUCCESSFUL.value

    def classify_failure(self, result: str | None) -> JobOutcome:
        return (
            JobOutcome.RETRYABLE_ERROR
            if result == JobResult.ERROR.value
            else JobOutcome.RETRYABLE_UNSUCCESSFUL
        )

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
        Zip ``entries``, upload the bundle to QDC, and submit an automated job for it.

        See :meth:`DeviceFarm.submit_bundle` for the shared contract. QDC
        specifics: ``hub_device_name`` is mapped to a QDC target device via
        :data:`HUB_DEVICE_TO_QDC_DEVICE_MAP`; the zipped bundle is uploaded as
        a single ``ArtifactType.TESTSCRIPT`` artifact; and, unlike the AWS
        backend, ``entry_script`` is supported and forwarded to QDC verbatim
        (QDC runs it directly rather than requiring a fixed platform test spec).
        QDC queues the job server-side, so submission does not block on a free
        device slot; ``timeout`` is accepted for interface conformance with
        :meth:`DeviceFarm.submit_bundle` but unused here.

        Parameters
        ----------
        hub_device_name
            QAI Hub device name; looked up in ``HUB_DEVICE_TO_QDC_DEVICE_MAP``
            to determine the QDC target device and test framework.
        entries
            ``(source_path, arcname)`` pairs zipped into the uploaded test
            bundle.
        entry_script
            Path (relative to the bundle root) QDC should execute as the
            job's entry point.
        job_name
            Job name shown in QDC; truncated to ``QDC_JOB_NAME_LIMIT``.
        timeout
            Unused by this backend; see above.
        job_timeout
            Seconds QDC lets the job run on the device before aborting it.
            Defaults to ``QDC_DEFAULT_JOB_RUN_TIMEOUT`` when None.

        Returns
        -------
        job_id : str
            The QDC job id returned by ``qdc_api.submit_job``.
        """
        qdc_device = QDCDevice(hub_device_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "test.zip")
            create_zip_from_entries(zip_path, entries)
            artifact_id = self._upload_file(zip_path, ArtifactType.TESTSCRIPT)
        return self._submit_automated_job(
            qdc_device,
            [artifact_id],
            entry_script,
            job_name=job_name,
            job_timeout=QDC_DEFAULT_JOB_RUN_TIMEOUT
            if job_timeout is None
            else job_timeout,
        )

    def _submit_automated_job(
        self,
        qdc_device: QDCDevice,
        job_artifacts: list[str],
        entry_script: str | None,
        job_name: str = "QDC Automated Job",
        job_timeout: int = QDC_DEFAULT_JOB_RUN_TIMEOUT,
    ) -> str:
        """
        Submit an automated application job to QDC and return its job_id.

        Submission does not wait for a free device: QDC queues the job
        server-side and runs it when capacity frees up, so callers can submit
        the whole batch at once and poll the returned ids later.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        job_artifacts
            List of artifact IDs/descriptors to attach to the job.
        entry_script
            Optional entry script path for the job.
        job_name
            Name of the job to submit.
        job_timeout
            Seconds QDC lets the job run on the device before aborting it.

        Returns
        -------
        job_id : str
            The submitted job's ID.
        """
        target_id = qdc_api.get_target_id(self.client, qdc_device.qdc_name)
        # submit_job returns job id as int, so we cast it to str
        return str(
            _call_with_retry(
                lambda: qdc_api.submit_job(
                    public_api_client=self.client,
                    target_id=target_id,
                    job_name=job_name[:QDC_JOB_NAME_LIMIT],
                    external_job_id="ExJobId001",
                    job_type=JobType.AUTOMATED,
                    job_mode=JobMode.APPLICATION,
                    timeout=job_timeout,
                    test_framework=qdc_device.test_framework,
                    entry_script=entry_script,
                    job_artifacts=job_artifacts,
                    monkey_events=None,
                    monkey_session_timeout=None,
                    job_parameters=[JobSubmissionParameter.WIFIENABLED],
                ),
                f"submit_job({job_name})",
                extra_retryable_codes=(400,),
                max_retries=SUBMIT_MAX_RETRIES,
            )
        )

    def log_upload_status(
        self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT
    ) -> None:
        """
        Poll until job logs are uploaded (completed/failed).

        Parameters
        ----------
        job_id
            ID of the job to monitor.
        timeout
            Maximum time to wait for log upload in seconds.
            Defaults to DEFAULT_JOB_TIMEOUT (6 hours).

        Raises
        ------
        TimeoutError
            If logs are not uploaded within the timeout period.
        """
        status: LogUploadStatus | None = None
        elapsed = 0
        while elapsed <= timeout:
            status = _call_with_retry(
                lambda: qdc_api.get_job_log_upload_status(self.client, job_id),
                f"get_job_log_upload_status({job_id})",
            )
            if status == LogUploadStatus.NOLOGS and elapsed >= NOLOGS_GRACE_SECONDS:
                print(f"Job {job_id} still reports no logs after {elapsed}s.")
                return
            if status not in _TERMINAL_LOG_UPLOAD_STATES:
                print(
                    f"Job is completed and the server is uploading logs, "
                    f"waiting for {POLL_INTERVAL} seconds."
                )
                time.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL
            else:
                print("Job logs are uploaded.")
                return

        raise TimeoutError(
            f"Log upload for job {job_id} did not complete within {timeout} seconds. "
            f"Last status: {status}"
        )

    def get_job_log_files(self, job_id: str, wait_for_logs: bool = False) -> list:
        """Wrapper to get job log files using the QDC API.

        Parameters
        ----------
        job_id
            ID of the job to retrieve logs for.
        wait_for_logs
            If True, re-list (with capped exponential backoff) while the
            listing comes back empty, up to ``LOG_LISTING_MAX_RETRIES`` times.
            The file listing lags log-upload-status on the QDC backend, so a
            successful job can momentarily list zero files; callers that need
            the logs to extract metrics should set this. An empty list is still
            returned if the listing never populates -- the caller decides
            whether that is fatal.

        Returns
        -------
        job_log_files: list
            List of job log files, minus ``_UNPARSED_LOG_SUFFIXES``.
        """
        for attempt in range(LOG_LISTING_MAX_RETRIES):
            job_log_files = _call_with_retry(
                lambda: [
                    f
                    for f in qdc_api.get_job_log_files(self.client, job_id)
                    if getattr(f, "filename", None)
                    and not f.filename.endswith(_UNPARSED_LOG_SUFFIXES)
                ],
                f"get_job_log_files({job_id})",
            )
            if job_log_files or not wait_for_logs:
                return job_log_files
            if attempt < LOG_LISTING_MAX_RETRIES - 1:
                delay = _backoff_seconds(attempt)
                print(
                    f"[QDC] job {job_id} log listing is empty (upload reported "
                    f"complete); the file listing lags behind. Attempt "
                    f"{attempt + 1}/{LOG_LISTING_MAX_RETRIES}, retrying in {delay}s."
                )
                time.sleep(delay)
        return job_log_files

    def download_job_log_files(self, filename: str, target_path: str) -> None:
        """Download job log files from QDC.

        Parameters
        ----------
        filename
            Name of the log file to download.
        target_path
            Local path to save the downloaded file.
        """
        # v2 serves the raw file from a presigned URL; re-wrap it into the
        # single-entry zip that v1 returned and callers unpack. Safe to retry:
        # each attempt rewrites raw_path in a single 'wb' write.
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = os.path.join(tmpdir, "log")
            _call_with_retry(
                lambda: qdc_api.download_job_log_files_v2(
                    self.client, filename, raw_path
                ),
                f"download_job_log_files_v2({filename})",
            )
            create_zip_from_entries(
                target_path, [(raw_path, os.path.basename(filename))]
            )

    def _upload_file(self, file_path: str, artifact_type: ArtifactType) -> str:
        """Upload a file to QDC.

        Parameters
        ----------
        file_path
            Path to the file to upload.
        artifact_type
            Type of artifact being uploaded.

        Returns
        -------
        artifact_id: str
            ID of the uploaded artifact.
        """
        # The SDK uploads to presigned URLs (start -> PUT parts -> end); the
        # artifact only commits at the final `end_upload_v2`, and upload_file_v2
        # returns the uuid of the *successful* attempt -- so a retry after a
        # mid-transfer drop at worst leaves an orphaned, unreferenced partial
        # artifact server-side (a benign leak), never a duplicate job.
        artifact_id = _call_with_retry(
            lambda: qdc_api.upload_file_v2(self.client, file_path, artifact_type),
            f"upload_file_v2({file_path})",
        )
        if artifact_id is None:
            raise RuntimeError(f"upload_file_v2 returned None for {file_path}")
        return artifact_id
