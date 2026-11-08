# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Backend-agnostic device-farm job submission/collection."""

from __future__ import annotations

import contextlib
import os
import pathlib
import re
import shutil
import sys
import tempfile
import uuid
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import qai_hub as hub
import ruamel.yaml
from filelock import FileLock

from qai_hub_models.scorecard.device import ScorecardDevice

# Default timeout for job status polling (in seconds).
DEFAULT_JOB_TIMEOUT = 21600  # 6 hours
# Log-upload poll budget when salvaging logs from an already-failed job. Short on
# purpose: the job is lost either way, so waiting out DEFAULT_JOB_TIMEOUT is wasted.
FAILED_JOB_LOG_TIMEOUT = 300  # 5 minutes


def sanitize_job_id(job_id: str) -> str:
    """Make a job ID safe as a single path component.

    AWS Device Farm job IDs are ARNs (``arn:aws:devicefarm:...:run:.../...``):
    "/" would nest into missing directories, and upload-artifact rejects ":".
    """
    return re.sub(r"[:/\\]", "_", job_id)


# ---- zip helpers (shared by artifact staging and log archiving) -----------


def safe_extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extract a device log archive, rejecting zip-slip members."""
    safe_root = pathlib.Path(dest_dir).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            dest = (safe_root / member).resolve()
            if not str(dest).startswith(str(safe_root) + os.sep):
                raise ValueError(f"Zip slip detected in log archive: {member}")
        zf.extractall(safe_root)


def create_zip(zip_path: str, source_dir: str | os.PathLike) -> None:
    """Zip ``source_dir`` into ``zip_path`` with no compression.

    ZIP_STORED (no compression) is used for speed; the bundled files are
    already-compressed binaries.
    """
    source_dir = str(source_dir)
    entries: list[tuple[str, str]] = []
    for root, _, files in os.walk(source_dir):
        for fn in files:
            abs_path = os.path.join(root, fn)
            entries.append((abs_path, os.path.relpath(abs_path, source_dir)))
    create_zip_from_entries(zip_path, entries)


def walk_dir_entries(src_dir: str, arcname_prefix: str = "") -> list[tuple[str, str]]:
    """List (abs_path, arcname) pairs for every file under ``src_dir``.

    ``arcname_prefix`` (if non-empty) is prepended to each arcname so the
    zip preserves a directory layout without needing a filesystem copy.
    """
    entries: list[tuple[str, str]] = []
    src_dir = os.fspath(src_dir)
    for root, _, files in os.walk(src_dir):
        for fn in files:
            abs_path = os.path.join(root, fn)
            rel = os.path.relpath(abs_path, src_dir)
            arcname = os.path.join(arcname_prefix, rel) if arcname_prefix else rel
            entries.append((abs_path, arcname))
    return entries


def create_zip_from_entries(zip_path: str, entries: list[tuple[str, str]]) -> None:
    """Zip an explicit list of (source_path, arcname) into ``zip_path``.

    ZIP_STORED with force_zip64 so stored members over 2 GiB don't abort
    mid-write. Streams each source through shutil.copyfileobj so peak RAM
    is constant regardless of member size.

    Prefer this over :func:`create_zip` when the bundle would otherwise
    require a full-tree copytree to compose: pass paths from the original
    locations directly so /scratch doesn't need to hold a duplicate copy.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for src_path, arcname in entries:
            with (
                open(src_path, "rb") as src,
                zf.open(arcname, "w", force_zip64=True) as dest,
            ):
                shutil.copyfileobj(src, dest)


# ---- hub-device introspection -----------------------------------------


class HubDevicePlatform:
    """Wraps a QAI Hub device and exposes platform/capability properties."""

    def __init__(self, hub_device_name: str) -> None:
        self.device = hub.get_devices(hub_device_name)[-1]
        self.device_attributes = getattr(self.device, "attributes", [])

    @property
    def hexagon_version(self) -> str:
        """Hexagon version string parsed from the device's hub attributes."""
        htp_version = None
        for attr in self.device_attributes:
            if "hexagon" in attr:
                htp_version = attr.split(":")[-1]
        assert htp_version is not None, (
            f"Hexagon/HTP version not found in device attributes. "
            f"Device: {getattr(self.device, 'name', 'unknown')!r}. "
            f"Attributes: {self.device_attributes!r}"
        )
        return htp_version

    @property
    def is_windows_platform(self) -> bool:
        """True if the device runs Windows, based on hub attributes."""
        for attr in self.device_attributes:
            if "os" in attr and attr.endswith("windows"):
                return True
        return False

    @property
    def is_mobile_platform(self) -> bool:
        """True if the device is a phone form factor, based on hub attributes."""
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("phone"):
                return True
        return False

    @property
    def is_auto_platform(self) -> bool:
        """True if the device is an automotive form factor, based on hub attributes."""
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("auto"):
                return True
        return False

    @property
    def is_iot_platform(self) -> bool:
        """True if the device is an IoT form factor, based on hub attributes."""
        for attr in self.device_attributes:
            if "format" in attr and attr.endswith("iot"):
                return True
        return False


# ---- persistent job records for LLM perf collection ------------------------
# Flat YAML mapping of `<model>_<precision>_<runtime>_<device>` -> record.
# Submit writes one entry per job, collect polls each.

_JobRecordRetT = TypeVar("_JobRecordRetT")


class JobOutcome(str, Enum):
    SUCCESS = "success"
    RETRYABLE_ERROR = "retryable_error"
    RETRYABLE_UNSUCCESSFUL = "retryable_unsuccessful"
    RETRYABLE_EMPTY_LOGS = "retryable_empty_logs"


@dataclass
class JobRecord:
    job_id: str
    # Saved at submit so collect needn't re-download the bundle for metadata.json.
    context_lengths: list[int] | None = None


def make_key(model_id: str, precision: str, runtime: str, device_name: str) -> str:
    return f"{model_id}_{precision}_{runtime}_{device_name}"


def load_jobs(jobs_file: str | Path) -> dict[str, JobRecord]:
    p = Path(jobs_file)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    with p.open(encoding="utf-8") as f:
        raw = ruamel.yaml.YAML().load(f) or {}
    return {
        k: JobRecord(
            job_id=str(v["job_id"]),
            context_lengths=(
                [int(c) for c in v["context_lengths"]]
                if "context_lengths" in v
                else None
            ),
        )
        for k, v in raw.items()
        if isinstance(v, dict) and "job_id" in v
    }


def poll_job(
    job_id: str,
    collect_fn: Callable[[str], tuple[_JobRecordRetT, JobOutcome, str | None]],
) -> _JobRecordRetT:
    """Poll a device-farm job once; raise RuntimeError if it did not succeed."""
    result, outcome, reason = collect_fn(job_id)
    if outcome is not JobOutcome.SUCCESS:
        raise RuntimeError(f"{reason}. Check device farm job logs for details.")
    return result


def save_job(
    jobs_file: str | Path,
    key: str,
    job_id: str,
    context_lengths: list[int] | None = None,
) -> None:
    """Upsert one row into ``jobs_file``. FileLock-guarded for parallel submitters."""
    p = Path(jobs_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(f"{p}.lock"), p.open("a+", encoding="utf-8") as f:
        f.seek(0)
        raw = f.read()
        mapping: dict[str, Any] = (
            dict(ruamel.yaml.YAML().load(raw) or {}) if raw.strip() else {}
        )
        row: dict[str, Any] = {"job_id": job_id}
        if context_lengths is not None:
            row["context_lengths"] = list(context_lengths)
        mapping[key] = row
        f.seek(0)
        f.truncate(0)
        ruamel.yaml.YAML().dump(mapping, f)
        f.flush()


# ---- base class -------------------------------------------------------------


class DeviceFarm(ABC):
    """Shared contract for device-farm backends (QDC, AWS Device Farm)."""

    @abstractmethod
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
        Zip ``entries``, upload the bundle, and submit an automated job for it.

        Implementations queue the job on the backend's device pool and return
        immediately once submission succeeds; they do not block for job
        completion (see :meth:`status`).

        Parameters
        ----------
        hub_device_name
            QAI Hub device name identifying which physical device/pool to run
            on. Each backend maps this to its own device identifier (e.g. a
            QDC target id or an AWS device ARN).
        entries
            ``(source_path, arcname)`` pairs (see :func:`create_zip_from_entries`)
            describing the files to zip into the test bundle uploaded to the
            device.
        entry_script
            Path (relative to the bundle root) of the script the device
            should execute as the job's entry point. Backend support for this
            varies -- some backends require a fixed platform-specific test
            spec instead and reject a non-``None`` value.
        job_name
            Human-readable job name shown in the backend's UI/API. May be
            truncated to the backend's length limit.
        timeout
            Maximum seconds to wait for a device/job slot to free up before
            submission gives up. Does not bound job execution time.
        job_timeout
            Maximum seconds the job may run on the device. If None, the
            backend's default applies: QDC uses its fixed default run
            timeout, and AWS Device Farm falls back to ``timeout`` (which on
            AWS does bound execution).

        Returns
        -------
        job_id : str
            Backend-native identifier for the submitted job (e.g. a QDC job
            id or an AWS Device Farm run ARN). Passed back into :meth:`status`,
            :meth:`result`, and the log-retrieval methods.
        """

    @abstractmethod
    def status(self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT) -> str:
        """
        Poll ``job_id`` until it reaches a terminal lifecycle state, then return that state.

        Parameters
        ----------
        job_id
            Backend-native job identifier returned by :meth:`submit_bundle`.
        timeout
            Maximum seconds to poll before giving up.

        Returns
        -------
        status : str
            Backend-native terminal lifecycle status string (e.g. ``"COMPLETED"``).
            This describes whether the job *finished running*, not whether it
            passed -- see :meth:`result` for the pass/fail outcome.

        Raises
        ------
        TimeoutError
            If the job has not reached a terminal state within ``timeout`` seconds.
        """

    @abstractmethod
    def is_done(self, job_id: str) -> bool:
        """Whether ``job_id`` has reached a terminal state. One call, never blocks."""

    def count_waiting_jobs(self, hub_device_name: str) -> int | None:
        """Jobs this account has queued for ``hub_device_name`` but not yet on a device.

        Counts every submitter (parallel CI legs included), not only this
        process. None means the backend can't tell, so callers shouldn't throttle.
        """
        return None

    def count_unfinished_jobs(self) -> int | None:
        """Jobs this account has queued or running, across every device.

        None means the backend can't tell.
        """
        return None

    @abstractmethod
    def result(self, job_id: str) -> str | None:
        """
        Return the terminal, backend-native result string for a completed job.

        Parameters
        ----------
        job_id
            Backend-native job identifier for a job that has already reached
            a terminal status (see :meth:`status`).

        Returns
        -------
        result : str | None
            Backend-native pass/fail result string (e.g. ``"PASSED"``,
            ``"FAILED"``), or ``None`` if the backend has no result recorded
            for this job. Pass the value to :meth:`is_successful` or
            :meth:`classify_failure` to interpret it.
        """

    @abstractmethod
    def is_successful(self, result: str | None) -> bool:
        """
        Whether a value from :meth:`result` indicates the job passed.

        Parameters
        ----------
        result
            A value previously returned by :meth:`result` for this backend.

        Returns
        -------
        success : bool
            True only for the backend-native result string that means the
            job's on-device test actually passed; False for every other
            result value, including ``None``.
        """

    def classify_failure(self, result: str | None) -> JobOutcome:
        """
        Classify a non-successful :meth:`result` value into a retry-relevant outcome.

        Only called when :meth:`is_successful` returned False for ``result``.
        The base implementation always returns ``RETRYABLE_UNSUCCESSFUL``;
        override this to distinguish failure modes a backend can tell apart
        (e.g. an infra error worth an automatic retry vs. a genuine on-device
        test failure) so callers can retry only the former.

        Parameters
        ----------
        result
            A non-successful value previously returned by :meth:`result`.

        Returns
        -------
        outcome : JobOutcome
            The failure classification used by the caller's retry logic.
        """
        return JobOutcome.RETRYABLE_UNSUCCESSFUL

    @abstractmethod
    def get_job_log_files(self, job_id: str, wait_for_logs: bool = False) -> list:
        """
        Return the listing of log files the backend has recorded for ``job_id``.

        Parameters
        ----------
        job_id
            Backend-native job identifier for a job that has already reached
            a terminal status (see :meth:`status`).
        wait_for_logs
            If True, wait for the backend to finish uploading logs before
            returning the listing, rather than returning whatever is
            available immediately.

        Returns
        -------
        log_files : list
            Backend-native log-file descriptor objects, each exposing at
            least a ``.filename`` attribute suitable for passing to
            :meth:`download_job_log_files`. Empty if the job produced no logs.
        """

    @abstractmethod
    def download_job_log_files(self, filename: str, target_path: str) -> None:
        """
        Download one job log file (zip-wrapped) to ``target_path``.

        Parameters
        ----------
        filename
            The ``.filename`` of a log-file descriptor from
            :meth:`get_job_log_files`.
        target_path
            Local filesystem path to write the downloaded zip to. Any
            existing file at this path is overwritten.

        Raises
        ------
        Exception
            Backend-specific; on any failure to fetch the file (network
            error, missing/expired artifact, etc.). Callers that want a
            best-effort download should use :meth:`try_download_job_log_files`
            instead of calling this directly.
        """

    def log_upload_status(
        self, job_id: str, timeout: int = DEFAULT_JOB_TIMEOUT
    ) -> None:
        """Wait for logs to finish uploading. No-op by default.

        QDC exposes log-upload-status as a signal independent of the file
        listing and overrides this; other backends' ``get_job_log_files``
        already reflects the final state once the job is terminal.
        """
        return

    def try_download_job_log_files(self, filename: str, target_path: str) -> bool:
        """Best-effort ``download_job_log_files``; return False if it never landed.

        One unreadable log file shouldn't discard a whole collection: a run was
        lost on a debug log no metric parser reads, after its device job had
        already succeeded. Callers skip the file and parse what did arrive; a
        missing results file still shows up as absent metrics.
        """
        try:
            self.download_job_log_files(filename, target_path)
            return True
        except Exception as err:
            # Type only, never the message: backend errors may embed secrets.
            print(
                f"[{type(self).__name__}] giving up on log file {filename} after "
                f"retries ({type(err).__name__}); continuing with the rest.",
                file=sys.stderr,
            )
            return False

    def download_job_logs(self, job_log_files: list, dest_dir: str) -> int:
        """Download and unwrap each log file at its ``.filename`` under ``dest_dir``.

        Returns the count that landed; see :meth:`try_download_job_log_files`
        for why a missing file is skipped rather than raised.
        """
        saved = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            for job_log in job_log_files:
                target = os.path.join(tmpdir, f"{uuid.uuid4().hex}.zip")
                if not self.try_download_job_log_files(job_log.filename, target):
                    continue
                dest = os.path.join(dest_dir, job_log.filename)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                # Logs are served as a zip; unwrap it so callers see readable
                # files. A non-zip payload is kept as-is.
                try:
                    safe_extract_zip(target, os.path.dirname(dest))
                except (zipfile.BadZipFile, ValueError):
                    shutil.move(target, dest)
                saved += 1
        return saved

    def archive_job_logs(
        self,
        job_id: str,
        logs_dir: str,
        save_logs_dir: str | None,
        label: str | None = None,
    ) -> None:
        """Best-effort: zip ``logs_dir`` (from :meth:`download_job_logs`) into
        ``<save_logs_dir>/<label or job_id>.zip``. Never raises.
        """
        if not save_logs_dir or not os.listdir(logs_dir):
            return
        try:
            os.makedirs(save_logs_dir, exist_ok=True)
            zip_name = sanitize_job_id(label or job_id)
            create_zip(os.path.join(save_logs_dir, f"{zip_name}.zip"), logs_dir)
        except Exception as err:
            # Type only, never the message: backend errors may embed secrets.
            print(
                f"[{type(self).__name__}] could not archive logs for job {job_id} "
                f"({type(err).__name__}); any failure reason still stands.",
                file=sys.stderr,
            )

    def save_job_logs(
        self,
        job_id: str,
        save_logs_dir: str | None,
        job_log_files: list | None = None,
        label: str | None = None,
    ) -> int:
        """Best-effort: archive one job's logs into a single zip; return the count.

        Kept for successful and failed jobs alike. A green job's logs are the
        baseline a red one is read against. Everything for a job lands in one
        ``<label or job_id>.zip`` holding the log files themselves, rather than a
        directory of individually-zipped files that each need unwrapping. Never
        raises: a log-fetch problem must not replace a real failure reason.
        """
        if not save_logs_dir:
            return 0
        saved = 0
        try:
            os.makedirs(save_logs_dir, exist_ok=True)
            if job_log_files is None:
                with contextlib.suppress(TimeoutError):
                    self.log_upload_status(job_id, timeout=FAILED_JOB_LOG_TIMEOUT)
                job_log_files = self.get_job_log_files(job_id, wait_for_logs=True)
            if not job_log_files:
                print(
                    f"[{type(self).__name__}] no log files listed for job {job_id} "
                    f"after retries.",
                    file=sys.stderr,
                )
                return 0
            with tempfile.TemporaryDirectory() as staged:
                saved = self.download_job_logs(job_log_files, staged)
                self.archive_job_logs(job_id, staged, save_logs_dir, label=label)
        except Exception as err:
            # Type only, never the message: backend errors may embed secrets.
            print(
                f"[{type(self).__name__}] could not save logs for job {job_id} "
                f"({type(err).__name__}); any failure reason still stands.",
                file=sys.stderr,
            )
            return 0
        if saved:
            print(
                f"[{type(self).__name__}] archived {saved} log file(s) for job {job_id}"
            )
        return saved


def device_logs_dir_name(backend: DeviceFarm) -> str:
    """On-device log directory name to render into a bundle's device scripts.

    QDC's server-side log harvesting only picks up a directory named
    ``QDC_logs`` -- undocumented, discovered from a prior rename to a
    backend-agnostic ``device_logs`` silently breaking QDC log retrieval.
    AWS Device Farm has no such requirement (its test spec does its own
    ``adb pull`` of a name it controls), so a distinct ``AWS_logs`` name
    keeps that pull path unambiguous rather than implying a shared convention
    that doesn't exist. Checked by class name (not ``isinstance``) so this
    module doesn't need to import the backend classes it only instantiates
    lazily in :func:`get_device_farm`.
    """
    name = type(backend).__name__
    if name == "QDCDeviceFarm":
        return "QDC_logs"
    if name == "AwsDeviceFarm":
        return "AWS_logs"
    raise ValueError(
        f"No on-device log directory name is defined for backend {name!r}; "
        "add a case to device_logs_dir_name."
    )


def get_device_farm(device: ScorecardDevice) -> DeviceFarm:
    """Instantiate the right :class:`DeviceFarm` backend for ``device``.

    Imports are lazy so the QDC SDK / boto3 are only required when that
    backend is actually used.
    """
    if device.devicefarm_backend == "aws":
        from qai_hub_models.utils.devicefarm.backends.aws.aws import AwsDeviceFarm

        return AwsDeviceFarm()

    if device.devicefarm_backend == "qdc":
        from qai_hub_models.utils.devicefarm.backends.qdc.qdc import (
            QDCDeviceFarm,
            get_qdc_api_token,
        )

        return QDCDeviceFarm(api_key=get_qdc_api_token(device))

    raise ValueError(
        f"Unknown devicefarm_backend {device.devicefarm_backend!r} for device "
        f"{device.name!r}; expected 'qdc' or 'aws'."
    )
