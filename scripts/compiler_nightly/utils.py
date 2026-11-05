# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared utilities and constants for compiler nightly scorecard scripts."""

from __future__ import annotations

import csv
import logging
import os
import re
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from prettytable import PrettyTable
from qai_hub import Client, CompileJob, Job, JobType, LinkJob
from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

# Project and configuration constants
AIHW_COMPILER_NIGHTLY_PROJECT = os.environ.get("COMPILER_NIGHTLY_PROJECT_ID", "")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_MAX_WORKERS = 10
MAX_RERUN_WORKERS = 50
KNOWN_FAILURES_CONFIG = Path(__file__).parent / "config" / "known_failures.yaml"

# Job status constants
JOB_STATUS_SUCCESS = "SUCCESS"
JOB_STATUS_FAILED = "FAILED"
MAX_JOB_RUNTIME_SECONDS = 3 * 3600  # 3 hours

# Display constants
DISPLAY_SEPARATOR = "=" * 80

# Model name parsing constants
DEVICE_PATTERNS = ("cs_", "samsung_")

_LOG_AND_PRINT_STATE = {"uses_logger_console": False}


def get_date_str() -> str:
    return datetime.now().strftime("%m-%d-%Y")


def wait_for_job_with_timeout(job: Job, model_name: str) -> str:
    """Return job's terminal status code, treating it as FAILED if still running past MAX_JOB_RUNTIME_SECONDS."""
    status = job.get_status()
    if status.finished:
        return status.code

    # Still running: fail it if it has exceeded the timeout threshold.
    job_created = job.date
    current_time = datetime.now(job_created.tzinfo)
    elapsed_time = current_time - job_created

    if elapsed_time.total_seconds() > MAX_JOB_RUNTIME_SECONDS:
        logger.warning(
            f"{model_name}: Job has been running for {elapsed_time.total_seconds() / 3600:.1f} hours. "
            f"Treating as failed (timeout after {MAX_JOB_RUNTIME_SECONDS / 3600:.1f}h)."
        )
        return JOB_STATUS_FAILED

    # Wait for the remaining time up to the timeout threshold. job.wait() returns
    # immediately for already-terminal jobs, so no separate get_status() guard.
    remaining_seconds = MAX_JOB_RUNTIME_SECONDS - elapsed_time.total_seconds()
    try:
        status = job.wait(timeout=max(1, int(remaining_seconds)))
        return status.code
    except TimeoutError:
        logger.warning(
            f"{model_name}: Job timed out after {MAX_JOB_RUNTIME_SECONDS / 3600:.1f} hours."
        )
        return JOB_STATUS_FAILED


def validate_tag(tag: str) -> str:
    """Reject tags with path-traversal or other unsafe characters."""
    if not re.fullmatch(r"[\w\-\.]+", tag):
        raise ValueError(f"Unsafe tag: {tag!r}")
    return tag


def extract_tag_and_dir_from_yaml(yaml_path: Path) -> tuple[str, Path]:
    """Extract tag and output directory from YAML file path.

    Expects filenames like: dev-compile-jobs__TAG.yaml
    """
    output_dir = yaml_path.parent
    stem = yaml_path.stem
    tag = stem.split("__", 1)[1] if "__" in stem else get_date_str()
    return validate_tag(tag), output_dir


def load_client(profile: str) -> Client:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading client with profile: {profile}")
    return Client(profile=profile)


def load_yaml_safe(yaml_path: Path, return_empty_on_not_found: bool = False) -> dict:
    logger = logging.getLogger(__name__)
    try:
        yaml = YAML(typ="safe", pure=True)
        with open(yaml_path) as f:
            return yaml.load(f) or {}
    except FileNotFoundError:
        if return_empty_on_not_found:
            logger.warning(f"File not found: {yaml_path}")
            return {}
        raise


def save_yaml_results(data: dict, output_path: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to {output_path}")
    yaml = YAML()
    yaml.default_flow_style = False
    with open(output_path, "w") as f:
        yaml.dump(data, f)


def setup_script_logging(
    output_dir: Path, script_name: str, verbose: bool, date_str: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"{script_name}-{date_str}.log"
    setup_logging(log_file, verbose)
    return log_file


def setup_logging(log_file: Path, verbose: bool = False) -> None:
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if verbose else logging.ERROR)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    _LOG_AND_PRINT_STATE["uses_logger_console"] = verbose and not root_logger.handlers
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
    )


def log_and_print(message: str, logger: logging.Logger) -> None:
    logger.info(message)
    if not (
        _LOG_AND_PRINT_STATE["uses_logger_console"]
        and logger.isEnabledFor(logging.INFO)
    ):
        print(message)


def get_aihw_compiler_nightly_project() -> str | None:
    # Returns None when unset (e.g. bespoke sandbox HUB_ENVs that have no fixed
    # project). Passing project=None to Hub lets the backend pick the default
    # project for the authenticated deployment. On 'dev' the env var pins the
    # shared compiler-nightly project so all runs land in the same place.
    return AIHW_COMPILER_NIGHTLY_PROJECT or None


def strip_device_suffix(model_name: str) -> str:
    """Strip device suffix from model name, preserving component suffix.

    Examples
    --------
        "model_name-cs_8_gen_3" -> "model_name"
        "model_name-cs_8_gen_3_encoder_1" -> "model_name_encoder_1"
        "model_name-samsung_s24" -> "model_name"
        "model_name-samsung_s24_decoder" -> "model_name_decoder"
        "model_name" -> "model_name"
    """
    # Match: base-device_suffix_component_suffix
    # Device patterns: cs_\d+(_gen_\d+|_elite|_elite_gen_\d+)? or samsung_\w+
    device_regex = r"^(.+)-(cs_\d+(?:_gen_\d+|_elite(?:_gen_\d+)?)?|samsung_\w+)(_.+)?$"
    match = re.match(device_regex, model_name)
    if match:
        base = match.group(1)
        component = match.group(3) or ""
        return base + component
    return model_name


def merge_job_options(base_options: str, extra_options: str | None) -> str:
    if extra_options:
        return f"{base_options} {extra_options}".strip()
    return base_options


def map_prod_by_model(prod_config: dict) -> dict:
    logger = logging.getLogger(__name__)
    prod_by_model = {}
    for prod_key, prod_info in prod_config.items():
        model_name_only = strip_device_suffix(prod_key)
        if model_name_only in prod_by_model:
            logger.warning(
                f"map_prod_by_model: duplicate model key '{model_name_only}' "
                f"(from '{prod_key}'), overwriting previous entry"
            )
        prod_by_model[model_name_only] = prod_info
    return prod_by_model


def create_results_table(
    models: dict,
    field_names: list[str],
    row_extractor: Callable[[str, dict], list[Any]],
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
) -> PrettyTable:
    table = PrettyTable()
    table.field_names = field_names
    for field in field_names:
        table.align[field] = "l"

    sorted_models = sorted(models.items(), key=sort_key) if sort_key else models.items()
    for model_name, info in sorted_models:
        table.add_row(row_extractor(model_name, info))

    return table


def print_results_table(
    models: dict,
    title: str,
    field_names: list[str],
    row_extractor: Callable[[str, dict], list[Any]],
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
    empty_message: str = "No models to display.",
    print_to_console: bool = True,
) -> None:
    logger = logging.getLogger(__name__)

    if not models:
        if print_to_console:
            log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
            log_and_print(empty_message, logger)
            log_and_print(DISPLAY_SEPARATOR, logger)
        else:
            logger.info(f"\n{DISPLAY_SEPARATOR}")
            logger.info(empty_message)
            logger.info(DISPLAY_SEPARATOR)
        return

    table = create_results_table(models, field_names, row_extractor, sort_key)
    table_str = str(table)

    if print_to_console:
        log_and_print(f"\n{DISPLAY_SEPARATOR}", logger)
        log_and_print(title, logger)
        log_and_print(DISPLAY_SEPARATOR, logger)
        log_and_print(table_str, logger)
    else:
        logger.info(f"\n{DISPLAY_SEPARATOR}")
        logger.info(title)
        logger.info(DISPLAY_SEPARATOR)
        logger.info(table_str)


def save_results_csv(
    results: dict,
    output_path: Path,
    field_names: list[str],
    row_extractor: Callable[[str, dict], list[Any]],
    sort_key: Callable[[tuple[str, Any]], Any] | None = None,
) -> Path:
    logger = logging.getLogger(__name__)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_results = (
        sorted(results.items(), key=sort_key) if sort_key else results.items()
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(field_names)
        for model_name, info in sorted_results:
            writer.writerow(row_extractor(model_name, info))

    log_and_print(f"Saved results to: {output_path}", logger)
    return output_path


@cache
def load_known_failures(job_type: str) -> list:
    config = load_yaml_safe(KNOWN_FAILURES_CONFIG, return_empty_on_not_found=True)
    return config.get(job_type, []) or []


def job_log_contains_error(
    client: Client, job_id: str, error: str, job_type: JobType
) -> bool:
    job = client.get_job(job_id, job_type)
    with tempfile.TemporaryDirectory() as temp_dir:
        for log_path in job.download_job_logs(temp_dir):
            with open(log_path, errors="replace") as f:
                if error in f.read():
                    return True
    return False


def _is_known_failure(
    client: Client,
    model_name: str,
    job_id: str | None,
    known_failures: list,
    job_type: JobType,
) -> bool:
    if not job_id:
        return False
    for entry in known_failures:
        if any(m in model_name for m in entry["models"]) and job_log_contains_error(
            client, job_id, entry["error"], job_type
        ):
            return True
    return False


def filter_known_failures(
    regressions: dict,
    client: Client,
    job_type: JobType,
    job_id_key: str,
) -> tuple[dict, dict]:
    known_failures = load_known_failures(job_type.name.lower())
    real: dict = {}
    known: dict = {}

    for model_name, info in regressions.items():
        job_id = info.get(job_id_key)
        is_known = _is_known_failure(
            client, model_name, job_id, known_failures, job_type
        )
        (known if is_known else real)[model_name] = info

    return real, known


def _resubmit_identical_job(client: Client, job: Job, project_id: str | None) -> Job:
    """Resubmit a compile or link job with identical parameters on `client`.

    `project_id` pins the re-run into the same shared Hub project as the
    original nightly submission (see `get_aihw_compiler_nightly_project`).
    """
    if isinstance(job, CompileJob):
        return client.submit_compile_job(
            model=job.model,
            device=job.device,
            name=job.name,
            input_specs=job.shapes,
            options=job.options,
            project=project_id,
        )
    if isinstance(job, LinkJob):
        return client.submit_link_job(
            models=job.models,
            device=job.device,
            name=job.name,
            options=job.options,
            project=project_id,
        )
    raise TypeError(
        f"Cannot resubmit job {job.job_id}: unsupported type {type(job).__name__}"
    )


def _rerun_single_regression(
    client: Client,
    model_name: str,
    info: dict,
    job_id_key: str,
    url_key: str,
    project_id: str | None,
) -> tuple[str, dict, str]:
    """Re-run one regression job and wait for its terminal status.

    Returns (model_name, updated_info, status_code). The returned info repoints
    its job id / url fields to the re-run job and records the original job id.
    """
    logger = logging.getLogger(__name__)
    orig_job_id = info.get(job_id_key)
    orig_job = client.get_job(orig_job_id)
    new_job = _resubmit_identical_job(client, orig_job, project_id)
    if isinstance(new_job, list):
        if len(new_job) > 1:
            logger.warning(
                f"Re-run of {model_name} ({orig_job_id}) returned "
                f"{len(new_job)} jobs; only the first is awaited/tracked"
            )
        new_job = new_job[0]

    status_code = wait_for_job_with_timeout(new_job, model_name)

    updated = info.copy()
    updated["original_job"] = orig_job_id
    updated["original_job_url"] = info.get(url_key)
    updated["rerun_status"] = status_code
    # Keep dev_status in sync so downstream tables (e.g. post_link_results'
    # status table) reflect the re-run outcome rather than the stale failure.
    if "dev_status" in updated:
        updated["dev_status"] = status_code
    updated[job_id_key] = new_job.job_id
    updated[url_key] = new_job.url
    return model_name, updated, status_code


def rerun_regressions(
    regressions: dict,
    client: Client,
    job_id_key: str,
    max_workers: int | None = None,
) -> tuple[dict, dict]:
    """Re-run failed regression jobs to separate real regressions from infra flakes.

    Failed jobs flagged as regressions are sometimes caused by infrastructure
    issues rather than a genuine compiler/linker change. Each regression is
    resubmitted with identical parameters and awaited:
      - Fails again -> real regression.
      - Passes      -> infrastructure failure (transient).

    Parameters
    ----------
    regressions:
        Regression map (already filtered of known failures).
    client:
        Dev hub client used to fetch and resubmit the jobs.
    job_id_key:
        Key in each info dict holding the dev job id ("dev_job" for compile,
        "link_job" for link).
    max_workers:
        Parallelism for awaiting re-run jobs. Defaults to one worker per
        regression (capped at `MAX_RERUN_WORKERS`). This work is I/O-bound
        (blocking on network polling), so the shared `DEFAULT_MAX_WORKERS` fan-out
        cap would serialize large regression batches — the exact case this
        feature exists to catch.

    Returns
    -------
    tuple[dict, dict]
        (real_regressions, infra_failures). Both carry the re-run job id/url and
        an `original_job` field for traceability.
    """
    if not regressions:
        return {}, {}

    if max_workers is None:
        max_workers = min(len(regressions), MAX_RERUN_WORKERS)

    # Both compile and link result dicts expose the dev job url under this key.
    url_key = "dev_job_url"

    # Pin re-runs into the same shared Hub project as the original nightly runs.
    project_id = get_aihw_compiler_nightly_project()

    logger = logging.getLogger(__name__)
    log_and_print(
        f"Re-running {len(regressions)} regression(s) to rule out infra failures",
        logger,
    )

    real: dict = {}
    infra: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _rerun_single_regression,
                client,
                model_name,
                info,
                job_id_key,
                url_key,
                project_id,
            ): model_name
            for model_name, info in regressions.items()
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                model_name, updated, status_code = future.result()
            except Exception:
                # If the re-run itself errors, keep it as a real regression so it
                # is not silently dropped.
                logger.exception(
                    f"Re-run failed for {model_name}; keeping as regression"
                )
                real[model_name] = regressions[model_name]
                continue

            if status_code == JOB_STATUS_SUCCESS:
                infra[model_name] = updated
                log_and_print(
                    f"  {model_name}: INFRA FAILURE (passed on re-run)", logger
                )
            else:
                real[model_name] = updated
                log_and_print(
                    f"  {model_name}: REAL REGRESSION (failed on re-run)", logger
                )

    log_and_print(
        f"Re-run complete: {len(real)} real regression(s), "
        f"{len(infra)} infrastructure failure(s)",
        logger,
    )
    return real, infra


def find_passing_known_failures(passing: dict, job_type: JobType) -> dict:
    known_failures = load_known_failures(job_type.name.lower())
    models = [m for e in known_failures for m in e["models"]]
    return {
        model_name: info
        for model_name, info in passing.items()
        if any(m in model_name for m in models)
    }
