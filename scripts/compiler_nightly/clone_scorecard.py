#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from ruamel.yaml import YAML

sys.path.insert(0, str(Path(__file__).parent))
from qai_hub import Client, JobType, Model, api_utils
from qai_hub import public_rest_api as api
from utils import (
    DEFAULT_MAX_WORKERS,
    get_aihw_compiler_nightly_project,
    load_client,
    load_yaml_safe,
    log_and_print,
    setup_script_logging,
)

PROD_MODEL_SEPARATOR = "__prod__"
MODEL_BATCH_SIZE = 100
DRY_RUN_MODEL_LIMIT = 10
MAX_DEV_MODELS = 2000

logger = logging.getLogger(__name__)


def format_dev_model_name(model_name: str, prod_model_id: str) -> str:
    return f"{model_name}{PROD_MODEL_SEPARATOR}{prod_model_id}"


def fetch_existing_dev_models(
    dev_client: Client, project_id: str | None = None
) -> dict[str, str]:
    existing_models_by_source: dict[str, str] = {}

    if project_id:
        logger.info(f"Fetching models from dev project {project_id}")
        config = dev_client._config
        session = requests.Session()
        header = api_utils.auth_header(config)
        offset = 0

        while True:
            url = api_utils.api_url(config, "projects", project_id, "models")
            url_params = api_utils.offset_limit_url_params(offset, MODEL_BATCH_SIZE)
            url_params["query"] = PROD_MODEL_SEPARATOR

            response = session.get(url, headers=header, params=url_params)
            model_list_pb = api_utils.response_as_protobuf(
                response, api.api_pb.ModelList
            )

            for model_pb in model_list_pb.models:
                source_id = model_pb.name.split(PROD_MODEL_SEPARATOR)[-1]
                existing_models_by_source.setdefault(source_id, model_pb.model_id)

            if len(model_list_pb.models) < MODEL_BATCH_SIZE:
                break
            offset += MODEL_BATCH_SIZE
    else:
        logger.info("Fetching models from dev (all projects)")
        offset = 0

        while offset < MAX_DEV_MODELS:
            model_list_pb = dev_client._api_call(
                api.get_model_list, offset=offset, limit=MODEL_BATCH_SIZE
            )

            # Early exit if no models returned
            if len(model_list_pb.models) == 0:
                break

            for model_pb in model_list_pb.models:
                if PROD_MODEL_SEPARATOR in model_pb.name:
                    source_id = model_pb.name.split(PROD_MODEL_SEPARATOR)[-1]
                    existing_models_by_source.setdefault(source_id, model_pb.model_id)

            # Stop if we received fewer models than requested (last page)
            if len(model_list_pb.models) < MODEL_BATCH_SIZE:
                break
            offset += MODEL_BATCH_SIZE

    log_and_print(
        f"Found {len(existing_models_by_source)} unique models in dev",
        logger,
    )
    return existing_models_by_source


def get_or_upload_model(
    source_model: Model,
    model_name: str,
    uploaded_models: dict[str, str],
    existing_dev_models: dict[str, str],
    dev_client: Client,
    project_id: str | None,
    lock: Lock,
) -> tuple[str, bool]:
    # Check read-only dict first (no lock needed)
    if source_model.model_id in existing_dev_models:
        dev_model_id = existing_dev_models[source_model.model_id]
        with lock:
            uploaded_models[source_model.model_id] = dev_model_id
        return dev_model_id, True

    # Check if already uploaded by another thread
    with lock:
        if source_model.model_id in uploaded_models:
            return uploaded_models[source_model.model_id], True

    # Perform upload without holding lock (allows parallel uploads)
    model_name_with_prod_id = format_dev_model_name(model_name, source_model.model_id)
    with tempfile.TemporaryDirectory(prefix=f"model_{model_name}_") as temp_dir:
        model_path = Path(temp_dir) / f"{model_name}_{source_model.model_id}"
        source_model.download(str(model_path))

        downloaded_files = list(
            Path(temp_dir).glob(f"{model_name}_{source_model.model_id}.*")
        )
        if not downloaded_files:
            raise RuntimeError(f"No file found after download in {temp_dir}")

        dev_model = dev_client.upload_model(
            str(downloaded_files[0]),
            name=model_name_with_prod_id,
            project=project_id,
        )

        with lock:
            uploaded_models[source_model.model_id] = dev_model.model_id
        return dev_model.model_id, False


def process_compile_job(
    prod_client: Client,
    dev_client: Client,
    job_id: str,
    model_name: str,
    uploaded_models: dict[str, str],
    existing_dev_models: dict[str, str],
    project_id: str | None,
    lock: Lock,
) -> tuple[dict, bool]:
    job = prod_client.get_job(job_id, JobType.COMPILE)

    dev_model_id, was_reused = get_or_upload_model(
        job.model,
        model_name,
        uploaded_models,
        existing_dev_models,
        dev_client,
        project_id,
        lock,
    )
    job_status = job.get_status()

    return {
        "prod_job": job_id,
        "prod_job_url": job.url,
        "prod_job_status": job_status.code,
        "model_id": dev_model_id,
        "model_name": model_name,
        "device": job.device.name,
        "input_specs": dict(job.shapes),
        "compile_options": job.options,
    }, was_reused


def process_link_job(prod_client: Client, job_id: str) -> tuple[dict, bool]:
    job = prod_client.get_job(job_id, JobType.LINK)
    job_status = job.get_status()

    result = {
        "prod_job": job_id,
        "prod_job_url": job.url,
        "prod_job_status": job_status.code,
        "device": job.device.name,
        "options": job.options,
    }
    return result, False


def filter_scorecard_by_runtime_name(
    scorecard: dict, runtime_filter: str | None
) -> dict:
    if not runtime_filter:
        return scorecard

    runtime = runtime_filter.upper()
    filtered = {
        model_name: job_id
        for model_name, job_id in scorecard.items()
        if f"_{runtime}-" in model_name or model_name.endswith(f"_{runtime}")
    }

    logger.info(
        f"Filtered scorecard by runtime name '{runtime_filter}': "
        f"{len(filtered)}/{len(scorecard)} jobs"
    )
    return filtered


def filter_scorecard_by_device_name(scorecard: dict, device_filter: str | None) -> dict:
    if not device_filter:
        return scorecard

    device_pattern = f"-{device_filter.lower()}"
    filtered = {}

    for model_name, job_id in scorecard.items():
        model_lower = model_name.lower()
        idx = model_lower.find(device_pattern)
        if idx == -1:
            continue

        after_idx = idx + len(device_pattern)

        # Valid if at end, followed by '-', or followed by '_' (but not '_gen')
        if (
            after_idx >= len(model_lower)
            or model_lower[after_idx] == "-"
            or (
                model_lower[after_idx] == "_"
                and not model_lower[after_idx:].startswith("_gen")
            )
        ):
            filtered[model_name] = job_id

    logger.info(
        f"Filtered scorecard by device name '{device_filter}': "
        f"{len(filtered)}/{len(scorecard)} jobs"
    )
    return filtered


def process_profile_job(prod_client: Client, job_id: str) -> tuple[dict, bool]:
    job = prod_client.get_job(job_id, JobType.PROFILE)
    job_status = job.get_status()

    result: dict[str, str | int | dict | None] = {
        "prod_job": job_id,
        "prod_job_url": job.url,
        "prod_job_status": job_status.code,
        "device": job.device.name,
        "options": job.options,
    }

    if job_status.success:
        try:
            profile = job.download_profile()
            result["estimated_inference_time"] = int(
                profile["execution_summary"]["estimated_inference_time"]
            )
        except Exception as e:
            logger.warning(f"Failed to extract latency from {job_id}: {e}")

    return result, False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone AI Hub Models scorecard (compile, link, profile) to dev"
    )
    parser.add_argument(
        "--prod-profile",
        type=str,
        default="prod",
        help="Hub client profile for prod environment (default: prod)",
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--intermediates-dir",
        type=Path,
        required=True,
        help="Directory containing scorecard intermediate files (compile-jobs.yaml, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for logs and scorecard configs (default: results)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 10 models for testing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--compile-runtime",
        type=str,
        default=None,
        help="Filter compile jobs by runtime (case-insensitive, e.g., 'qnn_dlc', 'tflite', 'onnx')",
    )
    parser.add_argument(
        "--link-device",
        type=str,
        default=None,
        help="Filter link jobs by device (case-insensitive, e.g., 'cs_8_elite', 'monaco'). Also applies compile runtime filter to link jobs.",
    )
    parser.add_argument(
        "--profile-runtime",
        type=str,
        default="qnn_dlc",
        help="Filter profile jobs by runtime (case-insensitive, e.g., 'qnn_dlc', 'tflite', 'onnx')",
    )
    parser.add_argument(
        "--profile-device",
        type=str,
        default="cs_8_elite",
        help="Filter profile jobs by device (case-insensitive, e.g., 'cs_8_elite', 'monaco')",
    )
    parser.add_argument(
        "--always-upload",
        action="store_true",
        help="Always upload models from prod, skip checking for existing models in dev",
    )

    args = parser.parse_args()

    try:
        project_id = get_aihw_compiler_nightly_project()

        # Extract tag from intermediates directory name (e.g., intermediates_0.48.0)
        aihm_tag = args.intermediates_dir.name.replace("intermediates_", "")

        log_file = setup_script_logging(
            args.output_dir, "clone-aihm-scorecard", args.verbose, aihm_tag
        )
        log_and_print(f"Full logs: {log_file}", logger)
        log_and_print(f"Processing scorecard from AIHM tag: {aihm_tag}", logger)

        prod_client = load_client(args.prod_profile)
        dev_client = load_client(args.dev_profile)

        if args.always_upload:
            log_and_print("--always-upload: skipping model existence check", logger)
            existing_dev_models = {}
        else:
            existing_dev_models = fetch_existing_dev_models(dev_client, project_id)

        all_jobs: dict[str, dict[str, str]] = {
            "compile": load_yaml_safe(args.intermediates_dir / "compile-jobs.yaml"),
            "link": load_yaml_safe(args.intermediates_dir / "link-jobs.yaml"),
            "profile": load_yaml_safe(args.intermediates_dir / "profile-jobs.yaml"),
        }

        # Apply compile filter (runtime only)
        all_jobs["compile"] = filter_scorecard_by_runtime_name(
            all_jobs["compile"], args.compile_runtime
        )

        # Apply link filters
        # Link jobs only exist for qnn_dlc runtime, so skip if compile runtime is not qnn_dlc
        if args.compile_runtime and "qnn_dlc" not in args.compile_runtime.lower():
            all_jobs["link"] = {}
        else:
            all_jobs["link"] = filter_scorecard_by_device_name(
                all_jobs["link"], args.link_device
            )

        # Apply profile filters if specified
        all_jobs["profile"] = filter_scorecard_by_runtime_name(
            all_jobs["profile"], args.profile_runtime
        )
        all_jobs["profile"] = filter_scorecard_by_device_name(
            all_jobs["profile"], args.profile_device
        )

        if args.dry_run:
            all_jobs = {
                job_type: dict(list(jobs.items())[:DRY_RUN_MODEL_LIMIT])
                for job_type, jobs in all_jobs.items()
            }

        total_jobs = sum(len(jobs) for jobs in all_jobs.values())
        log_and_print(
            f"Processing {len(all_jobs['compile'])} compile, {len(all_jobs['link'])} link, "
            f"{len(all_jobs['profile'])} profile jobs ({total_jobs} total)",
            logger,
        )

        uploaded_models: dict[str, str] = {}
        lock = Lock()
        scorecards: dict[str, dict[str, dict]] = {
            "compile": {},
            "link": {},
            "profile": {},
        }
        stats = {"uploaded": 0, "reused": 0, "failed": 0}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: dict[Future[tuple[dict, bool]], tuple[str, str]] = {}

            for model_name, job_id in all_jobs["compile"].items():
                future = executor.submit(
                    process_compile_job,
                    prod_client,
                    dev_client,
                    job_id,
                    model_name,
                    uploaded_models,
                    existing_dev_models,
                    project_id,
                    lock,
                )
                futures[future] = ("compile", model_name)

            for model_name, job_id in all_jobs["link"].items():
                future = executor.submit(process_link_job, prod_client, job_id)
                futures[future] = ("link", model_name)

            for model_name, job_id in all_jobs["profile"].items():
                future = executor.submit(process_profile_job, prod_client, job_id)
                futures[future] = ("profile", model_name)

            for completed, future in enumerate(as_completed(futures), start=1):
                job_type, model_name = futures[future]

                try:
                    # All jobs now return (result_dict, was_reused)
                    result_dict, was_reused = future.result()
                    scorecards[job_type][model_name] = result_dict

                    if job_type == "compile":
                        stats["reused" if was_reused else "uploaded"] += 1
                        status = "reused" if was_reused else "uploaded"
                    else:
                        status = "processed"

                    print(
                        f"  [{completed}/{total_jobs}] {job_type} {model_name}: {status}"
                    )
                except Exception:
                    stats["failed"] += 1
                    logger.exception(f"Failed {job_type} {model_name}")
                    print(
                        f"  [{completed}/{total_jobs}] {job_type} {model_name}: FAILED"
                    )

        yaml_writer = YAML()
        yaml_writer.default_flow_style = False

        output_files = {
            "compile": args.output_dir / f"dev-compile-scorecard__{aihm_tag}.yaml",
            "link": args.output_dir / f"link-scorecard__{aihm_tag}.yaml",
            "profile": args.output_dir / f"profile-scorecard__{aihm_tag}.yaml",
        }

        for job_type, output_file in output_files.items():
            with open(output_file, "w") as f:
                yaml_writer.dump(scorecards[job_type], f)
            log_and_print(
                f"{job_type.capitalize()}: {len(scorecards[job_type])}/{len(all_jobs[job_type])} -> {output_file}",
                logger,
            )

        log_and_print(
            f"Models: {stats['uploaded']} uploaded, {stats['reused']} reused, {stats['failed']} failed",
            logger,
        )
        return 0

    except Exception:
        logger.exception("Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
