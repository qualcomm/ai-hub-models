#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import argparse
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qai_hub import Client, Device
from utils import (
    DEFAULT_MAX_WORKERS,
    get_aihw_compiler_nightly_project,
    get_date_str,
    load_client,
    load_yaml_safe,
    log_and_print,
    merge_job_options,
    save_yaml_results,
    setup_script_logging,
)

logger = logging.getLogger(__name__)

TFL_CONVERTER_OPTION = "--use_tfl_converter"
TFL_CONVERTER_PRECISIONS = {
    "float",
    "mixed_with_float",
    "w8a8",
    "w8a16",
    "mixed",
    "w8a8_mixed_int16",
    "w8a16_mixed_int16",
    "w8a16_mixed_fp16",
}


def filter_scorecard_by_runtime(scorecard: dict, runtimes: list[str]) -> dict:
    if not runtimes:
        return scorecard

    filtered = {}
    for model_name, job_spec in scorecard.items():
        compile_options = job_spec.get("compile_options", "")

        for runtime in runtimes:
            if runtime == "qnn_dlc_via_ep":
                matches = (
                    "--target_runtime qnn_dlc" in compile_options
                    and "--use_qnn_onnx_ep_converter" in compile_options
                )
            elif runtime == "qnn_dlc":
                matches = (
                    "--target_runtime qnn_dlc" in compile_options
                    and "--use_qnn_onnx_ep_converter" not in compile_options
                )
            else:
                matches = f"--target_runtime {runtime}" in compile_options

            if matches:
                filtered[model_name] = job_spec
                break

    logger.info(
        f"Filtered: {len(filtered)}/{len(scorecard)} jobs match runtime(s): {runtimes}"
    )
    return filtered


def filter_scorecard_by_device(scorecard: dict, devices: list[str]) -> dict:
    if not devices:
        return scorecard

    filtered = {}
    for model_name, job_spec in scorecard.items():
        device = job_spec.get("device", "")
        if device in devices:
            filtered[model_name] = job_spec

    logger.info(
        f"Filtered: {len(filtered)}/{len(scorecard)} jobs match device(s): {devices}"
    )
    return filtered


def format_input_specs(input_specs: dict) -> dict:
    return {name: (tuple(spec[0]), spec[1]) for name, spec in input_specs.items()}


def _get_qnn_compile_path(model_name: str) -> str | None:
    if "_QNN_DLC_VIA_QNN_EP" in model_name:
        return "QNN_DLC_VIA_QNN_EP"
    if "_QNN_DLC" in model_name:
        return "QNN_DLC"
    return None


def _get_precision(model_name: str, compile_path: str) -> str:
    model_and_precision = model_name.split(f"_{compile_path}", 1)[0]
    if match := re.search(
        r"_(w\d+a\d+(?:_mixed_(?:int16|fp16))?|mixed(?:_with_float)?)$",
        model_and_precision,
    ):
        return match.group(1)
    return "float"


def _make_tfl_converter_scorecard(scorecard: dict) -> dict:
    tfl_converter_scorecard = {}
    for model_name, job_spec in scorecard.items():
        compile_path = _get_qnn_compile_path(model_name)
        if compile_path is None:
            continue

        if _get_precision(model_name, compile_path) not in TFL_CONVERTER_PRECISIONS:
            continue

        compile_options = job_spec.get("compile_options", "")
        if "--target_runtime qnn_dlc" not in compile_options:
            continue

        converter_name = model_name.replace(f"_{compile_path}", "_TFL_CONVERTER", 1)
        if converter_name in scorecard or converter_name in tfl_converter_scorecard:
            continue

        converter_options = compile_options.replace(
            "--target_runtime qnn_dlc", "--target_runtime tflite", 1
        ).replace("--use_qnn_onnx_ep_converter", "")
        converter_options = merge_job_options(converter_options, TFL_CONVERTER_OPTION)

        tfl_converter_scorecard[converter_name] = {
            **job_spec,
            "compile_options": converter_options,
        }

    return tfl_converter_scorecard


def add_tfl_converter_jobs(scorecard: dict) -> dict:
    return {**scorecard, **_make_tfl_converter_scorecard(scorecard)}


def submit_single_compile_job(
    client: Client,
    model_name: str,
    job_spec: dict,
    project_id: str | None,
    extra_compiler_args: str | None,
) -> tuple[str, dict | None, Exception | None]:
    """Submit a single compile job.

    Returns
    -------
    tuple[str, dict | None, Exception | None]
        (model_name, job_result_dict, error)
        If successful, job_result_dict is populated and error is None.
        If failed, job_result_dict is None and error contains the exception.
    """
    try:
        logger.info(f"Submitting {model_name} (model_id: {job_spec['model_id']})")
        compile_options = merge_job_options(
            job_spec["compile_options"], extra_compiler_args
        )
        formatted_input_specs = format_input_specs(job_spec["input_specs"])

        model = client.get_model(job_spec["model_id"])
        device = Device(job_spec["device"])

        job = client.submit_compile_job(
            model=model,
            name=model_name,
            device=device,
            input_specs=formatted_input_specs,
            options=compile_options,
            project=project_id,
        )
        result = {
            "prod_job": job_spec["prod_job"],
            "prod_job_url": job_spec.get("prod_job_url", job_spec["prod_job"]),
            "prod_job_status": job_spec["prod_job_status"],
            "dev_job": job.job_id,
            "dev_job_url": job.url,
            "compile_options": compile_options,
        }
        return model_name, result, None
    except Exception as e:
        logger.exception(f"Failed to submit {model_name}")
        return model_name, None, e


def submit_compile_jobs(
    client: Client,
    scorecard: dict,
    project_id: str | None,
    extra_compiler_args: str | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Submit compile jobs in parallel.

    Returns
    -------
    tuple[dict[str, dict], dict[str, dict]]
        (job_map, failed_jobs)
        job_map: successful job submissions
        failed_jobs: dict mapping model_name to job_spec + error info
    """
    job_map = {}
    failed_jobs = {}
    logger.info(f"Submitting {len(scorecard)} compile jobs with {max_workers} workers")

    if extra_compiler_args:
        logger.info(f"Extra compiler arguments: {extra_compiler_args}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                submit_single_compile_job,
                client,
                model_name,
                job_spec,
                project_id,
                extra_compiler_args,
            ): model_name
            for model_name, job_spec in scorecard.items()
        }

        for future in as_completed(futures):
            model_name, result, error = future.result()
            if error is None and result is not None:
                job_map[model_name] = result
            else:
                failed_jobs[model_name] = {
                    **scorecard[model_name],
                    "error": str(error),
                }

    log_and_print(
        f"Successfully submitted {len(job_map)}/{len(scorecard)} jobs", logger
    )
    return job_map, failed_jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Submit compile jobs from a dev scorecard"
    )
    parser.add_argument(
        "--dev-profile",
        type=str,
        default="dev",
        help="Hub client profile for dev environment (default: dev)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dev-compile-scorecard.yaml file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--extra-compiler-args",
        type=str,
        default=None,
        help='Additional compiler arguments as CLI string (e.g., "--flag1 value1 --flag2")',
    )
    parser.add_argument(
        "--runtime",
        type=str,
        action="append",
        default=None,
        help="Filter by runtime(s). Can be specified multiple times. If not specified, runs all runtimes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        action="append",
        default=None,
        help="Filter by device(s). Can be specified multiple times. If not specified, runs all devices.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag used for output file identifier (default: current date)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Number of parallel workers for job submission (default: {DEFAULT_MAX_WORKERS})",
    )

    args = parser.parse_args()

    job_yaml_tag = args.tag or get_date_str()
    log_file = setup_script_logging(
        args.output_dir, "run-compile-jobs", args.verbose, job_yaml_tag
    )
    log_and_print(f"Full logs: {log_file}", logger)

    try:
        project_id = get_aihw_compiler_nightly_project()
        dev_client = load_client(args.dev_profile)
        scorecard = load_yaml_safe(args.config)
        log_and_print(f"Loaded {len(scorecard)} jobs from {args.config}", logger)
        scorecard = add_tfl_converter_jobs(scorecard)
        scorecard = filter_scorecard_by_runtime(scorecard, args.runtime)
        scorecard = filter_scorecard_by_device(scorecard, args.device)
        log_and_print(f"Launching {len(scorecard)} compile jobs", logger)
        job_map, failed_jobs = submit_compile_jobs(
            dev_client, scorecard, project_id, args.extra_compiler_args, args.workers
        )

        dev_jobs_file = args.output_dir / f"dev-compile-jobs__{job_yaml_tag}.yaml"
        save_yaml_results(job_map, dev_jobs_file)
        log_and_print(f"Saved to {dev_jobs_file}", logger)

        if failed_jobs:
            failed_jobs_file = (
                args.output_dir
                / f"dev-compile-jobs-submission-failures__{job_yaml_tag}.yaml"
            )
            save_yaml_results(failed_jobs, failed_jobs_file)
            log_and_print(
                f"Failed to create {len(failed_jobs)} compile job(s). Saved to {failed_jobs_file}",
                logger,
            )

        return 0

    except Exception:
        logger.exception("✗ Script failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
