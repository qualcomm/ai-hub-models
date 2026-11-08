# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Backend-agnostic Genie job orchestration: bundle staging, metric/eval parsing, submit/collect."""

from __future__ import annotations

import contextlib
import fnmatch
import importlib
import json
import logging
import os
import re
import shutil
import statistics
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from transformers import AutoTokenizer

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.datasets.device_eval import (
    DEFAULT_DEVICE_EVAL_DATASET,
    DeviceEvalItem,
    label_eval_results,
    load_device_eval_items,
)
from qai_hub_models.models.templates.llm.model import LLMBase
from qai_hub_models.models.templates.llm.perf_collection import (
    get_llm_eval_device,
    load_release_assets_for_model,
    record_perf_scope,
    update_perf_yaml,
)
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.scorecard.envvars import QAIRTVersionEnvvar
from qai_hub_models.scorecard.utils.fetch_prerelease_assets import (
    download_prerelease_asset,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    get_presigned_download_url,
    get_qaihm_s3,
)
from qai_hub_models.utils.devicefarm.devicefarm import (
    DeviceFarm,
    HubDevicePlatform,
    JobOutcome,
    JobRecord,
    device_logs_dir_name,
    get_device_farm,
    load_jobs,
    make_key,
    poll_job,
    save_job,
    walk_dir_entries,
)
from qai_hub_models.utils.llm.eval_io import (
    save_eval_metadata_json,
    save_eval_results_json,
)
from qai_hub_models.utils.llm.geniex.jobs import get_geniex_qairt_sdk_url

# Perf/job diagnostics go through logging, not print: nightly runs pytest under
# xdist, where worker stdout is dropped even with --capture=no. INFO is set
# explicitly because pytest leaves the root logger at WARNING.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# QDC submission no longer waits for a free device slot (it queues jobs
# server-side), so this status-poll budget now has to cover the queue wait as
# well as the on-device run, rather than just the run.
GENIE_JOB_TIMEOUT = 43200  # 12 hours

DEFAULT_LLM_SYSTEM_PROMPT = LLMBase.default_system_prompt

# QAIRT SDK downloaded on-device when no version is requested.
DEFAULT_GENIE_QAIRT_SDK_VERSION = "2.50.0.260828"


def _write_eval_prompts_to_dir(
    prompts_dir: str,
    prompts: list[str],
    genie_bundle_path: str,
    model_id: str | None = None,
) -> None:
    """Write chat-templated prompt files into ``prompts_dir``.

    Loads the tokenizer from the local Genie bundle: it matches what ships
    on-device, avoids gated-HF auth, and (with a current transformers)
    parses the newer list-of-dicts chat_template format correctly. Falls
    back to the model's HF repo (HF_REPO_NAME) only when the bundle path
    isn't available.

    Thinking mode is disabled for the eval prompts so the model returns a
    direct answer within the on-device token budget instead of spending it
    on a reasoning trace that may be truncated before any answer is produced.
    Passing enable_thinking=False is safe for non-thinking models -- their
    chat templates simply ignore the unused variable.
    """
    hf_repo: str | None = None
    if model_id:
        model_module = importlib.import_module(f"qai_hub_models.models.{model_id}")
        hf_repo = getattr(model_module, "HF_REPO_NAME", None)
    tokenizer = AutoTokenizer.from_pretrained(
        genie_bundle_path if genie_bundle_path else hf_repo
    )

    os.makedirs(prompts_dir, exist_ok=True)
    for idx, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": DEFAULT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        prompt_file = os.path.join(prompts_dir, f"prompt_{idx:03d}.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(formatted)  # type: ignore[arg-type, unused-ignore]


class GenieArtifactHandler(ABC):
    """Abstract base class for Genie artifact handlers."""

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        device_logs_dir: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
        genie_bundle_url: str | None = None,
    ) -> list[tuple[str, str]]:
        """Stage the on-device bundle and return (abs_path, arcname) entries.

        ``device_logs_dir`` is the on-device directory name (backend-specific;
        see :func:`device_logs_dir_name`) the rendered script writes its logs
        into.

        ``eval_prompts_dir`` (if provided) points at a small tmpdir holding
        the chat-templated ``prompt_NNN.txt`` files; the handler adds those
        to the entry list under the appropriate on-device path. Callers
        avoid duplicating the multi-GB genie bundle by keeping the prompt
        files in a separate directory from the bundle itself.

        ``genie_bundle_url`` (if provided) is a presigned S3 GET URL for the
        genie bundle zip; Linux/Windows curl it on-device instead of the
        caller shipping the bundle weights through the artifact zip. When
        None (e.g. auto/mobile carve-outs), the handler falls back to
        referencing ``genie_bundle_path`` directly via ``walk_dir_entries``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class GenieAndroidArtifactHandler(GenieArtifactHandler):
    def __init__(self, test_script: str) -> None:
        self.test_script: str = test_script

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        device_logs_dir: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
        genie_bundle_url: str | None = None,
    ) -> list[tuple[str, str]]:
        # Write only the small placeholder-substituted files (test_appium.py,
        # requirements.txt) to dest_dir. When genie_bundle_url is set, the
        # device itself curls+unzips the bundle from S3 -- the multi-GB local
        # genie_bundle_path is never referenced. Otherwise (no URL, e.g. the
        # Auto carve-out) fall back to referencing it directly via arcname
        # prefix -- copying it here would triple /scratch usage while staging.
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        test_appium_path = os.path.join(test_folder, "test_appium.py")
        with open(
            os.path.join(curr_dirname, "device_scripts", self.test_script),
            encoding="utf-8",
        ) as f:
            file_content = f.read()
        with open(test_appium_path, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("<<HEXAGON_VERSION>>", hexagon_version)
                .replace("<<QAIRT_VERSION>>", qairt_version)
                .replace("<<QAIRT_SDK_URL>>", get_geniex_qairt_sdk_url(qairt_version))
                .replace("<<NUM_TRIALS>>", str(num_trials))
                .replace("<<DEVICE_LOGS_DIR>>", device_logs_dir)
                .replace("<<GENIE_BUNDLE_URL>>", genie_bundle_url or "")
            )

        requirements_dest = os.path.join(dest_dir, "requirements.txt")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "requirements.txt"),
            requirements_dest,
        )

        entries: list[tuple[str, str]] = [
            (test_appium_path, os.path.join("tests", "test_appium.py")),
            (requirements_dest, "requirements.txt"),
        ]
        if genie_bundle_url:
            # Bundle weights come from the on-device curl; eval prompts (if
            # any) are small enough to still ship through the artifact zip,
            # under a path the device script copies into genie_bundle/prompts
            # after extracting the downloaded bundle.
            if eval_prompts_dir:
                entries.extend(
                    walk_dir_entries(eval_prompts_dir, arcname_prefix="eval_prompts")
                )
        else:
            entries.extend(
                walk_dir_entries(
                    os.fspath(genie_bundle_path), arcname_prefix="genie_bundle"
                )
            )
            if eval_prompts_dir:
                entries.extend(
                    walk_dir_entries(
                        eval_prompts_dir,
                        arcname_prefix=os.path.join("genie_bundle", "prompts"),
                    )
                )
        return entries


class GenieAutoArtifactHandler(GenieAndroidArtifactHandler):
    """Artifact handler for automotive (auto) devices.

    Extends the Android handler by bundling the QAIRT SDK into the artifact,
    since auto devices cannot download it at runtime.
    """

    def __init__(self, test_script: str, qairt_sdk_path: str) -> None:
        """
        Parameters
        ----------
        test_script
            Filename of the Appium/PyTest script to bundle (e.g., ``run_auto_android.py``).
        qairt_sdk_path
            Path to the QAIRT SDK zip file to bundle with the artifact.
            Must be an accessible, valid zip file.
        """
        super().__init__(test_script)
        if not os.path.isfile(qairt_sdk_path):
            raise FileNotFoundError(
                f"QAIRT SDK path '{qairt_sdk_path}' does not exist or is not a file. "
                "Please verify the --qairt-sdk-path argument."
            )
        self.qairt_sdk_path: str = qairt_sdk_path

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        device_logs_dir: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
        genie_bundle_url: str | None = None,
    ) -> list[tuple[str, str]]:
        # Carve-out: auto devices' network reachability to S3 is unconfirmed,
        # so this handler always ships the full bundle through the artifact
        # zip regardless of whether the caller minted a presigned URL.
        entries = super().create_artifact(
            curr_dirname,
            genie_bundle_path,
            dest_dir,
            hexagon_version,
            qairt_version,
            device_logs_dir,
            num_trials,
            eval_prompts_dir=eval_prompts_dir,
            genie_bundle_url=None,
        )
        entries.append(
            (self.qairt_sdk_path, os.path.join("genie_bundle", "qairt_sdk.zip"))
        )
        return entries


class GenieLinuxArtifactHandler(GenieArtifactHandler):
    """Artifact handler for Linux IoT devices (e.g., IQ9).
    Uses Bash test framework — no Appium wrapper needed.
    """

    @property
    def entry_script(self) -> str:
        return "/bin/bash /data/local/tmp/TestContent/run_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        device_logs_dir: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
        genie_bundle_url: str | None = None,
    ) -> list[tuple[str, str]]:
        # Write the version-substituted run_linux.sh into dest_dir; reference
        # the multi-GB genie bundle via arcname prefix rather than copying it
        # -- see GenieAndroidArtifactHandler for context.
        script_name = "run_linux.sh"
        script_dest = os.path.join(dest_dir, script_name)
        with open(
            os.path.join(curr_dirname, "device_scripts", script_name),
            encoding="utf-8",
        ) as f:
            file_content = f.read()
        with open(script_dest, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version)
                .replace("{QAIRT_VERSION}", qairt_version)
                .replace("{QAIRT_SDK_URL}", get_geniex_qairt_sdk_url(qairt_version))
                .replace("{NUM_TRIALS}", str(num_trials))
                .replace("{DEVICE_LOGS_DIR}", device_logs_dir)
            )

        entries: list[tuple[str, str]] = [(script_dest, script_name)]
        entries.extend(
            walk_dir_entries(
                os.fspath(genie_bundle_path), arcname_prefix="genie_bundle"
            )
        )
        if eval_prompts_dir:
            entries.extend(
                walk_dir_entries(
                    eval_prompts_dir,
                    arcname_prefix=os.path.join("genie_bundle", "prompts"),
                )
            )
        return entries


class GenieWindowsArtifactHandler(GenieArtifactHandler):
    @property
    def entry_script(self) -> str:
        return "C:\\Temp\\TestContent\\run_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        device_logs_dir: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
        genie_bundle_url: str | None = None,
    ) -> list[tuple[str, str]]:
        # Windows layout: run_windows.ps1 + bundle contents live at the top
        # of the zip (no genie_bundle/ prefix). Substitute placeholders in
        # the script; reference the bundle via arcnames without copying it.
        script_name = "run_windows.ps1"
        dest_script = os.path.join(dest_dir, script_name)
        with open(
            os.path.join(curr_dirname, "device_scripts", script_name),
            encoding="utf-8",
        ) as f:
            file_content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version)
                .replace("{QAIRT_VERSION}", qairt_version)
                .replace("{QAIRT_SDK_URL}", get_geniex_qairt_sdk_url(qairt_version))
                .replace("{NUM_TRIALS}", str(num_trials))
                .replace("{DEVICE_LOGS_DIR}", device_logs_dir)
            )

        entries: list[tuple[str, str]] = [(dest_script, script_name)]
        entries.extend(walk_dir_entries(os.fspath(genie_bundle_path)))
        if eval_prompts_dir:
            entries.extend(walk_dir_entries(eval_prompts_dir, arcname_prefix="prompts"))
        return entries


def _get_artifact_handler(
    platform: HubDevicePlatform,
    qairt_sdk_path: str | None = None,
) -> GenieArtifactHandler:
    """Get the appropriate artifact handler based on device platform.

    Parameters
    ----------
    platform
        HubDevicePlatform instance (passed to avoid redundant instantiation).
    qairt_sdk_path
        Path to the QAIRT SDK zip file. Required for auto devices.

    Returns
    -------
    genie_artifact_handler: GenieArtifactHandler
        Instance of the appropriate GenieArtifactHandler subclass.
    """
    if platform.is_windows_platform:
        return GenieWindowsArtifactHandler()
    if platform.is_iot_platform:
        return GenieLinuxArtifactHandler()
    if platform.is_auto_platform:
        if qairt_sdk_path is None:
            raise ValueError(
                "qairt_sdk_path is required for auto devices. "
                "Please provide the path to the automotive QAIRT SDK zip file."
            )
        return GenieAutoArtifactHandler(
            test_script="run_auto_android.py", qairt_sdk_path=qairt_sdk_path
        )
    if platform.is_mobile_platform:
        return GenieAndroidArtifactHandler(test_script="run_android.py")
    raise ValueError("Unsupported platform type for Genie artifact handler.")


def add_genie_bundle_entries(
    platform: HubDevicePlatform,
    genie_bundle_path: str,
    dest_dir: str,
    device_logs_dir: str,
    qairt_sdk_path: str | None = None,
    qairt_version: str | None = None,
    eval_prompts: list[str] | None = None,
    num_trials: int = 25,
    model_id: str | None = None,
    genie_bundle_url: str | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Stage a Genie bundle into backend-agnostic (path, arcname) entries.

    Parameters
    ----------
    platform
        HubDevicePlatform for the target device.
    genie_bundle_path
        Directory path containing the genie bundle.
    dest_dir
        Directory to stage small placeholder-substituted files into (test
        scripts, requirements.txt, prompt files). Must outlive the returned
        entries -- callers zip/upload them before it is cleaned up.
    device_logs_dir
        On-device log directory name (see :func:`device_logs_dir_name`)
        rendered into the bundle's device scripts.
    qairt_sdk_path
        Path to the QAIRT SDK zip file. Required for auto devices.
    qairt_version
        Software Center QAIRT SDK version to download on-device (e.g.
        ``"2.50.0.260828"``). None uses ``DEFAULT_GENIE_QAIRT_SDK_VERSION``.
    eval_prompts
        If provided, list of prompts to evaluate. Each prompt is formatted
        using the bundle's tokenizer and run sequentially on device.
    num_trials
        Number of profiling trials to run.
    model_id
        Model identifier used to load the HF tokenizer if the bundle
        tokenizer lacks a chat template.
    genie_bundle_url
        Presigned S3 GET URL for the genie bundle zip. When provided, the
        device curls+extracts the bundle itself instead of the caller
        shipping it through the artifact zip. Ignored by handlers that carve
        themselves out of this (e.g. auto -- see ``GenieAutoArtifactHandler``).

    Returns
    -------
    entries: list[tuple[str, str]]
        (abs_path, arcname) pairs a backend zips and uploads as one artifact.
    entry_script: str | None
        Optional entry script path used by the test framework.
    """
    qairt_version = qairt_version or DEFAULT_GENIE_QAIRT_SDK_VERSION
    curr_dirname = os.path.dirname(os.path.abspath(__file__))
    artifact_handler = _get_artifact_handler(platform, qairt_sdk_path)

    # Write chat-templated eval prompts into a subdir of dest_dir (~100 KB of
    # .txt files) rather than cloning the multi-GB genie bundle just to add a
    # prompts/ subdir. Nested under dest_dir so it stays alive exactly as long
    # as the entries the handler references -- callers zip/upload them before
    # dest_dir is cleaned up.
    eval_prompts_dir: str | None = None
    if eval_prompts:
        eval_prompts_dir = os.path.join(dest_dir, "_eval_prompts_staging")
        _write_eval_prompts_to_dir(
            eval_prompts_dir, eval_prompts, genie_bundle_path, model_id
        )

    entries = artifact_handler.create_artifact(
        curr_dirname,
        genie_bundle_path,
        dest_dir,
        platform.hexagon_version,
        qairt_version,
        device_logs_dir,
        num_trials,
        eval_prompts_dir=eval_prompts_dir,
        genie_bundle_url=genie_bundle_url,
    )
    return entries, artifact_handler.entry_script


def compute_genie_metrics(
    backend: DeviceFarm,
    job_log_files: list,
) -> tuple[float | None, float | None, float | None]:
    """Compute and print performance metrics from job logs.

    Parameters
    ----------
    backend
        Device-farm backend able to fetch one named log file, zip-wrapped,
        into a local path.
    job_log_files
        List of job log files retrieved from the backend.

    Returns
    -------
    avg_tokens_per_second : float | None
        Average tokens per second.
    min_time_to_first_token: float | None
        Minimum time to first token in ms.
    prefill_tokens_per_second : float | None
        Prefill (prompt-processing) tokens per second.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tps: list[float] = []
        ttft: list[float] = []
        prefill_tps: list[float] = []

        if job_log_files:
            for job_log in job_log_files:
                target_path = os.path.join(
                    tmpdirname, "logs", f"{job_log.filename}.zip"
                )
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                if not backend.try_download_job_log_files(
                    job_log.filename, target_path
                ):
                    continue

                if "genie" in job_log.filename:
                    shutil.unpack_archive(target_path, tmpdirname, "zip")
                    genie_log_path = os.path.join(tmpdirname, "genie.log")
                    displayed = False
                    for encoding in ("utf-8", "utf-16", "utf-16-le"):
                        try:
                            with open(genie_log_path, encoding=encoding) as file:
                                genie_content = file.read()
                                logger.info(
                                    "On device output (genie.log):\n%s",
                                    genie_content,
                                )
                                displayed = True
                                break
                        except Exception:
                            pass
                    if not displayed:
                        logger.warning("Could not read %s", genie_log_path)

                if fnmatch.fnmatch(os.path.basename(job_log.filename), "profile*.json"):
                    shutil.unpack_archive(target_path, tmpdirname, "zip")
                    profile_path = os.path.join(
                        tmpdirname, job_log.filename.split("/")[-1]
                    )
                    with open(profile_path, encoding="utf-8") as file:
                        file_content = json.loads(file.read())

                    components = file_content.get("components", [])
                    if (
                        isinstance(components, list)
                        and len(components) > 0
                        and isinstance(components[0], dict)
                        and "events" in components[0]
                        and isinstance(components[0]["events"], list)
                        and len(components[0]["events"]) > 1
                    ):
                        component = components[0]["events"][1]
                        tps.append(float(component["token-generation-rate"]["value"]))
                        ttft.append(float(component["time-to-first-token"]["value"]))
                        prefill_tps.append(
                            float(component["prompt-processing-rate"]["value"])
                        )
                    else:
                        logger.warning(
                            "Unexpected profile log structure in %s, "
                            "skipping metrics for this file.",
                            profile_path,
                        )

    if len(tps) > 0:
        # TTFT in profile logs is in microseconds, convert to milliseconds
        ttft_ms = [t / 1000.0 for t in ttft]

        logger.info("Perf metrics:")
        logger.info("  Tokens Per Second (all trials): %s", tps)
        logger.info("  Time to First Token ms (all trials): %s", ttft_ms)
        logger.info("  Prefill Tokens Per Second (all trials): %s", prefill_tps)
        logger.info(
            "  Tokens Per Second — average: %.2f, median: %.2f",
            statistics.mean(tps),
            statistics.median(tps),
        )
        logger.info(
            "  Time to First Token (ms) — average: %.2f, median: %.2f",
            statistics.mean(ttft_ms),
            statistics.median(ttft_ms),
        )
        logger.info(
            "  Prefill Tokens Per Second — average: %.2f, median: %.2f",
            statistics.mean(prefill_tps),
            statistics.median(prefill_tps),
        )
        return (
            statistics.median(tps),
            statistics.median(prefill_tps),
            statistics.median(ttft_ms),
        )

    logger.error("No performance metrics found.")
    if job_log_files:
        logger.error(
            "Available log files:\n%s",
            "\n".join(f"  {job_log.filename}" for job_log in job_log_files),
        )
    return None, None, None


def _parse_eval_outputs(content: str) -> dict[int, str]:
    """Parse a single eval_outputs.txt file with delimiter markers.

    Format: ===EVAL_IDX_NNN=== followed by the model output for that prompt.
    """
    outputs: dict[int, str] = {}
    parts = re.split(r"===EVAL_IDX_(\d+)===\n?", content)
    for i in range(1, len(parts) - 1, 2):
        idx = int(parts[i])
        outputs[idx] = parts[i + 1].strip()
    return outputs


def _extract_model_output(raw_output: str) -> str:
    """Extract just the model's response from raw genie-t2t-run output.

    The raw output mixes debug logs, the chat-templated prompt echo, and
    the actual response between ``[BEGIN]:`` and ``[END]`` markers. We
    return only the text between those markers; if neither is present,
    fall back to the raw output stripped.
    """
    begin_marker = "[BEGIN]:"
    end_marker = "[END]"
    begin_idx = raw_output.find(begin_marker)
    if begin_idx == -1:
        return raw_output.strip()
    text = raw_output[begin_idx + len(begin_marker) :]
    end_idx = text.find(end_marker)
    if end_idx != -1:
        text = text[:end_idx]
    return text.strip()


def compute_genie_eval_results(
    backend: DeviceFarm,
    job_log_files: list,
    prompts: list[str],
) -> list[dict]:
    """Parse eval outputs from job logs.

    The device scripts write a single eval_outputs.txt file with
    delimiter markers (===EVAL_IDX_NNN===) separating each prompt's
    output.

    Parameters
    ----------
    backend
        Device-farm backend able to fetch one named log file, zip-wrapped,
        into a local path.
    job_log_files
        List of job log files retrieved from the backend.
    prompts
        Original list of prompts (used to attach prompt text to results).

    Returns
    -------
    results: list[dict]
        List of dicts with keys: idx, prompt, output.
    """
    outputs: dict[int, str] = {}

    with tempfile.TemporaryDirectory() as tmpdirname:
        for job_log in job_log_files:
            if "eval_outputs" not in job_log.filename:
                continue

            target_path = os.path.join(tmpdirname, "logs", f"{job_log.filename}.zip")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            if not backend.try_download_job_log_files(job_log.filename, target_path):
                continue

            safe_root = Path(tmpdirname).resolve()
            with zipfile.ZipFile(target_path) as zf:
                for member in zf.namelist():
                    dest = (safe_root / member).resolve()
                    if not str(dest).startswith(str(safe_root) + os.sep):
                        raise ValueError(f"Zip slip detected in log archive: {member}")
                zf.extractall(safe_root)

            extracted_name = job_log.filename.split("/")[-1]
            extracted_path = os.path.join(tmpdirname, extracted_name)
            if not os.path.exists(extracted_path):
                continue

            content = None
            for encoding in ("utf-8", "utf-16", "utf-16-le"):
                try:
                    with open(extracted_path, encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    pass

            if content is None:
                logger.warning("Could not decode %s", extracted_name)
                continue

            outputs = _parse_eval_outputs(content)

    results: list[dict] = [
        {
            "idx": idx,
            "prompt": prompts[idx] if idx < len(prompts) else "",
            "output": _extract_model_output(outputs.get(idx, "")),
        }
        for idx in sorted(outputs.keys())
    ]

    if not results:
        logger.warning("No eval results found in job logs.")
        logger.warning(
            "Available log files:\n%s",
            "\n".join(f"  {job_log.filename}" for job_log in job_log_files),
        )

    return results


# Kept for callers that predate named datasets: it now just names the default one.
_USE_DEFAULT_PROMPTS = DEFAULT_DEVICE_EVAL_DATASET

# A device-eval dataset name, an explicit prompt list, or None for no eval.
EvalPrompts = str | list[str] | None


def _resolve_eval_items(eval_prompts: EvalPrompts) -> list[DeviceEvalItem] | None:
    if isinstance(eval_prompts, str):
        return load_device_eval_items(eval_prompts)
    if isinstance(eval_prompts, list):
        return [DeviceEvalItem(idx, prompt) for idx, prompt in enumerate(eval_prompts)]
    return None


def submit_genie_bundle(
    backend: DeviceFarm,
    hub_device_name: str,
    genie_bundle_path: str,
    job_name: str = "LLM Genie",
    qairt_sdk_path: str | None = None,
    qairt_version: str | None = None,
    eval_prompts: EvalPrompts = None,
    num_trials: int = 25,
    model_id: str | None = None,
    genie_bundle_url: str | None = None,
) -> str:
    """Upload artifacts and submit a Genie job, returning the backend job id.

    Companion to :func:`collect_genie_bundle`. Does no waiting or result
    parsing -- the caller records the job id (typically to a jobs_file) and
    polls later. ``genie_bundle_url``, when provided, is a presigned S3 GET
    URL the device curls the bundle from directly, instead of the caller
    shipping the (potentially multi-GB) bundle through the artifact zip.
    """
    items = _resolve_eval_items(eval_prompts)
    prompts_to_use = [item.prompt for item in items] if items else None
    platform = HubDevicePlatform(hub_device_name)
    device_logs_dir = device_logs_dir_name(backend)

    # Staging dir must outlive add_genie_bundle_entries: the returned entries
    # reference files inside it, and submit_bundle needs them to still exist
    # when it zips/uploads.
    with tempfile.TemporaryDirectory(prefix="genie_artifact_staging_") as dest_dir:
        entries, entry_script = add_genie_bundle_entries(
            platform,
            genie_bundle_path,
            dest_dir,
            device_logs_dir,
            qairt_sdk_path,
            qairt_version,
            eval_prompts=prompts_to_use,
            num_trials=num_trials,
            model_id=model_id,
            genie_bundle_url=genie_bundle_url,
        )

        # No explicit timeout: submission itself doesn't block on either
        # backend (QDC queues server-side; AWS's own submit_bundle default
        # already reflects its execution cap, which AWS limits to 150
        # minutes) -- the wait now lives entirely in collect_genie_bundle's
        # status() poll (see GENIE_JOB_TIMEOUT).
        job_id = backend.submit_bundle(
            hub_device_name,
            entries,
            entry_script,
            job_name=job_name,
        )
    if job_id is None:
        raise RuntimeError("Job submission failed.")
    logger.info("Submitted job with ID: %s", job_id)
    return job_id


def collect_genie_bundle(
    backend: DeviceFarm,
    hub_device_name: str,
    job_id: str,
    eval_prompts: EvalPrompts = None,
    save_logs_dir: str | None = None,
    log_label: str | None = None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    list[dict],
    JobOutcome,
    str | None,
]:
    """Poll a submitted Genie job and, on success, download + parse logs.

    Returns ``(tps, prefill_tps, ttft, eval_results, outcome, reason)``.
    On non-SUCCESS outcomes, the metric fields are None and ``reason`` is
    a human-readable failure description. eval_prompts is only consulted
    on success to attach prompt text to the parsed outputs. ``save_logs_dir``,
    when set, is where every job's logs are archived for diagnosis (success and
    failure alike), named by ``log_label``.
    """
    items = _resolve_eval_items(eval_prompts)

    job_status = backend.status(job_id, timeout=GENIE_JOB_TIMEOUT)
    job_result = backend.result(job_id)
    logger.info(
        "Job %s completed with status: %s, result: %s",
        job_id,
        job_status,
        job_result,
    )

    if not backend.is_successful(job_result):
        reason = (
            f"Job {job_id} on device '{hub_device_name}' finished with "
            f"status='{job_status}', result='{job_result}'"
        )
        outcome = backend.classify_failure(job_result)
        logger.error("[result=%s] %s", job_result, reason)
        backend.save_job_logs(job_id, save_logs_dir, label=log_label)
        return None, None, None, [], outcome, reason

    backend.log_upload_status(job_id)
    # The file listing can lag the log-upload signal on some backends, so wait
    # for it to populate -- otherwise a successful job yields no metrics.
    job_log_files = backend.get_job_log_files(job_id, wait_for_logs=True)

    if not job_log_files:
        reason = (
            f"Job {job_id} on device '{hub_device_name}' reported result="
            f"'{job_result}' but produced no retrievable log files"
        )
        logger.error("[empty logs] %s", reason)
        return None, None, None, [], JobOutcome.RETRYABLE_EMPTY_LOGS, reason

    # Archive before parsing so a green job's logs are kept too -- they are the
    # baseline a later failure gets read against.
    backend.save_job_logs(job_id, save_logs_dir, job_log_files, label=log_label)

    tps, prefill_tps, ttft = compute_genie_metrics(backend, job_log_files)

    eval_results: list[dict] = []
    if items:
        eval_results = label_eval_results(
            compute_genie_eval_results(
                backend, job_log_files, [item.prompt for item in items]
            ),
            items,
        )

    if tps is None:
        reason = (
            f"Job {job_id} on device '{hub_device_name}' reported result="
            f"'{job_result}' but its logs contained no parseable performance "
            f"metrics (missing/unreadable profile*.json)"
        )
        logger.error("[no metrics] %s", reason)
        return None, None, None, eval_results, JobOutcome.RETRYABLE_EMPTY_LOGS, reason

    return tps, prefill_tps, ttft, eval_results, JobOutcome.SUCCESS, None


def submit_and_collect_genie_bundle(
    backend: DeviceFarm,
    hub_device_name: str,
    genie_bundle_path: str,
    job_name: str = "LLM Genie",
    qairt_sdk_path: str | None = None,
    qairt_version: str | None = None,
    eval_prompts: EvalPrompts = None,
    num_trials: int = 25,
    model_id: str | None = None,
    save_logs_dir: str | None = None,
) -> tuple[float | None, float | None, float | None, list[dict]]:
    """
    Submit a Genie bundle and wait for the result, in one call.

    Runs profiling and (optionally) evaluation in a single job. Eval is
    skipped by default; pass a device-eval dataset name (e.g. ``"grace2"``)
    to run that set, or a list of prompts to use a custom set.

    Composed wrapper over ``submit_genie_bundle`` + ``collect_genie_bundle``
    for manual/on-device-test callers that want a single submit-and-wait
    call rather than the jobs_file-driven submit/collect split the CI
    perf-collection path uses.
    """

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        tps, prefill_tps, ttft, eval_results, outcome, reason = collect_genie_bundle(
            backend, hub_device_name, job_id, eval_prompts, save_logs_dir
        )
        return (tps, prefill_tps, ttft, eval_results), outcome, reason

    job_id = submit_genie_bundle(
        backend,
        hub_device_name,
        genie_bundle_path,
        job_name=job_name,
        qairt_sdk_path=qairt_sdk_path,
        qairt_version=qairt_version,
        eval_prompts=eval_prompts,
        num_trials=num_trials,
        model_id=model_id,
    )
    return poll_job(job_id, _collect)


# ---- CI perf-collection harness: jobs_file bookkeeping, perf.yaml updates --

GENIE_BUNDLES_ROOT = "genie_bundles"


def fetch_genie_bundle_for_perf(
    model_id: str,
    precision: Precision,
    chipset: str,
    output_dir: Path,
) -> Path:
    """Download and extract the pre-compiled genie bundle for this model.

    Looks up release-assets.yaml for (precision, chipset, GENIE), downloads
    the zip from S3, and extracts it into output_dir. Returns the extracted
    bundle directory.

    Raises a clear error if no matching asset exists.
    """
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIE)
    if asset is None:
        available_chipsets: list[str] = []
        prec_details = assets.precisions.get(precision)
        if prec_details is not None:
            available_chipsets = sorted(prec_details.chipset_assets.keys())
        raise RuntimeError(
            f"No genie release asset found in release-assets.yaml for "
            f"model_id={model_id!r}, precision={precision!s}, chipset={chipset!r}. "
            f"Available chipsets for this precision: {available_chipsets or '<none>'}. "
            "Build and update release-assets.yaml before running LLM perf collection."
        )

    bundle_dir = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_id, TargetRuntime.GENIE, precision, chipset
    )
    if bundle_dir.exists():
        # Already fetched on a previous test in this session.
        return bundle_dir

    zip_path = download_prerelease_asset(
        asset,
        model_id=model_id,
        runtime=TargetRuntime.GENIE,
        precision=precision,
        chipset=chipset,
        output_folder=output_dir,
        verbose=True,
    )
    shutil.unpack_archive(str(zip_path), extract_dir=str(output_dir))
    zip_path.unlink()
    if not bundle_dir.exists():
        raise RuntimeError(
            f"Extracted genie bundle missing expected directory {bundle_dir}; "
            f"contents of {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
        )
    return bundle_dir


def _bundle_context_lengths(bundle_dir: Path) -> list[int]:
    metadata = ModelMetadata.from_json(bundle_dir / "metadata.json")
    assert metadata is not None and metadata.genie is not None
    return list(metadata.genie.context_lengths)


def get_genie_bundle_url(model_id: str, precision: Precision, chipset: str) -> str:
    """Mint a presigned S3 GET URL for the current genie release asset.

    Re-resolves the asset fresh (same lookup as :func:`fetch_genie_bundle_for_perf`)
    so a stale/failed compile is never signed -- always whatever
    ``load_release_assets_for_model`` currently selects.
    """
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIE)
    if asset is None or asset.s3_key is None:
        raise RuntimeError(
            f"No genie release asset for {model_id!r} precision={precision} "
            f"chipset={chipset!r}; cannot mint a presigned download URL."
        )
    bucket = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
    return get_presigned_download_url(bucket, asset.s3_key)


def submit_llm_perf_job(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    output_dir: Path | str,
    jobs_file: str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
    eval_dataset: str = DEFAULT_DEVICE_EVAL_DATASET,
) -> str:
    """Fetch the genie bundle, submit one device-farm job, upsert its record.

    Does not wait. Returns the job id. The bundle is deleted once submitted
    (its upload is done by then); the record keeps the context lengths so
    collect only re-fetches the bundle if it has to resubmit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genie_bundle_path = fetch_genie_bundle_for_perf(
        model_id, precision, device.chipset, output_dir
    )
    try:
        context_lengths = _bundle_context_lengths(genie_bundle_path)
        genie_bundle_url = get_genie_bundle_url(model_id, precision, device.chipset)

        eval_prompts = eval_dataset if device == get_llm_eval_device() else None
        job_name = f"Genie {model_id} {precision}"

        backend = get_device_farm(device)
        job_id = submit_genie_bundle(
            backend,
            device.reference_device.name,
            str(genie_bundle_path),
            job_name=job_name,
            qairt_sdk_path=qairt_sdk_path,
            qairt_version=QAIRTVersionEnvvar.get_on_device_sdk_version(
                TargetRuntime.GENIE
            ),
            eval_prompts=eval_prompts,
            model_id=model_id,
            genie_bundle_url=genie_bundle_url,
        )
    finally:
        shutil.rmtree(genie_bundle_path, ignore_errors=True)

    key = make_key(model_id, str(precision), "GENIE", device.name)
    save_job(jobs_file, key, job_id, context_lengths=context_lengths)
    return job_id


def collect_llm_perf_job(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    record: JobRecord,
    jobs_file: str,
    output_dir: Path | str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
    eval_dataset: str = DEFAULT_DEVICE_EVAL_DATASET,
) -> tuple[float | None, float | None, float | None]:
    """Poll a submitted Genie job.

    ``record`` is the current entry read from jobs_file. Everything else is
    re-derived from (model_id, precision, device) so the collect runner is
    independent of whatever filesystem state the submit runner had.
    """
    if not skip_perf_update:
        record_perf_scope(
            model_id=model_id,
            profile_path=ScorecardProfilePath.GENIE,
            device_name=device.reference_device_name,
            precision=precision,
        )

    eval_prompts = eval_dataset if device == get_llm_eval_device() else None
    key = make_key(model_id, str(precision), "GENIE", device.name)
    hub_device_name = device.reference_device.name
    backend = get_device_farm(device)
    context_lengths = record.context_lengths

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        tps, prefill_tps, ttft, eval_results, outcome, reason = collect_genie_bundle(
            backend,
            hub_device_name,
            job_id,
            eval_prompts=eval_prompts,
            save_logs_dir=os.path.join(output_dir, "device_logs"),
            log_label=f"{key}_{job_id}",
        )
        return (tps, prefill_tps, ttft, eval_results), outcome, reason

    tps, prefill_tps, ttft, eval_results = poll_job(record.job_id, _collect)

    if context_lengths is None:
        # Jobs file written before submit recorded context_lengths.
        bundle_path = fetch_genie_bundle_for_perf(
            model_id, precision, device.chipset, Path(output_dir)
        )
        try:
            context_lengths = _bundle_context_lengths(bundle_path)
        finally:
            shutil.rmtree(bundle_path, ignore_errors=True)

    if not skip_perf_update and tps is not None:
        # Genie runs a single measurement, reported at the largest exported length.
        update_perf_yaml(
            model_id,
            device.reference_device_name,
            precision,
            max(context_lengths),
            tps,
            prefill_tps=prefill_tps,
        )

    if eval_results:
        base = f"{model_id}_{device.chipset}_{precision}_eval"
        save_eval_results_json(eval_results, f"{base}.json")
        save_eval_metadata_json(
            model_id,
            device.chipset,
            str(precision),
            f"{base}.meta.json",
            path=ScorecardProfilePath.GENIE,
            dataset_name=eval_dataset,
        )

    return tps, ttft, prefill_tps


def run_llm_perf_test(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    output_dir: Path | str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Compose submit + collect over an ephemeral jobs_file.

    Returns (tokens_per_second, time_to_first_token_ms, prefill_tokens_per_second).
    """
    with tempfile.NamedTemporaryFile(
        prefix="genie_jobs_", suffix=".yaml", delete=False
    ) as tmp:
        jobs_file = tmp.name
    try:
        submit_llm_perf_job(
            model_id=model_id,
            device=device,
            precision=precision,
            output_dir=output_dir,
            jobs_file=jobs_file,
            qairt_sdk_path=qairt_sdk_path,
            skip_perf_update=skip_perf_update,
        )
        key = make_key(model_id, str(precision), "GENIE", device.name)
        record = load_jobs(jobs_file)[key]
        return collect_llm_perf_job(
            model_id=model_id,
            device=device,
            precision=precision,
            record=record,
            jobs_file=jobs_file,
            output_dir=output_dir,
            qairt_sdk_path=qairt_sdk_path,
            skip_perf_update=skip_perf_update,
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(jobs_file)
