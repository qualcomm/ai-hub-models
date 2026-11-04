# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Backend-agnostic GenieX-bench job orchestration: bundle staging, metric/eval parsing, submit/collect."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

from qai_hub_models import Precision
from qai_hub_models.models.templates.llm.model import LLMBase
from qai_hub_models.models.templates.llm.perf_collection import (
    load_release_assets_for_model,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    QAIHM_PUBLIC_S3_BUCKET,
    get_presigned_download_url,
    get_qaihm_s3,
    s3_file_exists,
)
from qai_hub_models.utils.devicefarm.devicefarm import (
    FAILED_JOB_LOG_TIMEOUT,
    DeviceFarm,
    HubDevicePlatform,
    JobOutcome,
    device_logs_dir_name,
    walk_dir_entries,
)

DEFAULT_LLM_SYSTEM_PROMPT = LLMBase.default_system_prompt

PERF_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "perf_prompts"
)

# QDC submission no longer waits for a free device slot (it queues jobs
# server-side), so this status-poll budget now has to cover the queue wait as
# well as the on-device run, rather than just the run.
GENIEX_BENCH_JOB_TIMEOUT = 43200  # 12 hours

# Versioned URLs follow the geniex release workflow's flat S3 layout
# (<stem>-<vX.Y.Z>.<ext>); the unversioned mirror is refreshed on every
# stable tag and is used when no version is pinned.
_S3_BASE = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex"


def _bench_url(platform_stem: str, ext: str, version: str | None) -> str:
    # Callers pass bare SemVer ("0.3.7"); S3 filenames are v-prefixed.
    suffix = f"-v{version}" if version else ""
    return f"{_S3_BASE}/geniex-bench-{platform_stem}{suffix}.{ext}"


DEFAULT_CONTEXT_LENGTHS = [512, 1024, 4096]

_MIN_RESULT_SCHEMA_VERSION = 3

_N_GEN = 128
QAIRT_SDK_S3_PREFIX = "qai-hub-models/qairt-sdk"

# Each accuracy prompt runs as its own geniex-bench process: the timeout kills a
# crashed/hung DSP run so the loop advances, the sleep lets the DSP release
# before the next process attaches.
_EVAL_N_GEN = 2048
_EVAL_TIMEOUT_S = 600
_EVAL_SLEEP_S = 10
_SELF_ABORT_RE = re.compile(r"\[warn\]\s*self-abort: reason=([\w-]+)")
_STOP_REASON_RE = re.compile(r"\[info\]\s*stop_reason=([\w-]+)")
_EVAL_MAX_GEN_TIME_S = 420
_EVAL_NO_PROGRESS_TIMEOUT_S = 120
_EVAL_REPETITION_MAX_REPEATS = 40
_EVAL_NGRAM_MAX_REPEATS = 4

# Matches Genie's on-device eval seed (genie_config.json's sampler default,
# see llm_helpers.create_genie_config) -- pinned explicitly rather than
# relying on geniex-bench's own CLI default, which could drift independently.
_DEFAULT_EVAL_SEED = 42

# geniex-bench result schemas this parser reads. 4 kept the fields 3 exposed and
# only added some, so both parse identically. 5 added qairt_version/
# llama_cpp_version (unread until 6). 6 added geniex_version. An unlisted
# version is reported by compute_metrics rather than dropped, because a silent
# skip reads as "the device produced nothing" and now costs the committed perf
# rows.
_SUPPORTED_BENCH_SCHEMAS = frozenset({"3", "4", "5", "6"})


@dataclass
class GenieXBenchMetrics:
    cell_id: str
    plugin: str
    device_alias: str
    context_length: int
    ttft_ms: float
    prefill_tps: float
    decode_tps: float
    prompt_tokens: int
    gen_tokens: int
    geniex_version: str | None = None
    qairt_version: str | None = None
    llama_cpp_version: str | None = None
    stop_reason: str | None = None

    @property
    def zero_rate_note(self) -> str | None:
        """Why a reported rate is 0 (e.g. EOS as the first token), else None."""
        if self.ttft_ms > 0 and self.prefill_tps > 0 and self.decode_tps > 0:
            return None
        return (
            f"zero_rate (gen_tokens={self.gen_tokens}, stop_reason={self.stop_reason})"
        )


class GenieXBenchArtifactHandler(ABC):
    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        qairt_bundle_urls: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
        device_logs_dir: str,
        qairt_version: str | None,
        eval_seed: int = _DEFAULT_EVAL_SEED,
    ) -> list[tuple[str, str]]:
        """Stage the on-device bundle into ``dest_dir`` and return its entries.

        ``qairt_bundle_urls`` (if provided), keyed the same as ``qairt_bundles``,
        holds a presigned S3 GET URL per model so a handler can curl the bundle
        on-device instead of the caller shipping its weights through the
        artifact zip. Handlers that don't support this (e.g. Linux/Windows,
        which only ever run through QDC, not the AWS backend with its upload
        cap) may ignore it and fall back to ``_stage_qairt_bundles``.
        """
        # device_logs_dir is the on-device log directory name (see
        # device_logs_dir_name); unused by handlers whose device script has no
        # {DEVICE_LOGS_DIR} placeholder to fill (e.g. Windows, which still
        # writes its own fixed QDC_Logs).
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError

    @staticmethod
    def _stage_eval_prompts(
        dest_dir: os.PathLike | str, eval_prompts: list[str]
    ) -> None:
        """Write raw prompts to prompts/prompt_NNN.txt in the artifact.

        The device script feeds each file to its own `geniex-bench --accuracy
        --prompt-file` process, which now applies the bundle's own chat
        template on-device; presence of the directory is what turns eval on.
        """
        prompts_dir = os.path.join(dest_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        for idx, prompt in enumerate(eval_prompts):
            with open(
                os.path.join(prompts_dir, f"prompt_{idx:03d}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(prompt)

    @staticmethod
    def _stage_perf_prompts(
        dest_dir: os.PathLike | str,
        context_lengths: list[int],
    ) -> None:
        """Stage sample_prompt_<ctx>.txt into perf_prompts/ for qairt timing runs.

        Each ctx uses the largest shipped prompt size <= ctx (smallest if below
        all). Kept apart from prompts/, whose presence turns accuracy eval on.
        """
        src_dir = PERF_PROMPTS_DIR
        sizes = sorted(
            int(m.group(1))
            for fn in os.listdir(src_dir)
            if (m := re.fullmatch(r"sample_prompt_(\d+)\.txt", fn))
        )
        dest_prompts = os.path.join(dest_dir, "perf_prompts")
        os.makedirs(dest_prompts, exist_ok=True)
        for ctx in context_lengths:
            size = max((s for s in sizes if s <= ctx), default=sizes[0])
            shutil.copyfile(
                os.path.join(src_dir, f"sample_prompt_{size}.txt"),
                os.path.join(dest_prompts, f"sample_prompt_{ctx}.txt"),
            )

    @staticmethod
    def _stage_system_prompt(dest_dir: os.PathLike | str, system_prompt: str) -> None:
        """Write the eval system prompt to prompts/system_prompt.txt in the artifact."""
        prompts_dir = os.path.join(dest_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        with open(
            os.path.join(prompts_dir, "system_prompt.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(system_prompt.rstrip("\n"))

    @staticmethod
    def _apply_common_replacements(
        text: str,
        context_lengths: list[int],
        run_perf: bool,
        eval_prompts: list[str] | None,
        device_logs_dir: str,
        eval_seed: int,
        plugin: str,
        qairt_version: str | None,
    ) -> str:
        """Substitute the placeholders shared by all device scripts.

        CTX_LIST is comma-separated and the RUN_* flags are 1/0 regardless of
        platform; each script parses these into its own types. DEVICE_LOGS_DIR
        is a no-op on scripts without that placeholder (e.g. Windows, which
        still writes its own fixed ``QDC_Logs``).
        """
        return (
            text.replace("{EVAL_CTX}", str(max(context_lengths)))
            .replace("{EVAL_N_GEN}", str(_EVAL_N_GEN))
            .replace("{EVAL_SEED}", str(eval_seed))
            .replace("{EVAL_TIMEOUT_S}", str(_EVAL_TIMEOUT_S))
            .replace("{EVAL_SLEEP_S}", str(_EVAL_SLEEP_S))
            .replace("{EVAL_MAX_GEN_TIME_S}", str(_EVAL_MAX_GEN_TIME_S))
            .replace("{EVAL_NO_PROGRESS_TIMEOUT_S}", str(_EVAL_NO_PROGRESS_TIMEOUT_S))
            .replace("{EVAL_REPETITION_MAX_REPEATS}", str(_EVAL_REPETITION_MAX_REPEATS))
            .replace("{EVAL_NGRAM_MAX_REPEATS}", str(_EVAL_NGRAM_MAX_REPEATS))
            .replace("{CTX_LIST}", ",".join(str(c) for c in context_lengths))
            .replace("{RUN_PERF}", "1" if run_perf else "0")
            .replace("{RUN_EVAL}", "1" if eval_prompts else "0")
            .replace("{DEVICE_LOGS_DIR}", device_logs_dir)
            .replace("{PLUGIN}", plugin)
            # Empty QAIRT_VERSION keeps the QAIRT libs shipped in the geniex-bench bundle.
            .replace("{QAIRT_VERSION}", qairt_version or "")
        )

    @staticmethod
    def _rewrite_matrix_for_qairt_bundles(
        matrix_rows: list[str],
        qairt_bundles: dict[str, str],
        device_root: str,
    ) -> list[str]:
        # Replace matrix col-4 with the on-device bundle path so qairt plugin
        # skips the model-manager fetch.
        sep = "\\" if "\\" in device_root else "/"
        out: list[str] = []
        for row in matrix_rows:
            parts = row.split("|")
            name = parts[0]
            if name in qairt_bundles:
                parts[3] = f"{device_root}{sep}qairt_bundles{sep}{name}"
            out.append("|".join(parts))
        return out

    @staticmethod
    def _stage_qairt_bundles(
        dest_dir: os.PathLike | str, qairt_bundles: dict[str, str]
    ) -> None:
        base = os.path.join(dest_dir, "qairt_bundles")
        for model_id, bundle_dir in qairt_bundles.items():
            if not os.path.isdir(bundle_dir):
                raise FileNotFoundError(
                    f"QAIRT bundle for {model_id!r} is not a directory: {bundle_dir!r}"
                )
            shutil.copytree(
                bundle_dir, os.path.join(base, model_id), dirs_exist_ok=True
            )


class GenieXBenchAndroidArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "/data/local/tmp/pkg-geniex"

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        qairt_bundle_urls: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
        device_logs_dir: str,
        qairt_version: str | None,
        eval_seed: int = _DEFAULT_EVAL_SEED,
    ) -> list[tuple[str, str]]:
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        pytest_dir = os.path.join(ds_dir, "geniex_pytest")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        # AWS Device Farm's ~4GB artifact-upload cap makes shipping qairt
        # weights through the zip (_stage_qairt_bundles) infeasible for
        # larger models; when a presigned URL is available, the device
        # curls it directly instead (see qairt_bundle_urls below).
        stage_bundles_locally = bool(
            plugin == "qairt" and qairt_bundles and not qairt_bundle_urls
        )

        bench_url = _bench_url("android-arm64", "tar.gz", geniex_version)
        bundle_url = (qairt_bundle_urls or {}).get(next(iter(qairt_bundles or {}), ""))
        sdk_url = (
            get_geniex_qairt_sdk_url(qairt_version)
            if plugin == "qairt" and qairt_version
            else ""
        )
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)
        for fn in os.listdir(pytest_dir):
            if fn.endswith(".pyc"):
                continue
            src = os.path.join(pytest_dir, fn)
            if not os.path.isfile(src):
                continue
            with open(src, encoding="utf-8") as f:
                content = f.read()
            if fn.endswith(".py"):
                content = self._apply_common_replacements(
                    content.replace("{ANDROID_BENCH_URL}", bench_url)
                    .replace("{N_GEN}", str(_N_GEN))
                    .replace("{QAIRT_BUNDLE_URL}", bundle_url or "")
                    .replace("{QAIRT_SDK_URL}", sdk_url),
                    context_lengths,
                    run_perf,
                    eval_prompts,
                    device_logs_dir,
                    eval_seed,
                    plugin,
                    qairt_version,
                )
            out_path = (
                os.path.join(dest_dir, fn)
                if fn == "requirements.txt"
                else os.path.join(test_folder, fn)
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(content)

        with open(
            os.path.join(dest_dir, "matrix_rows.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(matrix_rows) + "\n")
        with open(os.path.join(dest_dir, "chipset.txt"), "w", encoding="utf-8") as f:
            f.write(chipset + "\n")

        if stage_bundles_locally:
            assert qairt_bundles is not None
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        if plugin == "qairt" and run_perf:
            self._stage_perf_prompts(dest_dir, context_lengths)

        if eval_prompts:
            self._stage_eval_prompts(dest_dir, eval_prompts)
            self._stage_system_prompt(dest_dir, DEFAULT_LLM_SYSTEM_PROMPT)

        return walk_dir_entries(os.fspath(dest_dir))


class GenieXBenchLinuxArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "/data/local/tmp/TestContent"

    @property
    def entry_script(self) -> str:
        return f"/bin/bash {self.DEVICE_ROOT}/run_geniex_bench_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        qairt_bundle_urls: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
        device_logs_dir: str,
        qairt_version: str | None,
        eval_seed: int = _DEFAULT_EVAL_SEED,
    ) -> list[tuple[str, str]]:
        # QDC only (never AWS Device Farm, which has no Linux/Windows devices),
        # so there's no upload cap to work around; always stage locally.
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        sh_src = os.path.join(ds_dir, "run_geniex_bench_linux.sh")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        with open(sh_src, encoding="utf-8") as f:
            script = f.read()
        script = self._apply_common_replacements(
            script.replace(
                "{LINUX_BENCH_URL}",
                _bench_url("linux-arm64", "tar.gz", geniex_version),
            )
            .replace("{CHIPSET}", chipset)
            .replace("{MODELS}", "\n".join(matrix_rows))
            .replace(
                "{BENCH_SIZE_FLAGS}",
                (
                    f'--prompt-file "$TC/perf_prompts/sample_prompt_$ctx.txt" --chat-template -c "$((ctx + {_N_GEN}))" -n {_N_GEN}'
                    if plugin == "qairt"
                    else f'-c "$((ctx + {_N_GEN}))" -p "$((ctx - {_N_GEN}))" -n {_N_GEN}'
                ),
            ),
            context_lengths,
            run_perf,
            eval_prompts,
            device_logs_dir,
            eval_seed,
            plugin,
            qairt_version,
        )
        sh_dest = os.path.join(dest_dir, "run_geniex_bench_linux.sh")
        with open(sh_dest, "w", encoding="utf-8") as f:
            f.write(script)
        os.chmod(sh_dest, 0o755)

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        if plugin == "qairt" and run_perf:
            self._stage_perf_prompts(dest_dir, context_lengths)

        if eval_prompts:
            self._stage_eval_prompts(dest_dir, eval_prompts)
            self._stage_system_prompt(dest_dir, DEFAULT_LLM_SYSTEM_PROMPT)

        return walk_dir_entries(os.fspath(dest_dir))


class GenieXBenchWindowsArtifactHandler(GenieXBenchArtifactHandler):
    DEVICE_ROOT = "C:\\Temp\\TestContent"

    @property
    def entry_script(self) -> str:
        return f"{self.DEVICE_ROOT}\\run_geniex_bench_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        dest_dir: os.PathLike | str,
        chipset: str,
        matrix_rows: list[str],
        context_lengths: list[int],
        plugin: str,
        qairt_bundles: dict[str, str] | None,
        qairt_bundle_urls: dict[str, str] | None,
        geniex_version: str | None,
        eval_prompts: list[str] | None,
        run_perf: bool,
        device_logs_dir: str,
        qairt_version: str | None,
        eval_seed: int = _DEFAULT_EVAL_SEED,
    ) -> list[tuple[str, str]]:
        # device_logs_dir is unused here: run_geniex_bench_windows.ps1 has no
        # {DEVICE_LOGS_DIR} placeholder -- it still writes its own fixed
        # QDC_Logs (never migrated by the device_logs rename this fix undoes
        # elsewhere), and QDC's harvesting already works for it as-is.
        ds_dir = os.path.join(curr_dirname, "device_scripts")
        ps1_src = os.path.join(ds_dir, "run_geniex_bench_windows.ps1")

        if plugin == "qairt" and qairt_bundles:
            matrix_rows = self._rewrite_matrix_for_qairt_bundles(
                matrix_rows, qairt_bundles, device_root=self.DEVICE_ROOT
            )

        with open(ps1_src, encoding="utf-8") as f:
            script = f.read()
        sdk_url = (
            get_geniex_qairt_sdk_url(qairt_version)
            if plugin == "qairt" and qairt_version
            else ""
        )
        script = self._apply_common_replacements(
            script.replace(
                "{WINDOWS_BENCH_URL}",
                _bench_url("windows-arm64", "zip", geniex_version),
            )
            .replace("{QAIRT_SDK_URL}", sdk_url)
            .replace("{CHIPSET}", chipset)
            .replace("{MODELS}", "\n".join(matrix_rows))
            .replace(
                "{BENCH_SIZE_FLAGS_ARGS}",
                (
                    f'"--prompt-file", "$TC\\perf_prompts\\sample_prompt_$ctx.txt", "--chat-template", "-c", "$($ctx + {_N_GEN})", "-n", "{_N_GEN}",'
                    if plugin == "qairt"
                    else f'"-c", "$($ctx + {_N_GEN})", "-p", "$($ctx - {_N_GEN})", "-n", "{_N_GEN}",'
                ),
            ),
            context_lengths,
            run_perf,
            eval_prompts,
            device_logs_dir,
            eval_seed,
            plugin,
            qairt_version,
        )
        with open(
            os.path.join(dest_dir, "run_geniex_bench_windows.ps1"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(script)

        if plugin == "qairt" and qairt_bundles:
            self._stage_qairt_bundles(dest_dir, qairt_bundles)

        if plugin == "qairt" and run_perf:
            self._stage_perf_prompts(dest_dir, context_lengths)

        if eval_prompts:
            self._stage_eval_prompts(dest_dir, eval_prompts)
            self._stage_system_prompt(dest_dir, DEFAULT_LLM_SYSTEM_PROMPT)

        return walk_dir_entries(os.fspath(dest_dir))


def _get_artifact_handler(platform: HubDevicePlatform) -> GenieXBenchArtifactHandler:
    if platform.is_windows_platform:
        return GenieXBenchWindowsArtifactHandler()
    if platform.is_iot_platform:
        return GenieXBenchLinuxArtifactHandler()
    if platform.is_mobile_platform:
        return GenieXBenchAndroidArtifactHandler()
    raise NotImplementedError(
        "geniex-bench currently supports Windows (Snapdragon X / X2 "
        "Elite), IoT Linux (Dragonwing IQ-9075 EVK), and Android "
        "(Samsung Galaxy S25 / S26). "
        f"Device {platform.device.name!r} is none of these."
    )


def add_geniex_bundle_entries(
    platform: HubDevicePlatform,
    dest_dir: str,
    chipset: str,
    matrix_rows: list[str],
    plugin: str,
    device_logs_dir: str,
    context_lengths: list[int] = DEFAULT_CONTEXT_LENGTHS,
    qairt_bundles: dict[str, str] | None = None,
    qairt_bundle_urls: dict[str, str] | None = None,
    geniex_version: str | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
    eval_seed: int = _DEFAULT_EVAL_SEED,
    qairt_version: str | None = None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Stage a GenieX-bench bundle into backend-agnostic (path, arcname) entries.

    ``dest_dir`` must outlive the returned entries -- callers zip/upload them
    before it is cleaned up. ``device_logs_dir`` is the on-device log
    directory name (see :func:`device_logs_dir_name`) rendered into the
    bundle's device scripts. ``qairt_bundle_urls`` (keyed like ``qairt_bundles``)
    holds presigned S3 GET URLs the Android handler curls on-device instead of
    shipping weights through the artifact zip. ``qairt_version`` (qairt plugin
    only) is a Software Center SDK version the device downloads and passes to
    geniex-bench via ``--qairt-lib``; None keeps the bundled QAIRT.
    """
    curr_dirname = os.path.dirname(os.path.abspath(__file__))
    handler = _get_artifact_handler(platform)
    if plugin != "qairt":
        qairt_version = None
    entries = handler.create_artifact(
        curr_dirname,
        dest_dir,
        chipset,
        matrix_rows,
        context_lengths,
        plugin,
        qairt_bundles,
        qairt_bundle_urls,
        geniex_version,
        eval_prompts,
        run_perf,
        device_logs_dir,
        qairt_version,
        eval_seed,
    )
    return entries, handler.entry_script


def _unreadable_cell_schema(path: str) -> str | None:
    """Schema version of a bench cell this parser rejected, else None.

    The log bundle also carries the model's own config/tokenizer JSON, so a
    file only counts as a bench cell if it has cell_id and agg.
    """
    try:
        with open(path, encoding="utf-8") as f:
            cell = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cell, dict) or "cell_id" not in cell or "agg" not in cell:
        return None
    version = str(cell.get("schema_version"))
    return None if version in _SUPPORTED_BENCH_SCHEMAS else version


def _parse_cell_metrics(path: str) -> GenieXBenchMetrics | None:
    try:
        with open(path, encoding="utf-8") as f:
            cell = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cell, dict):
        return None
    # geniex-bench bumps schema_version for additive changes; the fields we read
    # below are stable, and missing ones are rejected by the None checks.
    try:
        if int(cell.get("schema_version", 0)) < _MIN_RESULT_SCHEMA_VERSION:
            return None
    except (TypeError, ValueError):
        return None
    agg = cell.get("agg") or {}
    params = cell.get("params") or {}
    runs = cell.get("runs") or []

    def med(key: str) -> float | None:
        entry = agg.get(key) or {}
        return entry.get("median")

    ttft = med("ttft_ms")
    prefill = med("prefill_tps")
    decode = med("decode_tps")
    if ttft is None or prefill is None or decode is None:
        return None

    cell_id = cell.get("cell_id") or ""
    _, sep, suffix = cell_id.rpartition("-c")
    ctx = int(suffix) if sep and suffix.isdigit() else int(params.get("n_ctx") or 0)
    if ctx == 0:
        return None

    return GenieXBenchMetrics(
        cell_id=cell_id,
        plugin=cell.get("plugin") or "",
        device_alias=cell.get("device") or "",
        context_length=ctx,
        ttft_ms=float(ttft),
        prefill_tps=float(prefill),
        decode_tps=float(decode),
        prompt_tokens=int((agg.get("prompt_tokens") or {}).get("median") or 0),
        gen_tokens=int((agg.get("gen_tokens") or {}).get("median") or 0),
        geniex_version=cell.get("geniex_version"),
        qairt_version=cell.get("qairt_version"),
        llama_cpp_version=cell.get("llama_cpp_version"),
        stop_reason=runs[0].get("stop_reason") if runs else None,
    )


def compute_geniex_metrics(
    logs_dir: str,
    save_results_dir: str | None = None,
) -> list[GenieXBenchMetrics]:
    """Parse metrics from job logs fetched by ``DeviceFarm.download_job_logs``."""
    metrics: list[GenieXBenchMetrics] = []
    unreadable_schemas: dict[str, str] = {}
    for root, _, files in os.walk(logs_dir):
        for fn in sorted(files):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(root, fn)
            parsed = _parse_cell_metrics(path)
            if parsed is None:
                if version := _unreadable_cell_schema(path):
                    unreadable_schemas[fn] = version
                continue
            metrics.append(parsed)
            if save_results_dir:
                dest = os.path.join(save_results_dir, os.path.relpath(path, logs_dir))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(path, dest)

    if metrics:
        print(f"Parsed {len(metrics)} geniex-bench cells:")
        for m in metrics:
            print(
                f"  [{m.cell_id} ctx={m.context_length}] "
                f"decode={m.decode_tps:.2f} tok/s, prefill={m.prefill_tps:.2f} tok/s, "
                f"TTFT={m.ttft_ms:.1f} ms"
                + (f" [{m.zero_rate_note}]" if m.zero_rate_note else "")
            )
    elif unreadable_schemas:
        seen = sorted(set(unreadable_schemas.values()))
        raise RuntimeError(
            f"geniex-bench produced results this parser cannot read: schema "
            f"version(s) {', '.join(seen)}, supported "
            f"{', '.join(sorted(_SUPPORTED_BENCH_SCHEMAS))}. Add the version to "
            "_SUPPORTED_BENCH_SCHEMAS once its fields are confirmed compatible. "
            "Failing rather than reporting no metrics, so a schema bump can't be "
            "replayed as a measurement that never happened.\n"
            + "\n".join(
                f"  {fn}: schema_version={v}"
                for fn, v in sorted(unreadable_schemas.items())
            )
        )
    else:
        print(
            "Warning: no geniex-bench results (schema_version >= "
            f"{_MIN_RESULT_SCHEMA_VERSION}) found in logs."
        )

    return metrics


def compute_geniex_eval_results(logs_dir: str, prompts: list[str]) -> list[dict]:
    """Parse ``geniex_eval_outputs.txt`` into [{idx, prompt, output, cause}].

    ``logs_dir`` holds job logs fetched by ``DeviceFarm.download_job_logs``.
    The device scripts run ``geniex-bench --accuracy`` once per prompt and
    append each invocation's stdout to a single ``geniex_eval_outputs.txt``,
    with ``===EVAL_IDX_NNN===`` markers separating prompts.
    """
    outputs: dict[int, str] = {}
    for root, _, files in os.walk(logs_dir):
        for fn in files:
            if "geniex_eval_outputs" not in fn or fn.endswith(".zip"):
                continue
            with open(os.path.join(root, fn), "rb") as f:
                outputs.update(_parse_eval_outputs(_decode_device_log(f.read())))

    return [
        {
            "idx": idx,
            "prompt": prompts[idx] if idx < len(prompts) else "",
            "output": _extract_model_output(outputs.get(idx, "")),
            "cause": _extract_eval_cause(outputs.get(idx, "")),
        }
        for idx in sorted(outputs.keys())
    ]


def _decode_device_log(data: bytes) -> str:
    """Decode a device eval log, tolerating CP1252 from older Windows runs.

    PS 5.1's Add-Content defaulted to the ANSI codepage, so logs collected
    before that was pinned to utf8 hold lone high bytes that strict UTF-8
    rejects. Decoding those as CP1252 recovers the original text instead of
    replacing every accented character with U+FFFD.
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _parse_eval_outputs(content: str) -> dict[int, str]:
    """Split ``geniex_eval_outputs.txt`` into a ``{idx: raw_stdout}`` map.

    Format: ``===EVAL_IDX_NNN===`` followed by that prompt's raw geniex-bench
    stdout, up to the next marker.
    """
    outputs: dict[int, str] = {}
    parts = re.split(r"===EVAL_IDX_(\d+)===\n?", content)
    for i in range(1, len(parts) - 1, 2):
        outputs[int(parts[i])] = parts[i + 1].strip()
    return outputs


def _extract_model_output(raw_output: str) -> str:
    """Extract the generated text from one ``geniex-bench --accuracy`` stdout.

    ``--accuracy`` prints each generated line prefixed with ``[gen ] `` and a
    final ``[ok  ] ...`` perf-summary line. We keep only the ``[gen ]`` lines
    (stripping the prefix) and join them; everything else (the ``[ok  ]``
    summary, stray log lines) is dropped.
    """
    gen_prefix = "[gen ] "
    lines = [
        line[len(gen_prefix) :]
        for line in raw_output.splitlines()
        if line.startswith(gen_prefix)
    ]
    return "\n".join(lines).strip()


def _extract_eval_cause(raw_output: str) -> str:
    """Classify why one ``geniex-bench --accuracy`` run ended.

    Returns ``divergence:<reason>`` when geniex-bench self-aborted, else the
    reported ``stop_reason`` (e.g. ``length``, ``stop``), else ``freeze`` for
    a run that produced neither marker (killed externally or hung).
    """
    abort = _SELF_ABORT_RE.search(raw_output)
    if abort:
        return f"divergence:{abort.group(1)}"
    stop = _STOP_REASON_RE.search(raw_output)
    if stop:
        return stop.group(1)
    return "freeze"


def _hf_repo(model_url: str) -> str:
    if "huggingface.co/" not in model_url:
        raise ValueError(f"Only HuggingFace URLs are supported: {model_url}")
    parts = model_url.split("huggingface.co/")[1].split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse HF repo from: {model_url}")
    if not model_url.endswith(".gguf"):
        raise ValueError(f"Expected .gguf URL, got: {model_url}")
    return f"{parts[0]}/{parts[1]}"


def _build_matrix_rows(
    model_rows: list[tuple[str, str]],
    plugin: str,
    device_alias: str,
    llamacpp_quant: str | None,
) -> tuple[list[str], dict[str, str]]:
    matrix_rows: list[str] = []
    qairt_bundles: dict[str, str] = {}
    for name, ref in model_rows:
        if plugin == "qairt":
            qairt_bundles[name] = ref
            model_id = name
        else:
            model_id = f"{_hf_repo(ref)}:{llamacpp_quant}"
        matrix_rows.append(f"{name}|{plugin}|{device_alias}|{model_id}||")
    return matrix_rows, qairt_bundles


def get_geniex_qairt_bundle_url(
    model_id: str, precision: Precision, chipset: str
) -> str:
    """Mint a presigned S3 GET URL for the current geniex_qairt release asset.

    Re-resolves the asset fresh (same lookup as :func:`fetch_geniex_qairt_bundle`
    in run_geniex_bench_benchmarks.py) so a stale/failed compile is never
    signed -- always whatever ``load_release_assets_for_model`` currently
    selects.
    """
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIEX_QAIRT)
    if asset is None or asset.s3_key is None:
        raise RuntimeError(
            f"No geniex_qairt release asset for {model_id!r} precision={precision} "
            f"chipset={chipset!r}; cannot mint a presigned download URL."
        )
    bucket = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)[0]
    return get_presigned_download_url(bucket, asset.s3_key)


def get_qairt_sdk_software_center_url(qairt_version: str) -> str:
    """Build the Software Center download URL for a QAIRT SDK zip.

    Parameters
    ----------
    qairt_version
        Full SDK version, e.g. "2.50.0.260828".

    Returns
    -------
    url : str
        The public Software Center URL for that version's SDK zip.
    """
    return (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{qairt_version}/v{qairt_version}.zip"
    )


def get_geniex_qairt_sdk_url(qairt_version: str) -> str:
    """Mint a presigned S3 GET URL for a QAIRT SDK zip.

    The device curls this instead of Software Center, the same way it pulls
    the model bundle (see :func:`get_geniex_qairt_bundle_url`). Software
    Center cut short several 2.5GB transfers; only versions mirrored to S3
    can be served from it, so an unmirrored one falls back to Software Center.

    Parameters
    ----------
    qairt_version
        Full SDK version, e.g. "2.50.0.260828".

    Returns
    -------
    url : str
        A presigned S3 URL, or the Software Center URL if S3 has no such zip.
    """
    key = f"{QAIRT_SDK_S3_PREFIX}/v{qairt_version}.zip"
    bucket = get_qaihm_s3(QAIHM_PUBLIC_S3_BUCKET)[0]
    if not s3_file_exists(bucket, key):
        print(
            f"No QAIRT SDK at s3://{QAIHM_PUBLIC_S3_BUCKET}/{key}; "
            "falling back to Software Center."
        )
        return get_qairt_sdk_software_center_url(qairt_version)
    return get_presigned_download_url(bucket, key)


def submit_geniex_bench(
    backend: DeviceFarm,
    hub_device_name: str,
    chipset: str,
    model_rows: list[tuple[str, str]],
    context_lengths: list[int] = DEFAULT_CONTEXT_LENGTHS,
    plugin: str = "llama_cpp",
    device_alias: str = "hybrid",
    job_name: str = "geniex-bench",
    geniex_version: str | None = None,
    llamacpp_quant: str | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
    qairt_bundle_urls: dict[str, str] | None = None,
    eval_seed: int = _DEFAULT_EVAL_SEED,
    qairt_version: str | None = None,
) -> tuple[str, list[str], dict[str, str]]:
    """Upload artifacts and submit a geniex-bench job, returning the id.

    Companion to :func:`collect_geniex_bench`. Also returns the computed
    ``matrix_rows`` and ``qairt_bundles`` so the caller can persist them for
    a later resubmit.

    eval_prompts set => accuracy collection: staged raw into the bundle for
    one ``geniex-bench --accuracy`` pass, which applies the bundle's own chat
    template on-device. run_perf=False submits an eval-only job (no TPS/TTFT
    sweep). A presigned S3 GET URL per model in ``qairt_bundles``, keyed the
    same way, lets the Android handler curl the bundle on-device instead of
    shipping weights through the artifact zip; callers mint these fresh per
    submit/resubmit via :func:`get_geniex_qairt_bundle_url`.
    """
    if plugin == "llama_cpp" and not llamacpp_quant:
        raise ValueError("llamacpp_quant is required when plugin='llama_cpp'.")

    platform = HubDevicePlatform(hub_device_name)
    matrix_rows, qairt_bundles = _build_matrix_rows(
        model_rows, plugin, device_alias, llamacpp_quant
    )
    device_logs_dir = device_logs_dir_name(backend)

    # Staging dir must outlive add_geniex_bundle_entries: the returned entries
    # reference files inside it, and submit_bundle needs them to still exist
    # when it zips/uploads. eval_prompts are staged raw -- geniex-bench applies
    # the bundle's own chat template on-device (no host-side templating).
    with tempfile.TemporaryDirectory(prefix="geniex_artifact_staging_") as dest_dir:
        entries, entry_script = add_geniex_bundle_entries(
            platform,
            dest_dir,
            chipset,
            matrix_rows,
            plugin,
            device_logs_dir,
            context_lengths=context_lengths,
            qairt_bundles=qairt_bundles or None,
            qairt_bundle_urls=qairt_bundle_urls or None,
            geniex_version=geniex_version,
            eval_prompts=eval_prompts,
            run_perf=run_perf,
            eval_seed=eval_seed,
            qairt_version=qairt_version,
        )

        # No explicit timeout: submission itself doesn't block on either
        # backend (QDC queues server-side; AWS's own submit_bundle default
        # already reflects its execution cap, which AWS limits to 150
        # minutes) -- the wait now lives entirely in collect_geniex_bench's
        # status() poll (see GENIEX_BENCH_JOB_TIMEOUT).
        job_id = backend.submit_bundle(
            hub_device_name,
            entries,
            entry_script,
            job_name=job_name,
        )
    if job_id is None:
        raise RuntimeError("Job submission failed.")
    print(f"Submitted job with ID: {job_id}")
    return job_id, matrix_rows, qairt_bundles


def _fetch_and_parse_job_logs(
    backend: DeviceFarm,
    job_id: str,
    job_log_files: list,
    save_logs_dir: str | None,
    log_label: str | None,
    save_results_dir: str | None,
    eval_prompts: list[str] | None,
    run_perf: bool,
) -> tuple[list[GenieXBenchMetrics], list[dict]]:
    """Download the logs once, archive them, then parse metrics + eval from them.

    Archiving comes before parsing so the logs survive whatever verdict follows.
    """
    with tempfile.TemporaryDirectory() as logs_dir:
        backend.download_job_logs(job_log_files, logs_dir)
        backend.archive_job_logs(job_id, logs_dir, save_logs_dir, label=log_label)
        metrics = (
            compute_geniex_metrics(logs_dir, save_results_dir=save_results_dir)
            if run_perf
            else []
        )
        # eval_prompts holds the raw questions so compute_eval_results labels each
        # output with the human-readable prompt (not the templated form).
        eval_results = (
            compute_geniex_eval_results(logs_dir, eval_prompts) if eval_prompts else []
        )
    return metrics, eval_results


def collect_geniex_bench(
    backend: DeviceFarm,
    hub_device_name: str,
    job_id: str,
    save_results_dir: str | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
    save_logs_dir: str | None = None,
    log_label: str | None = None,
) -> tuple[list[GenieXBenchMetrics], list[dict], JobOutcome, str | None]:
    """Poll a submitted geniex-bench job and download+parse its logs.

    Returns ``(metrics, eval_results, outcome, reason)``; ``reason`` carries the
    failure description on a non-SUCCESS outcome. A failed job's logs are parsed
    too, since it can fail after writing every result (e.g. geniex-bench
    segfaulting in teardown); the caller decides whether those are enough.
    eval_prompts attaches the human-readable prompt text to each parsed output
    (run_perf=False yields eval-only results).
    ``log_label`` names the per-job log archive written under ``save_logs_dir``.
    """
    job_status = backend.status(job_id, timeout=GENIEX_BENCH_JOB_TIMEOUT)
    job_result = backend.result(job_id)
    print(f"Job {job_id} completed with status: {job_status}, result: {job_result}")

    if not backend.is_successful(job_result):
        reason = (
            f"Job {job_id} on device '{hub_device_name}' finished with "
            f"status='{job_status}', result='{job_result}'"
        )
        outcome = backend.classify_failure(job_result)
        print(f"[result={job_result}] {reason}")
        try:
            with contextlib.suppress(TimeoutError):
                backend.log_upload_status(job_id, timeout=FAILED_JOB_LOG_TIMEOUT)
            metrics, eval_results = _fetch_and_parse_job_logs(
                backend,
                job_id,
                backend.get_job_log_files(job_id, wait_for_logs=True),
                save_logs_dir,
                log_label,
                save_results_dir,
                eval_prompts,
                run_perf,
            )
        except Exception as err:
            # Type only, never the message: backend errors may embed secrets.
            print(f"Could not parse logs of failed job {job_id} ({type(err).__name__})")
            return [], [], outcome, reason
        return metrics, eval_results, outcome, reason

    backend.log_upload_status(job_id)
    # The file listing can lag the log-upload signal on some backends, so wait
    # for it to populate -- otherwise a successful job yields no metrics.
    job_log_files = backend.get_job_log_files(job_id, wait_for_logs=True)

    if not job_log_files:
        reason = (
            f"Job {job_id} on device '{hub_device_name}' reported result="
            f"'{job_result}' but produced no retrievable log files"
        )
        print(f"[empty logs] {reason}")
        return [], [], JobOutcome.RETRYABLE_EMPTY_LOGS, reason

    # A 'Successful' job can still be failed below (e.g. no eval output), and the
    # logs archived here are what explains why.
    metrics, eval_results = _fetch_and_parse_job_logs(
        backend,
        job_id,
        job_log_files,
        save_logs_dir,
        log_label,
        save_results_dir,
        eval_prompts,
        run_perf,
    )
    # A job can report a successful result and return log files while carrying
    # no eval output at all: if the device drops off adb mid-run, device_logs is
    # never harvested and only the infrastructure logs come back. Treating that
    # as SUCCESS makes it non-retryable, so the run ends as
    # eval_incomplete (0/N) with an attempt still unspent. Classify it as
    # retryable-empty-logs instead, matching the no-log-files case above.
    if eval_prompts and not eval_results:
        reason = (
            f"Job {job_id} on device '{hub_device_name}' reported result="
            f"'{job_result}' but returned no eval output for "
            f"{len(eval_prompts)} prompt(s); the device logs were likely never "
            f"harvested"
        )
        print(f"[no eval output] {reason}")
        return metrics, [], JobOutcome.RETRYABLE_EMPTY_LOGS, reason
    return metrics, eval_results, JobOutcome.SUCCESS, None
