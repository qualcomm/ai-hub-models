# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from qai_hub_models import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.datasets.device_eval import (
    DEFAULT_DEVICE_EVAL_DATASET,
    DeviceEvalItem,
    label_eval_results,
    load_device_eval_items,
)
from qai_hub_models.models.templates.llm.perf_collection import (
    load_release_assets_for_model,
    record_perf_scope,
    resolve_llm_eval_device,
    update_perf_yaml,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import ScorecardDevice, cs_ventuno_q, cs_x_elite
from qai_hub_models.scorecard.envvars import (
    IgnoreKnownFailuresEnvvar,
    LLMPerfPrecisionsEnvvar,
    LLMPerfUpdatesEnvvar,
    QAIRTVersionEnvvar,
    SpecialLLMPerfPrecisionSetting,
)
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.utils.fetch_prerelease_assets import (
    download_prerelease_asset,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.devicefarm.devicefarm import (
    DeviceFarm,
    JobOutcome,
    JobRecord,
    get_device_farm,
    load_jobs,
    make_key,
    poll_job,
    sanitize_job_id,
    save_job,
)
from qai_hub_models.utils.llm.eval_io import (
    save_eval_metadata_json,
    save_eval_results_json,
)
from qai_hub_models.utils.llm.geniex.jobs import (
    GenieXBenchMetrics,
    collect_geniex_bench,
    get_geniex_qairt_bundle_url,
    submit_geniex_bench,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

DEFAULT_DEVICES = "cs_x2_elite,cs_x_elite"

# One device per platform bucket in geniex/jobs.py's artifact handlers.
# cs_8_elite/cs_8_elite_gen_5 run the same Android path via AWS Device Farm
# instead of QDC (see ScorecardDevice.devicefarm_backend).
ALL_GENIEX_DEVICES = (
    "cs_x_elite",
    "cs_x2_elite",
    "cs_9075",
    "cs_8_elite",
    "cs_8_elite_gen_5",
)
LLAMACPP_DEVICE_ALIASES = ("cpu", "gpu", "npu")
LLAMACPP_CONTEXT_LENGTHS = [512, 4096]

# Queued (not running) jobs allowed per device type: enough that a freed phone
# picks up the next job immediately, small enough that nothing waits for hours.
DEFAULT_MAX_WAITING = 2
DEFAULT_RUN_POLL_INTERVAL = 60
# QDC rejects submits (400 "already has N pending jobs") past this many of our
# jobs not yet finished, per account and across every device on it.
QDC_MAX_UNFINISHED_JOBS = 15


def _qairt_precisions(model_id: str) -> list[Precision]:
    # Filter on runtime capability AND manifest allow-rules so llamacpp-only
    # LLMs (only_allow_orchestrator_runtimes without GENIEX_QAIRT in
    # orchestrator_runtimes) aren't scheduled onto the QAIRT plugin.
    cg = QAIHMModelManifest.from_model(model_id)
    return [
        p
        for p in cg.supported_precisions
        if TargetRuntime.GENIEX_QAIRT.supports_precision(p)
        and cg.failure_reason(
            p,
            TargetRuntime.GENIEX_QAIRT,
            include_scorecard_failures=False,
            include_user_defined_failures=False,
            include_timeouts=False,
        )
        is None
    ]


def _llamacpp_assets(model_id: str) -> dict[Precision, str]:
    """Return {precision: gguf_url} for each precision with a geniex_llamacpp
    asset in release-assets.yaml. Quants vary across models (q4_0 for most,
    mxfp4 for gpt_oss_20b, etc.).
    """
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    out: dict[Precision, str] = {}
    for precision, prec_details in assets.precisions.items():
        asset = prec_details.universal_assets.get(ScorecardProfilePath.GENIEX_LLAMACPP)
        if asset and asset.download_url:
            out[precision] = asset.download_url
    return out


def _candidate_model_ids(filter_models: str | None) -> list[str]:
    if filter_models and filter_models.lower() != "all":
        return [m.strip() for m in filter_models.split(",") if m.strip()]
    return list(MODEL_IDS)


def discover_qairt_models(filter_models: str | None) -> list[str]:
    return [
        mid for mid in _candidate_model_ids(filter_models) if _qairt_precisions(mid)
    ]


def discover_llamacpp_models(filter_models: str | None) -> list[str]:
    return [mid for mid in _candidate_model_ids(filter_models) if _llamacpp_assets(mid)]


def _resolve_precisions(
    setting: set[str | SpecialLLMPerfPrecisionSetting],
    candidates: list[Precision],
) -> list[Precision]:
    if not candidates:
        return []
    if SpecialLLMPerfPrecisionSetting.ALL in setting:
        return candidates
    if SpecialLLMPerfPrecisionSetting.DEFAULT in setting:
        return [candidates[0]]
    candidate_set = set(candidates)
    return [
        Precision.parse(p)
        for p in setting
        if isinstance(p, str) and Precision.parse(p) in candidate_set
    ]


def fetch_geniex_qairt_bundle(
    model_id: str, precision: Precision, chipset: str, output_dir: Path
) -> tuple[Path, list[int]]:
    """Download/extract the CI-built geniex_qairt bundle. Returns (bundle_dir, context_lengths)."""
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIEX_QAIRT)
    if asset is None:
        available: list[str] = []
        prec_details = assets.precisions.get(precision)
        if prec_details is not None:
            available = sorted(prec_details.chipset_assets.keys())
        raise RuntimeError(
            f"No geniex_qairt release asset for {model_id!r} precision={precision} "
            f"chipset={chipset!r}. Available: {available or '<none>'}. "
            "Build release-assets.yaml with GENIEX_QAIRT before running geniex-bench QAIRT."
        )

    bundle_dir = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_id, TargetRuntime.GENIEX_QAIRT, precision, chipset
    )
    if not bundle_dir.exists():
        zip_path = download_prerelease_asset(
            asset,
            model_id=model_id,
            runtime=TargetRuntime.GENIEX_QAIRT,
            precision=precision,
            chipset=chipset,
            output_folder=output_dir,
            verbose=True,
        )
        shutil.unpack_archive(str(zip_path), extract_dir=str(output_dir))
        # Delete the zip once extracted so it doesn't fill the runner disk.
        zip_path.unlink(missing_ok=True)
        if not bundle_dir.exists():
            raise RuntimeError(
                f"Extracted geniex_qairt bundle missing expected directory {bundle_dir}; "
                f"contents of {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
            )

    metadata = ModelMetadata.from_json(bundle_dir / "metadata.json")
    if metadata is None or metadata.genie is None:
        raise RuntimeError(
            f"GenieX-QAIRT bundle for {model_id!r} has no genie metadata at "
            f"{bundle_dir / 'metadata.json'}"
        )
    return bundle_dir, metadata.genie.context_lengths


def _scorecard_device(name: str) -> ScorecardDevice:
    return ScorecardDevice.get(name)


def _device_alias(plugin: str) -> str:
    return ",".join(LLAMACPP_DEVICE_ALIASES) if plugin == "llama_cpp" else "npu"


def _runtime(plugin: str) -> str:
    return "GENIEX_QAIRT" if plugin == "qairt" else "GENIEX_LLAMACPP"


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Model",
                "Plugin",
                "Precision",
                "Device",
                "Ctx",
                "Decode TPS",
                "Prefill TPS",
                "TTFT (ms)",
                "Status",
                "Note",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["model"],
                    r.get("plugin", ""),
                    r.get("precision", ""),
                    r["device"],
                    r.get("ctx", ""),
                    r.get("decode_tps", ""),
                    r.get("prefill_tps", ""),
                    r.get("ttft_ms", ""),
                    r["status"],
                    r.get("note", ""),
                ]
            )


def write_summary(rows: list[dict]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return

    def _format_values(v: object, spec: str = ".2f") -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    with open(summary, "a") as f:
        f.write("## GenieX On-device Results\n\n")
        f.write(
            "| Model | Plugin | Precision | Device | Ctx | Decode TPS | Prefill TPS | TTFT (ms) | Status |\n"
        )
        f.write(
            "|-------|--------|-----------|--------|----:|-----------:|------------:|----------:|--------|\n"
        )
        f.writelines(
            f"| {r['model']} | {r.get('plugin', '-')} | {r.get('precision', '-')} | "
            f"{r['device']} | {r.get('ctx', '-')} | "
            f"{_format_values(r.get('decode_tps'))} | {_format_values(r.get('prefill_tps'))} | "
            f"{_format_values(r.get('ttft_ms'), '.1f')} | "
            f"{r['status']}{' (' + r['note'] + ')' if r.get('note') else ''} |\n"
            for r in rows
        )
        f.write("\n")


def _device_logs_dir(
    save_dir_root: str,
    model_id: str,
    precision: Precision,
    device_name: str,
    runtime: str,
    job_id: str,
) -> str:
    """Structured path for persisted device log zips (QDC or AWS Device Farm).

    Keyed on job_id so a retried job's logs don't clobber the prior attempt.
    """
    return os.path.join(
        save_dir_root,
        "device_logs",
        model_id,
        f"{precision}_{device_name}_{runtime}_{sanitize_job_id(job_id)}",
    )


def _print_job_banner(
    model_id: str,
    sd: ScorecardDevice,
    plugin: str,
    device_alias: str,
    model_ref: str,
    context_lengths: list[int],
    geniex_version: str | None,
    run_perf: bool = True,
    eval_items: list[DeviceEvalItem] | None = None,
    qairt_version: str | None = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Model:   {model_id}")
    print(f"Device:  {sd.name} ({sd.reference_device_name}, chipset={sd.chipset})")
    print(f"Plugin:  {plugin} (alias={device_alias})")
    print(f"Ref:     {model_ref}")
    print(f"Ctx:     {context_lengths}")
    print(f"GenieX:  {geniex_version or 'latest stable mirror'}")
    if plugin == "qairt":
        print(f"QAIRT:   {qairt_version}")
    print(f"Perf:    {'on' if run_perf else 'off'}")
    print(f"Eval:    {f'on ({len(eval_items)} prompts)' if eval_items else 'off'}")
    print(f"{'=' * 60}")


def _submit_one(
    model_id: str,
    model_ref: str,
    device_token: str,
    context_lengths: list[int],
    save_dir_root: str,
    plugin: str,
    geniex_version: str | None,
    precision: Precision,
    jobs_file: str,
    llamacpp_quant: str | None = None,
    eval_items: list[DeviceEvalItem] | None = None,
    run_perf: bool = True,
    qairt_version: str | None = None,
) -> str:
    """Submit one geniex-bench job and upsert a jobs_file entry.

    Returns the QDC job id. The collect side re-derives everything else
    from (model_id, precision, chipset) via
    ``fetch_geniex_qairt_bundle`` -- nothing local-path is persisted.
    """
    sd = _scorecard_device(device_token)
    device_alias = _device_alias(plugin)
    _print_job_banner(
        model_id,
        sd,
        plugin,
        device_alias,
        model_ref,
        context_lengths,
        geniex_version,
        run_perf,
        eval_items,
        qairt_version,
    )

    job_name = f"geniex-bench {plugin} {model_id}"
    backend = get_device_farm(sd)
    qairt_bundle_urls = (
        {model_id: get_geniex_qairt_bundle_url(model_id, precision, sd.chipset)}
        if plugin == "qairt"
        else None
    )
    job_id, _, _ = submit_geniex_bench(
        backend,
        hub_device_name=sd.reference_device_name,
        chipset=sd.chipset,
        model_rows=[(model_id, model_ref)],
        context_lengths=context_lengths,
        plugin=plugin,
        device_alias=device_alias,
        job_name=job_name,
        geniex_version=geniex_version,
        llamacpp_quant=llamacpp_quant,
        eval_prompts=_prompt_texts(eval_items),
        run_perf=run_perf,
        qairt_bundle_urls=qairt_bundle_urls,
        qairt_version=qairt_version,
    )
    key = make_key(model_id, str(precision), _runtime(plugin), sd.name)
    save_job(jobs_file, key, job_id)
    return job_id


def _prompt_texts(eval_items: list[DeviceEvalItem] | None) -> list[str] | None:
    return [item.prompt for item in eval_items] if eval_items else None


def _has_all_results(
    metrics: list[GenieXBenchMetrics],
    eval_results: list[dict],
    plugin: str,
    ctx_list: list[int],
    eval_items: list[DeviceEvalItem] | None,
    run_perf: bool,
) -> bool:
    if not run_perf and not eval_items:
        return False
    expected = {(a, c) for a in _device_alias(plugin).split(",") for c in ctx_list}
    got = {(m.device_alias, m.context_length) for m in metrics}
    perf_ok = not run_perf or expected <= got
    eval_ok = not eval_items or len(eval_results) >= len(eval_items)
    return perf_ok and eval_ok


def _collect_attempt(
    backend: DeviceFarm,
    sd: ScorecardDevice,
    model_id: str,
    precision: Precision,
    plugin: str,
    job_id: str,
    save_dir_root: str,
    ctx_list: list[int],
    eval_items: list[DeviceEvalItem] | None,
    run_perf: bool,
) -> tuple[list[GenieXBenchMetrics], list[dict], JobOutcome, str | None, str | None]:
    """Collect one job. Returns ``(metrics, eval_results, outcome, reason, kept_failure)``.

    A failed job whose logs hold every expected result is upgraded to SUCCESS,
    and its failure reason is returned as ``kept_failure``.
    """
    logs_dir = _device_logs_dir(
        save_dir_root, model_id, precision, sd.name, _runtime(plugin), job_id
    )
    metrics, eval_results, outcome, reason = collect_geniex_bench(
        backend,
        sd.reference_device_name,
        job_id,
        save_results_dir=os.path.join(save_dir_root, model_id, sd.name),
        eval_prompts=_prompt_texts(eval_items),
        run_perf=run_perf,
        save_logs_dir=logs_dir,
    )
    if eval_items:
        eval_results = label_eval_results(eval_results, eval_items)
    # e.g. geniex-bench segfaulting in teardown fails the job after every
    # result landed; a resubmit would crash the same way.
    if outcome is not JobOutcome.SUCCESS and _has_all_results(
        metrics, eval_results, plugin, ctx_list, eval_items, run_perf
    ):
        print(f"Keeping all results from failed job: {reason}")
        return metrics, eval_results, JobOutcome.SUCCESS, reason, reason
    return metrics, eval_results, outcome, reason, None


def _final_status(
    metrics: list[GenieXBenchMetrics],
    eval_results: list[dict],
    run_perf: bool,
    eval_items: list[DeviceEvalItem] | None,
) -> str:
    if metrics:
        return "success"
    # With perf off the device runs eval only, so an empty metrics list is
    # expected rather than a failure. But the eval must still return one result
    # per prompt sent; a short count means the run crashed or was truncated.
    if run_perf:
        return "no_metrics"
    if eval_items and len(eval_results) != len(eval_items):
        return f"eval_incomplete ({len(eval_results)}/{len(eval_items)})"
    return "eval_only"


def _collect_one(
    model_id: str,
    precision: Precision,
    device_name: str,
    plugin: str,
    record: JobRecord,
    save_dir_root: str,
    ctx_list: list[int],
    eval_items: list[DeviceEvalItem] | None = None,
    run_perf: bool = True,
) -> tuple[list[GenieXBenchMetrics], list[dict], str, str | None]:
    """Poll a submitted geniex-bench job. A failed job whose logs hold every
    expected result is kept rather than reported as failed.

    Returns ``(metrics, eval_results, status, job_failure)`` where status is
    ``"success"``, ``"eval_only"``, ``"eval_incomplete (...)"``,
    ``"no_metrics"``, or ``"failed"``, and ``job_failure`` is the reason the
    kept job failed, if it did.
    """
    sd = ScorecardDevice.get(device_name)
    backend = get_device_farm(sd)
    job_failure: str | None = None

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        nonlocal job_failure
        metrics, eval_results, outcome, reason, kept = _collect_attempt(
            backend,
            sd,
            model_id,
            precision,
            plugin,
            job_id,
            save_dir_root,
            ctx_list,
            eval_items,
            run_perf,
        )
        job_failure = kept or job_failure
        return (metrics, eval_results), outcome, reason

    try:
        metrics, eval_results = poll_job(record.job_id, _collect)
    except RuntimeError as e:
        print(
            f"ERROR: {e} for {model_id} @ {device_name}",
            file=sys.stderr,
        )
        return [], [], "failed", None

    status = _final_status(metrics, eval_results, run_perf, eval_items)
    return metrics, eval_results, status, job_failure


def _rows_from_metrics(
    model_id: str,
    sd: ScorecardDevice,
    plugin: str,
    precision: Precision,
    metrics: list[GenieXBenchMetrics],
    skip_perf_update: bool,
) -> list[dict]:
    """Extracted formatting so ``run`` and ``collect`` share output shape.

    Also upserts perf.yaml, which logs each metric to LLMPerfUpdatesEnvvar.
    """
    csv_rows: list[dict] = []
    for m in metrics:
        base_plugin = m.plugin or plugin
        plugin_label = (
            f"{base_plugin}_{m.device_alias}"
            if base_plugin == "llama_cpp" and m.device_alias
            else base_plugin
        )
        csv_rows.append(
            {
                "model": model_id,
                "plugin": plugin_label,
                "precision": str(precision),
                "device": sd.name,
                "ctx": m.context_length,
                "decode_tps": m.decode_tps,
                "prefill_tps": m.prefill_tps,
                "ttft_ms": m.ttft_ms,
                "status": "success",
                "note": m.zero_rate_note or "",
            }
        )
        if not skip_perf_update:
            profile_path = (
                ScorecardProfilePath.GENIEX_QAIRT
                if plugin == "qairt"
                else ScorecardProfilePath.GENIEX_LLAMACPP
            )
            # report.c writes qairt_version unconditionally on every cell
            # regardless of --plugin, so it's only meaningful for qairt-plugin
            # cells -- an llama_cpp cell's qairt_version reflects whatever
            # QAIRT plugin happens to also be registered in that binary.
            tool_versions = ToolVersions(
                geniex=m.geniex_version,
                qairt=(
                    QAIRTVersion(m.qairt_version, validate_exists_on_ai_hub=False)
                    if base_plugin == "qairt" and m.qairt_version
                    else None
                ),
            )
            update_perf_yaml(
                model_id=model_id,
                device_name=sd.reference_device_name,
                precision=precision,
                context_length=m.context_length,
                tps=m.decode_tps,
                prefill_tps=m.prefill_tps,
                profile_path=profile_path,
                desired_compute_unit=m.device_alias,
                tool_versions=tool_versions,
            )
    return csv_rows


def _iter_work(
    models_setting: str,
    devices_setting: str,
    plugin_setting: str,
    precisions_setting: str,
    results_dir: str,
    geniex_version: str | None,
) -> Iterator[tuple[str, str, Precision, str, str, list[int], str | None]]:
    """Yield (plugin, model_id, precision, device_token, model_ref, ctx_list,
    llamacpp_quant) tuples for every model x precision x device combo.

    Shared by ``submit`` and ``run``. Handles all the discovery and
    bundle-fetch logic so the two paths stay in step.
    """
    plugins = ["llama_cpp", "qairt"] if plugin_setting == "all" else [plugin_setting]
    precision_setting = LLMPerfPrecisionsEnvvar.parse(precisions_setting)

    if devices_setting.strip().lower() == "all":
        devices = list(ALL_GENIEX_DEVICES)
    else:
        devices = [d.strip() for d in devices_setting.split(",") if d.strip()]
    devices = [d for d in devices if _scorecard_device(d).qdc_enabled]

    for plugin in plugins:
        if plugin == "qairt":
            models = discover_qairt_models(models_setting)
            if not models:
                print("No models support the GENIEX_QAIRT runtime.")
                continue
        else:
            models = discover_llamacpp_models(models_setting)
            if not models:
                print("No models support the GENIEX_LLAMACPP runtime.")
                continue

        print(f"Plugin:  {plugin}")
        print(f"Models:  {models}")
        print(f"Devices: {devices}")

        for model_id in models:
            if plugin == "qairt":
                candidates = _qairt_precisions(model_id)
                llamacpp_urls: dict[Precision, str] = {}
            else:
                llamacpp_urls = _llamacpp_assets(model_id)
                candidates = list(llamacpp_urls.keys())

            precisions = _resolve_precisions(precision_setting, candidates)
            if not precisions:
                print(
                    f"Skipping {model_id} on {plugin}: no requested precision "
                    f"available (candidates={[str(p) for p in candidates]})."
                )
                continue

            manifest = QAIHMModelManifest.from_model(model_id)
            runtime = (
                TargetRuntime.GENIEX_QAIRT
                if plugin == "qairt"
                else TargetRuntime.GENIEX_LLAMACPP
            )
            for precision in precisions:
                for device_token in devices:
                    sd = _scorecard_device(device_token)
                    if not manifest.is_supported(
                        precision,
                        runtime,
                        consider_scorecard_failures=False,
                        consider_user_defined_failures=not IgnoreKnownFailuresEnvvar.get(),
                        device=sd.device_name,
                    ):
                        print(
                            f"Skipping {model_id} [{precision}] @ {device_token}: disabled in manifest."
                        )
                        continue
                    if plugin == "qairt":
                        try:
                            bundle_dir, ctx_list = fetch_geniex_qairt_bundle(
                                model_id,
                                precision,
                                sd.chipset,
                                Path(results_dir) / "qairt_bundles",
                            )
                        except RuntimeError as e:
                            print(
                                f"Skipping {model_id} [{precision}] @ {device_token}: {e}"
                            )
                            continue
                        yield (
                            plugin,
                            model_id,
                            precision,
                            device_token,
                            str(bundle_dir),
                            ctx_list,
                            None,
                        )
                    else:
                        yield (
                            plugin,
                            model_id,
                            precision,
                            device_token,
                            llamacpp_urls[precision],
                            LLAMACPP_CONTEXT_LENGTHS,
                            str(precision),
                        )


def _write_final_outputs(rows: list[dict], csv_path: str) -> int:
    print(f"\n{'=' * 60}\nRESULTS SUMMARY\n{'=' * 60}")
    for r in rows:
        prec = r.get("precision", "-")
        if r["status"] == "success":
            print(
                f"  {r['model']} [{prec}] @ {r['device']} ctx={r['ctx']}: "
                f"decode={r['decode_tps']:.2f} prefill={r['prefill_tps']:.2f} "
                f"TTFT={r['ttft_ms']:.1f}ms"
                + (f" [{r['note']}]" if r.get("note") else "")
            )
        else:
            print(f"  {r['model']} [{prec}] @ {r['device']}: {r['status']}")

    write_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")
    write_summary(rows)
    # eval_only rows are successful eval-only runs (perf disabled), not failures.
    ok_statuses = {"success", "eval_only"}
    failed = [r for r in rows if r["status"] not in ok_statuses]
    if not rows:
        print("No models were benchmarked (all skipped or no candidates).")
        return 0
    return 1 if failed else 0


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models", default="all", help='Comma-separated model IDs or "all".'
    )
    parser.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="Comma-separated cs_* names or hub device names, "
        'or "all" for every geniex-bench-supported device.',
    )
    parser.add_argument(
        "--plugin",
        default="all",
        choices=["all", "llama_cpp", "qairt"],
    )
    parser.add_argument(
        "--precisions",
        default="default",
        help='Comma-separated precisions (e.g. "w4,w4a16"), "all" for every '
        'supported precision, or "default" to use the model\'s default precision.',
    )
    parser.add_argument("--results-dir", default="geniex_bench_results")
    parser.add_argument(
        "--geniex-version",
        default=None,
        help='GenieX release tag (e.g. "v0.3.1") to pin geniex-bench/APK '
        'downloads to. Defaults to the unversioned "latest stable" mirror.',
    )
    parser.add_argument(
        "--qairt-version",
        default=None,
        help='QAIRT version for the qairt plugin (e.g. "2.50" or "qaihm_default"). '
        "A non-default version is downloaded on-device and swapped into the "
        "geniex-bench bundle in place of its shipped QAIRT libs. Defaults to "
        "the QAIHM_TEST_QAIRT_VERSION env var; ignored for llama_cpp.",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        default=os.environ.get("QAIHM_RUN_EVAL", "").lower() == "true",
        help="Also collect the accuracy eval (geniex-bench --accuracy) on "
        "--eval-dataset and write *_eval.json for grading. Defaults to the "
        "QAIHM_RUN_EVAL env var.",
    )
    parser.add_argument(
        "--eval-dataset",
        default=os.environ.get("QAIHM_EVAL_DATASET") or DEFAULT_DEVICE_EVAL_DATASET,
        help="Device-eval dataset to run with --run-eval. Defaults to the "
        "QAIHM_EVAL_DATASET env var, else %(default)s.",
    )
    parser.add_argument(
        "--no-run-perf",
        dest="run_perf",
        action="store_false",
        default=os.environ.get("QAIHM_RUN_PERF", "true").lower() != "false",
        help="Skip the TPS/TTFT perf sweep (and perf.yaml updates), collecting "
        "only the accuracy eval. Defaults to the QAIHM_RUN_PERF env var (on "
        "unless set to 'false').",
    )


def _resolve_eval_items(args: argparse.Namespace) -> list[DeviceEvalItem] | None:
    if not args.run_eval:
        return None
    return load_device_eval_items(args.eval_dataset)


def _eval_prompts_for_device(
    eval_items: list[DeviceEvalItem] | None, sd: ScorecardDevice, devices_setting: str
) -> list[DeviceEvalItem] | None:
    """Restrict the accuracy eval to the run's single accuracy device.

    Shares ``resolve_llm_eval_device`` with the genie path, so both honor the
    same rule: cs_x_elite normally, or the one device the caller named when
    they named exactly one.
    """
    return eval_items if sd == resolve_llm_eval_device(devices_setting) else None


def _save_eval_results(
    eval_results: list[dict],
    model_id: str,
    sd: ScorecardDevice,
    precision: Precision,
    plugin: str,
    dataset_name: str,
) -> None:
    if not eval_results:
        return
    profile_path = (
        ScorecardProfilePath.GENIEX_QAIRT
        if plugin == "qairt"
        else ScorecardProfilePath.GENIEX_LLAMACPP
    )
    base = f"{model_id}_{sd.chipset}_{precision}_{plugin}_eval"
    save_eval_results_json(eval_results, f"{base}.json")
    # Identity sidecar for collect_llm_accuracy_csv; without it the grade file
    # is dropped and no accuracy.csv row is written.
    save_eval_metadata_json(
        model_id,
        sd.chipset,
        str(precision),
        f"{base}.meta.json",
        path=profile_path,
        dataset_name=dataset_name,
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", default="geniex_bench_results.csv")
    parser.add_argument("--skip-perf-update", action="store_true")
    LLMPerfUpdatesEnvvar.add_arg(parser, default=Path("geniex_perf_updates.jsonl"))


def _resolve_qairt_version(args: argparse.Namespace) -> str | None:
    """Software Center SDK version to swap in on-device, or None for llama_cpp."""
    if args.plugin == "llama_cpp":
        return None
    return QAIRTVersionEnvvar.get_on_device_sdk_version(
        TargetRuntime.GENIEX_QAIRT, args.qairt_version
    )


def _cmd_submit(args: argparse.Namespace) -> int:
    eval_items = _resolve_eval_items(args)
    qairt_version = _resolve_qairt_version(args)
    if os.path.exists(args.jobs_file):
        os.unlink(args.jobs_file)
    submitted = 0
    failed_cases: list[str] = []
    for (
        plugin,
        model_id,
        precision,
        device_token,
        model_ref,
        ctx_list,
        llamacpp_quant,
    ) in _iter_work(
        args.models,
        args.devices,
        args.plugin,
        args.precisions,
        args.results_dir,
        args.geniex_version,
    ):
        case_name = f"{model_id}/{plugin}/{precision}@{device_token}"
        try:
            _submit_one(
                model_id,
                model_ref,
                device_token,
                ctx_list,
                args.results_dir,
                plugin,
                args.geniex_version,
                precision,
                args.jobs_file,
                llamacpp_quant=llamacpp_quant,
                eval_items=_eval_prompts_for_device(
                    eval_items, _scorecard_device(device_token), args.devices
                ),
                run_perf=args.run_perf,
                qairt_version=qairt_version if plugin == "qairt" else None,
            )
            submitted += 1
        except Exception as e:
            print(
                f"ERROR: submission failed for {case_name}: {e}",
                file=sys.stderr,
            )
            failed_cases.append(case_name)
    print(f"Submitted {submitted} geniex-bench job(s) to {args.jobs_file}")
    if failed_cases:
        print(f"FAILED submissions: {failed_cases}", file=sys.stderr)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write("## GenieX-bench Submission Failures\n\n")
                f.writelines(f"- {c}\n" for c in failed_cases)
                f.write("\n")
    # A total wipeout is still its own failure even with no cases at all
    # (empty _iter_work) so CI doesn't read a silent success.
    return 1 if failed_cases or not submitted else 0


def _cmd_collect(args: argparse.Namespace) -> int:
    """Re-iterate the submit-side work list and pick up each job by key.

    The jobs_file carries just the job_id, so we re-run the
    same _iter_work discovery the submit side used and look each
    (model, precision, plugin, device) tuple up in the jobs_file by key.
    """
    eval_items = _resolve_eval_items(args)
    rows: list[dict] = []
    records = load_jobs(args.jobs_file)
    for (
        plugin,
        model_id,
        precision,
        device_token,
        _model_ref,
        ctx_list,
        _llamacpp_quant,
    ) in _iter_work(
        args.models,
        args.devices,
        args.plugin,
        args.precisions,
        args.results_dir,
        args.geniex_version,
    ):
        sd = _scorecard_device(device_token)
        runtime = "GENIEX_QAIRT" if plugin == "qairt" else "GENIEX_LLAMACPP"
        key = make_key(model_id, str(precision), runtime, sd.name)
        record = records.get(key)
        if record is None:
            print(
                f"jobs_file has no entry for {key}; submission must have failed",
                file=sys.stderr,
            )
            rows.append(
                {
                    "model": model_id,
                    "plugin": plugin,
                    "precision": str(precision),
                    "device": sd.name,
                    "status": "not_submitted",
                }
            )
            continue

        # Recorded before collection so a bucket that fails below is still
        # dropped by apply_llm_perf_updates rather than keeping stale numbers.
        if args.run_perf and not args.skip_perf_update:
            record_perf_scope(
                model_id=model_id,
                profile_path=(
                    ScorecardProfilePath.GENIEX_QAIRT
                    if plugin == "qairt"
                    else ScorecardProfilePath.GENIEX_LLAMACPP
                ),
                device_name=sd.reference_device_name,
                precision=precision,
            )

        try:
            metrics, eval_results, status, job_failure = _collect_one(
                model_id=model_id,
                precision=precision,
                device_name=sd.name,
                plugin=plugin,
                record=record,
                save_dir_root=args.results_dir,
                ctx_list=ctx_list,
                eval_items=_eval_prompts_for_device(eval_items, sd, args.devices),
                run_perf=args.run_perf,
            )
        except Exception as e:
            print(
                f"ERROR: collection failed for {model_id} @ {sd.name}: {e}",
                file=sys.stderr,
            )
            rows.append(
                {
                    "model": model_id,
                    "plugin": plugin,
                    "precision": str(precision),
                    "device": sd.name,
                    "status": "failed",
                }
            )
            continue

        _record_outcome(
            rows,
            model_id,
            sd,
            plugin,
            precision,
            metrics,
            eval_results,
            status,
            job_failure,
            args.skip_perf_update,
            args.eval_dataset,
        )

    return _write_final_outputs(rows, args.csv)


def _status_row(
    model_id: str, plugin: str, precision: Precision, sd: ScorecardDevice, status: str
) -> dict:
    return {
        "model": model_id,
        "plugin": plugin,
        "precision": str(precision),
        "device": sd.name,
        "status": status,
    }


def _record_outcome(
    rows: list[dict],
    model_id: str,
    sd: ScorecardDevice,
    plugin: str,
    precision: Precision,
    metrics: list[GenieXBenchMetrics],
    eval_results: list[dict],
    status: str,
    job_failure: str | None,
    skip_perf_update: bool,
    eval_dataset: str,
) -> None:
    """Fold one finished (model, precision, device) into the output rows."""
    _save_eval_results(eval_results, model_id, sd, precision, plugin, eval_dataset)

    # Its results are applied like any other, but the extra row keeps the
    # failed job red in the CSV, step summary, and exit code.
    if job_failure:
        print(
            f"::warning::{model_id} [{precision}] @ {sd.name}: results kept "
            f"from a failed job. {job_failure}"
        )
        rows.append(
            _status_row(model_id, plugin, precision, sd, "job_failed (results kept)")
        )

    if status != "success":
        rows.append(_status_row(model_id, plugin, precision, sd, status))
        return

    rows.extend(
        _rows_from_metrics(model_id, sd, plugin, precision, metrics, skip_perf_update)
    )


@dataclass
class _Case:
    plugin: str
    model_id: str
    precision: Precision
    sd: ScorecardDevice
    model_ref: str
    ctx_list: list[int]
    llamacpp_quant: str | None
    eval_items: list[DeviceEvalItem] | None
    job_id: str | None = None

    @property
    def name(self) -> str:
        return f"{self.model_id}/{self.plugin}/{self.precision}@{self.sd.name}"


def _is_capped_qdc(sd: ScorecardDevice) -> bool:
    # Mirrors qdc.DEDICATED_POOL_DEVICES (not imported: it pulls in the QDC SDK).
    # That pool's private account isn't subject to the shared cap.
    return sd.devicefarm_backend == "qdc" and sd not in (cs_x_elite, cs_ventuno_q)


def _cmd_run(args: argparse.Namespace) -> int:
    """Submit and collect in one loop, keeping a bounded backlog per device.

    A new job is submitted only while fewer than ``--max-waiting`` jobs are
    queued for that device (across every submitter on the account), so phones
    never idle and no job waits long enough for its presigned URL to lapse.
    Shared-pool QDC jobs are further capped at ``QDC_MAX_UNFINISHED_JOBS``
    unfinished at once across the whole account.
    """
    eval_items = _resolve_eval_items(args)
    qairt_version = _resolve_qairt_version(args)
    if os.path.exists(args.jobs_file):
        os.unlink(args.jobs_file)

    pending: list[_Case] = []
    for plugin, model_id, precision, token, model_ref, ctx, quant in _iter_work(
        args.models,
        args.devices,
        args.plugin,
        args.precisions,
        args.results_dir,
        args.geniex_version,
    ):
        sd = _scorecard_device(token)
        pending.append(
            _Case(
                plugin,
                model_id,
                precision,
                sd,
                model_ref,
                ctx,
                quant,
                _eval_prompts_for_device(eval_items, sd, args.devices),
            )
        )

    backends: dict[str, DeviceFarm] = {}
    in_flight: list[_Case] = []
    rows: list[dict] = []

    def _backend(case: _Case) -> DeviceFarm:
        return backends.setdefault(case.sd.name, get_device_farm(case.sd))

    def _submit(case: _Case) -> None:
        case.job_id = _submit_one(
            case.model_id,
            case.model_ref,
            case.sd.name,
            case.ctx_list,
            args.results_dir,
            case.plugin,
            args.geniex_version,
            case.precision,
            args.jobs_file,
            llamacpp_quant=case.llamacpp_quant,
            eval_items=case.eval_items,
            run_perf=args.run_perf,
            qairt_version=qairt_version if case.plugin == "qairt" else None,
        )
        if args.run_perf and not args.skip_perf_update:
            # Like collect: only submitted buckets are scoped, so a failed
            # submit keeps its committed numbers but a failed job drops them.
            record_perf_scope(
                model_id=case.model_id,
                profile_path=ScorecardProfilePath[_runtime(case.plugin)],
                device_name=case.sd.reference_device_name,
                precision=case.precision,
            )

    def _is_done(case: _Case) -> bool:
        try:
            return _backend(case).is_done(case.job_id or "")
        except Exception as e:
            # A throttled or flaky status call shouldn't abort every other job.
            print(f"Status check failed for {case.name} ({type(e).__name__})")
            return False

    def _finish(
        case: _Case,
        status: str,
        metrics: list[GenieXBenchMetrics],
        evals: list[dict],
        kept: str | None,
    ) -> None:
        _record_outcome(
            rows,
            case.model_id,
            case.sd,
            case.plugin,
            case.precision,
            metrics,
            evals,
            status,
            kept,
            args.skip_perf_update,
            args.eval_dataset,
        )

    def _collect(case: _Case) -> None:
        assert case.job_id is not None
        metrics, evals, outcome, reason, kept = _collect_attempt(
            _backend(case),
            case.sd,
            case.model_id,
            case.precision,
            case.plugin,
            case.job_id,
            args.results_dir,
            case.ctx_list,
            case.eval_items,
            args.run_perf,
        )
        if outcome is JobOutcome.SUCCESS:
            status = _final_status(metrics, evals, args.run_perf, case.eval_items)
            _finish(case, status, metrics, evals, kept)
        else:
            print(f"ERROR: {reason} for {case.name}", file=sys.stderr)
            _finish(case, "failed", [], [], None)

    while pending or in_flight:
        for case in [c for c in in_flight if _is_done(c)]:
            in_flight.remove(case)
            try:
                _collect(case)
            except Exception as e:
                print(f"ERROR: collection failed for {case.name}: {e}", file=sys.stderr)
                _finish(case, "failed", [], [], None)

        # One fresh queue count per device per pass, bumped locally as we submit.
        waiting: dict[str, int | None] = {}
        qdc_unfinished: int | None = None
        for case in list(pending):
            device = case.sd.name
            if _is_capped_qdc(case.sd):
                if qdc_unfinished is None:
                    qdc_unfinished = _backend(case).count_unfinished_jobs()
                # Wait for an account slot to free rather than eating the 400.
                if qdc_unfinished is None or qdc_unfinished >= QDC_MAX_UNFINISHED_JOBS:
                    continue
            if device not in waiting:
                waiting[device] = _backend(case).count_waiting_jobs(
                    case.sd.reference_device_name
                )
            count = waiting[device]
            if count is not None and count >= args.max_waiting:
                continue
            pending.remove(case)
            try:
                _submit(case)
            except Exception as e:
                print(f"ERROR: submission failed for {case.name}: {e}", file=sys.stderr)
                _finish(case, "not_submitted", [], [], None)
                continue
            in_flight.append(case)
            if _is_capped_qdc(case.sd) and qdc_unfinished is not None:
                qdc_unfinished += 1
            if count is not None:
                waiting[device] = count + 1

        if pending or in_flight:
            time.sleep(args.poll_interval)

    return _write_final_outputs(rows, args.csv)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run geniex-bench benchmarks on QDC.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser(
        "submit", help="Submit jobs; append one row per submission to a jobs file."
    )
    _add_shared_args(p_submit)
    p_submit.add_argument(
        "--jobs-file",
        default="geniex_jobs.yaml",
        help="Path to write the YAML jobs file for the collect step.",
    )

    p_collect = sub.add_parser(
        "collect", help="Poll jobs listed in the jobs file and emit CSV/JSONL."
    )
    _add_shared_args(p_collect)
    p_collect.add_argument(
        "--jobs-file",
        default="geniex_jobs.yaml",
        help="Path to the YAML jobs file produced by ``submit``.",
    )
    _add_output_args(p_collect)

    p_run = sub.add_parser(
        "run",
        help="Submit and collect in one loop, capping each device's queued backlog.",
    )
    _add_shared_args(p_run)
    p_run.add_argument(
        "--jobs-file",
        default="geniex_jobs.yaml",
        help="Path to record submitted job ids (for debugging/resume).",
    )
    p_run.add_argument(
        "--max-waiting",
        type=int,
        default=DEFAULT_MAX_WAITING,
        help="Max jobs queued (not yet on a device) per device type. Running "
        "jobs never count, so every phone the farm frees up is used.",
    )
    p_run.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_RUN_POLL_INTERVAL,
        help="Seconds between status passes over in-flight jobs.",
    )
    _add_output_args(p_run)

    args = ap.parse_args()

    if not args.run_perf and not args.run_eval:
        print(
            "ERROR: nothing to do -- perf is disabled (--no-run-perf) and "
            "--run-eval was not given.",
            file=sys.stderr,
        )
        return 1

    if args.cmd == "submit":
        return _cmd_submit(args)
    if args.cmd == "run":
        return _cmd_run(args)
    return _cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
