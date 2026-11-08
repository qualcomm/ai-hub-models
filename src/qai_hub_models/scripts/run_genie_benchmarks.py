# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import TYPE_CHECKING

from qai_hub_models.datasets.device_eval import DEFAULT_DEVICE_EVAL_DATASET

if TYPE_CHECKING:
    from qai_hub_models.scorecard.device import ScorecardDevice

DEFAULT_MAX_WAITING = 2
DEFAULT_POLL_INTERVAL = 60


def _write_junit(junit_path: str, cases: list[tuple[str, str | None]]) -> None:
    """Minimal junit XML compatible with generate_test_summary."""
    import xml.etree.ElementTree as ET

    root = ET.Element(
        "testsuite",
        {
            "name": "llm_perf",
            "tests": str(len(cases)),
            "failures": str(sum(1 for _, m in cases if m)),
        },
    )
    for name, msg in cases:
        tc = ET.SubElement(root, "testcase", {"classname": "llm_perf", "name": name})
        if msg:
            fail = ET.SubElement(tc, "failure", {"message": msg[:200]})
            fail.text = msg
    pathlib.Path(junit_path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(junit_path, encoding="utf-8", xml_declaration=True)


def _cmd_submit(args: argparse.Namespace) -> int:
    from qai_hub_models.models.templates.llm.perf_collection import (
        LLMPerfConfig,
        get_llm_eval_device,
    )
    from qai_hub_models.scorecard.test.test_llm_perf import _build_params
    from qai_hub_models.utils.devicefarm.devicefarm import get_device_farm
    from qai_hub_models.utils.llm.genie.jobs import (
        GENIE_BUNDLES_ROOT,
        submit_llm_perf_job,
    )

    def _wait_for_queue_room(device: ScorecardDevice) -> None:
        # A job queued too long outlives its 12h presigned URL; backends that
        # can't count their queue (QDC) return None and are never throttled.
        backend = get_device_farm(device)
        while (
            waiting := backend.count_waiting_jobs(device.reference_device_name)
        ) is not None and waiting >= args.max_waiting:
            print(f"{waiting} job(s) queued for {device.name}; waiting to submit more")
            time.sleep(args.poll_interval)

    if os.path.exists(args.jobs_file):
        os.unlink(args.jobs_file)
    cfg = LLMPerfConfig.from_environment()
    eval_device = get_llm_eval_device()
    print(f"On-device accuracy: {eval_device.name if eval_device else 'off'}")
    submitted = 0
    cases: list[tuple[str, str | None]] = []
    for model_id, precision, device in _build_params():
        case_name = f"{model_id}-{precision}-{device.name}"
        try:
            _wait_for_queue_room(device)
            submit_llm_perf_job(
                model_id=model_id,
                device=device,
                precision=precision,
                output_dir=os.path.join(model_id, GENIE_BUNDLES_ROOT),
                jobs_file=args.jobs_file,
                qairt_sdk_path=cfg.qairt_sdk_path,
                skip_perf_update=cfg.skip_perf_update,
                eval_dataset=args.eval_dataset,
            )
            submitted += 1
            cases.append((case_name, None))
        except Exception as e:
            msg = f"submission failed for {case_name}: {e}"
            print(f"ERROR: {msg}", file=sys.stderr)
            cases.append((case_name, msg))
    print(f"Submitted {submitted} genie job(s) to {args.jobs_file}")

    if args.junit_xml:
        _write_junit(args.junit_xml, cases)

    failed = [name for name, msg in cases if msg]
    if failed:
        print(f"FAILED submissions: {failed}", file=sys.stderr)
    # A total wipeout is still its own failure even with no cases at all
    # (empty _build_params) so CI doesn't read a silent success.
    return 1 if failed or not submitted else 0


def _cmd_collect(args: argparse.Namespace) -> int:
    from qai_hub_models.models.templates.llm.llm_helpers import (
        log_perf_on_device_result,
    )
    from qai_hub_models.models.templates.llm.perf_collection import LLMPerfConfig
    from qai_hub_models.scorecard.test.test_llm_perf import _build_params
    from qai_hub_models.utils.devicefarm.devicefarm import load_jobs, make_key
    from qai_hub_models.utils.llm.genie.jobs import (
        GENIE_BUNDLES_ROOT,
        collect_llm_perf_job,
    )

    if not os.path.exists(args.jobs_file):
        print(f"jobs file not found: {args.jobs_file}", file=sys.stderr)
        return 1

    cfg = LLMPerfConfig.from_environment()
    records = load_jobs(args.jobs_file)
    cases: list[tuple[str, str | None]] = []
    for model_id, precision, device in _build_params():
        key = make_key(model_id, str(precision), "GENIE", device.name)
        record = records.get(key)
        case_name = f"{model_id}-{precision}-{device.name}"
        if record is None:
            msg = f"no jobs_file entry for {key}; submission must have failed"
            print(f"ERROR: {msg}", file=sys.stderr)
            cases.append((case_name, msg))
            continue
        try:
            tps, ttft, prefill_tps = collect_llm_perf_job(
                model_id=model_id,
                device=device,
                precision=precision,
                record=record,
                jobs_file=args.jobs_file,
                output_dir=os.path.join(model_id, GENIE_BUNDLES_ROOT),
                qairt_sdk_path=cfg.qairt_sdk_path,
                skip_perf_update=cfg.skip_perf_update,
                eval_dataset=args.eval_dataset,
            )
        except Exception as e:
            print(
                f"ERROR: collection failed for {case_name} (job {record.job_id}): {e}",
                file=sys.stderr,
            )
            cases.append((case_name, str(e)))
            continue
        log_perf_on_device_result(
            model_name=model_id,
            precision=str(precision),
            device=device.name,
            tps=tps,
            prefill_tps=prefill_tps,
            ttft_ms=ttft,
        )
        cases.append((case_name, None))

    if args.junit_xml:
        _write_junit(args.junit_xml, cases)

    failed = [name for name, msg in cases if msg]
    if failed:
        print(f"FAILED cases: {failed}", file=sys.stderr)
        return 1
    return 0


def _add_eval_dataset_arg(parser: argparse.ArgumentParser) -> None:
    # Submit and collect must agree: collect re-parses the logs against this dataset's prompts.
    parser.add_argument(
        "--eval-dataset",
        default=os.environ.get("QAIHM_EVAL_DATASET") or DEFAULT_DEVICE_EVAL_DATASET,
        help="Device-eval dataset to run when on-device accuracy is on. Defaults "
        "to the QAIHM_EVAL_DATASET env var, else %(default)s.",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser(
        "submit", help="Submit one device-farm job per (model, precision, device)."
    )
    p_submit.add_argument("--jobs-file", required=True)
    p_submit.add_argument("--junit-xml", default=None)
    p_submit.add_argument(
        "--max-waiting",
        type=int,
        default=DEFAULT_MAX_WAITING,
        help="Max jobs queued (not yet on a device) per device type before "
        "submitting another. Running jobs never count.",
    )
    p_submit.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between queue checks while throttled.",
    )
    _add_eval_dataset_arg(p_submit)

    p_collect = sub.add_parser(
        "collect", help="Poll jobs listed in the jobs file and update perf.yaml."
    )
    p_collect.add_argument("--jobs-file", required=True)
    p_collect.add_argument("--junit-xml", default=None)
    _add_eval_dataset_arg(p_collect)

    args = ap.parse_args()
    if args.cmd == "submit":
        return _cmd_submit(args)
    return _cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
