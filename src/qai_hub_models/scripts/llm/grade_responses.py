#!/usr/bin/env python3
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
r"""CLI for grading LLM responses.

Reads a JSON file containing a list of items in the form::

    [
      {"idx": 0, "category": "knowledge", "prompt": "What is gravity?",
       "output": "Gravity is ..."},
      ...
    ]

The ``prompt`` is passed through to the grader as-is, and ``output`` is the
generated response to grade. Grading is delegated to
:mod:`qai_hub_models.models.templates.llm.grader.grader`.

A closing pass sends the rationales back to the grader model and condenses them
into at most five recurring failure modes, printed last and stored under
``summary_items`` in ``--output-json``. Pass ``--no-summary`` to skip it.

Example usage::

    python -m qai_hub_models.scripts.llm.grade_responses responses.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from qai_hub_models.models.templates.llm.grace_tasks import GRACE_METRIC_NAME
from qai_hub_models.models.templates.llm.grader.grader import (
    DEFAULT_PROMPT_TEMPLATE,
    MAX_POINTS,
    ResponseGrader,
    resolve_device,
)
from qai_hub_models.models.templates.llm.grader.report import (
    build_summary,
    category_scores,
    resolve_categories,
)

GRADER_SEED = 42


def _sidecar_metric_name(responses_json: str) -> str:
    """The metric recorded next to a device eval file (``X.json`` -> ``X.meta.json``).

    Read from the sidecar rather than the dataset registry: this CLI runs in the
    grader venv, which installs qai_hub_models without its dependencies.
    """
    meta_path = Path(responses_json).with_suffix(".meta.json")
    if meta_path.exists():
        name = json.loads(meta_path.read_text()).get("metric_name")
        if name:
            return name
    return GRACE_METRIC_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade LLM responses from a JSON file on a 0-10 rubric.",
    )
    parser.add_argument(
        "responses_json",
        type=str,
        nargs="?",
        help="Path to a JSON file: list of {idx, prompt, output} objects.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to grading prompt template (must contain {response}). "
        "If omitted, the default template is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model id to use as grader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the grader model (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit grading on CPU. Without this, an unusable CUDA install is a "
        "hard error rather than a ~13-minute-per-file CPU fallback.",
    )
    parser.add_argument(
        "--check-device-only",
        action="store_true",
        help="Resolve and print the grader device, then exit. Lets CI fail fast on "
        "a misconfigured venv without loading a model or reading responses.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Grader model dtype.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write a machine-readable summary to this path.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the closing summary pass over the rationales.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the per-item score and the grader's rationale.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default=None,
        help="Label the score is filed under. Defaults to the metric_name in the "
        f"responses file's .meta.json sidecar, else {GRACE_METRIC_NAME}.",
    )
    args = parser.parse_args()

    # Keep this before the model loads: once CUDA has started,
    # CUBLAS_WORKSPACE_CONFIG is silently ignored.
    _make_deterministic()

    # Resolved up front so a misconfigured venv fails before a model download.
    try:
        device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    except RuntimeError as e:
        raise SystemExit(str(e)) from None
    if args.check_device_only:
        print(f"grader device: {device}")
        return
    if args.responses_json is None:
        parser.error("responses_json is required unless --check-device-only is set.")

    if args.prompt_file:
        prompt_template = Path(args.prompt_file).read_text()
    else:
        prompt_template = DEFAULT_PROMPT_TEMPLATE

    items = json.loads(Path(args.responses_json).read_text())
    if not items:
        raise ValueError(f"No items found in {args.responses_json}")
    print(f"Loaded {len(items)} items from {args.responses_json}")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    grader = ResponseGrader(
        model_id=args.model,
        device=device,
        dtype=dtype_map[args.dtype],
        prompt_template=prompt_template,
        allow_cpu=args.allow_cpu,
    )

    summary = grader.grade(items, summary=not args.no_summary)

    if args.verbose:
        for item, result in zip(items, summary.results, strict=True):
            print(
                f"  idx={item['idx']}: {result.points:2d} pts"
                + ("  [skipped: empty response]" if result.skipped else "")
                + ("  [rating forced after token limit]" if result.forced else "")
                + ("  [GRADER FAILURE: no rating]" if not result.parsed else "")
            )
            if result.rationale:
                print(f"      {result.rationale}")

    print()
    print("=" * 60)
    print(f"Grader: {args.model}")
    print(f"Responses graded: {len(items)}")
    print("=" * 60)
    print(
        f"Overall score: {summary.score_pct:.1f}%  "
        f"({summary.total_points}/{summary.max_points} pts)"
    )
    if summary.num_forced:
        print(
            f"Note: {summary.num_forced} item(s) ran out of tokens before rating; "
            f"the rating was recovered by a forced second pass."
        )
    if summary.num_unparsed:
        print(
            f"GRADER FAILURE: {summary.num_unparsed} item(s) produced no readable "
            f"rating and were scored 0. The score above is a floor, not a "
            f"measurement — fix the grader and re-run before trusting it."
        )
    print()
    flawed = sorted(
        (result.points, item["idx"], result.rationale)
        for item, result in zip(items, summary.results, strict=True)
        if result.points < MAX_POINTS
    )
    if flawed:
        print(f"Items scoring below {MAX_POINTS}, worst first:")
        for points, idx, rationale in flawed:
            print(f"  - idx={idx} ({points}/{MAX_POINTS}): {rationale}")
        print()

    categories = resolve_categories(items)
    per_category = category_scores(categories, summary.results)
    if per_category:
        print("By category:")
        for name, (pct, pts, num) in sorted(
            per_category.items(), key=lambda kv: kv[1][0]
        ):
            print(f"  {name:15s} {pct:5.1f}%  ({pts}/{MAX_POINTS * num} pts, n={num})")
        print()

    if summary.summary_items:
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        for number, text in enumerate(summary.summary_items, start=1):
            print(f"  {number}. {text}")
        print()

    if args.output_json:
        out = build_summary(
            items,
            summary,
            metric_name=args.metric_name or _sidecar_metric_name(args.responses_json),
            grader_model=args.model,
            input_file=str(args.responses_json),
        )
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"Wrote grading summary to {args.output_json}")


def _make_deterministic() -> None:
    """Make the grader's output repeat bit for bit.

    Decoding is greedy, so a tiny numeric difference can flip a near-tied token
    and change the rest of the rationale, and sometimes the rating.
    """
    # Read only when the CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # warn_only: an op with no deterministic kernel warns instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # cuDNN attention, which torch picks for fp16/bf16 on H100, is not
    # repeatable, and the deterministic flag above does not cover it.
    # Torch falls back to flash attention.
    torch.backends.cuda.enable_cudnn_sdp(False)
    # Nothing random is expected in a greedy decode; seeded to be safe.
    torch.manual_seed(GRADER_SEED)
    print(
        f"Grader deterministic mode: seed={GRADER_SEED}, "
        f"CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
        f"deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}, "
        f"cudnn_sdp={torch.backends.cuda.cudnn_sdp_enabled()}"
    )


if __name__ == "__main__":
    main()
