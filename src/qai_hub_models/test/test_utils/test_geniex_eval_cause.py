# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from pathlib import Path

import pytest

from qai_hub_models.utils.llm.geniex.jobs import (
    _extract_eval_cause,
    compute_geniex_eval_results,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[gen ] hi\n[info] stop_reason=eos\n[ok  ] done", "eos"),
        ("[gen ] hi\n[info] stop_reason=length\n[ok  ] done", "length"),
        ("[gen ] hi\n[info] stop_reason=stop\n[ok  ] done", "stop"),
        (
            "[gen ] the the\n[warn] self-abort: reason=repetition after=40tok",
            "divergence:repetition",
        ),
        (
            "[warn] self-abort: reason=no-progress after=12tok",
            "divergence:no-progress",
        ),
        (
            "[warn] self-abort: reason=wall-time after=114tok",
            "divergence:wall-time",
        ),
        (
            "[warn] self-abort: reason=signal after=120tok\n[info] stop_reason=eos",
            "divergence:signal",
        ),
        ("[gen ] partial output only", "freeze"),
        ("", "freeze"),
    ],
)
def test_extract_eval_cause(raw: str, expected: str) -> None:
    assert _extract_eval_cause(raw) == expected


def test_compute_geniex_eval_results_records_cause(tmp_path: Path) -> None:
    log = (
        "===EVAL_IDX_000===\n"
        "[gen ] Hello!\n"
        "[info] stop_reason=eos\n"
        "===EVAL_IDX_001===\n"
        "[gen ] the the the\n"
        "[warn] self-abort: reason=repetition after=40tok\n"
        "===EVAL_IDX_002===\n"
    )
    (tmp_path / "geniex_eval_outputs.txt").write_text(log, encoding="utf-8")

    results = compute_geniex_eval_results(str(tmp_path), ["a", "b", "c"])

    assert [(r["idx"], r["cause"]) for r in results] == [
        (0, "eos"),
        (1, "divergence:repetition"),
        (2, "freeze"),
    ]
    assert results[1]["output"] == "the the the"
