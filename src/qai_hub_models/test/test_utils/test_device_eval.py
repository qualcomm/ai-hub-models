# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from qai_hub_models.datasets.device_eval import (
    DEFAULT_DEVICE_EVAL_DATASET,
    DeviceEvalItem,
    get_device_eval_dataset,
    is_prompt_task,
    label_eval_results,
    load_device_eval_items,
)
from qai_hub_models.datasets.grace import Grace2
from qai_hub_models.models.templates.llm.grader.grace import (
    load_default_eval_prompts,
    load_eval_prompts,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scripts import run_genie_benchmarks
from qai_hub_models.scripts.llm import collect_llm_accuracy_csv
from qai_hub_models.scripts.llm.grade_responses import _sidecar_metric_name
from qai_hub_models.utils.llm.eval_io import save_eval_metadata_json
from qai_hub_models.utils.metrics import GRACE2_GRADE


def test_grace2_is_the_default_device_dataset() -> None:
    assert get_device_eval_dataset(DEFAULT_DEVICE_EVAL_DATASET) is Grace2
    assert Grace2.metric_metadata() is GRACE2_GRADE


def test_grace2_items_match_the_prompt_file() -> None:
    items = load_device_eval_items("grace2")
    assert [item.prompt for item in items] == load_default_eval_prompts()
    assert [(item.idx, item.category) for item in items] == [
        (p.idx, p.category) for p in load_eval_prompts()
    ]


def test_unknown_dataset_names_the_registered_ones() -> None:
    with pytest.raises(ValueError, match="grace2"):
        get_device_eval_dataset("no_such_dataset")


def test_prompt_tasks() -> None:
    assert is_prompt_task("grace2")
    assert is_prompt_task("multimodal_prompts")
    assert not is_prompt_task("wikitext")


def test_label_eval_results_maps_positions_to_dataset_items() -> None:
    items = [DeviceEvalItem(7, "a", "cat_a"), DeviceEvalItem(42, "b", "cat_b")]
    results = [
        {"idx": 1, "prompt": "b", "output": "out b"},
        {"idx": 0, "prompt": "a", "output": "out a"},
        {"idx": 5, "prompt": "", "output": "stray"},
    ]
    assert label_eval_results(results, items) == [
        {"idx": 42, "category": "cat_b", "prompt": "b", "output": "out b"},
        {"idx": 7, "category": "cat_a", "prompt": "a", "output": "out a"},
    ]


@pytest.mark.parametrize(
    ("env", "argv", "expected"),
    [
        ({}, [], DEFAULT_DEVICE_EVAL_DATASET),
        ({"QAIHM_EVAL_DATASET": "from_env"}, [], "from_env"),
        ({"QAIHM_EVAL_DATASET": "from_env"}, ["--eval-dataset", "flag"], "flag"),
    ],
)
@pytest.mark.parametrize("cmd", ["submit", "collect"])
def test_genie_runner_passes_the_eval_dataset(
    monkeypatch: pytest.MonkeyPatch,
    cmd: str,
    env: dict[str, str],
    argv: list[str],
    expected: str,
) -> None:
    monkeypatch.delenv("QAIHM_EVAL_DATASET", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        "sys.argv", ["run_genie_benchmarks", cmd, "--jobs-file", "j.yaml", *argv]
    )
    with patch.object(run_genie_benchmarks, f"_cmd_{cmd}", return_value=0) as run:
        assert run_genie_benchmarks.main() == 0
    assert run.call_args.args[0].eval_dataset == expected


def test_sidecar_carries_dataset_and_metric(tmp_path: Path) -> None:
    meta = tmp_path / "m_eval.meta.json"
    save_eval_metadata_json(
        "m", "chip", "w4a16", str(meta), ScorecardProfilePath.GENIE, "grace2"
    )
    assert json.loads(meta.read_text())["dataset_name"] == "grace2"
    assert _sidecar_metric_name(str(tmp_path / "m_eval.json")) == "Grace2"


def test_collect_files_the_row_under_the_dataset_metric(tmp_path: Path) -> None:
    save_eval_metadata_json(
        "m",
        "chip",
        "w4a16",
        str(tmp_path / "m_eval.meta.json"),
        ScorecardProfilePath.GENIE,
        "grace2",
    )
    (tmp_path / "m_eval_grade.json").write_text(
        json.dumps({"score_pct": 81.5, "num_items": 100})
    )
    with (
        patch.object(collect_llm_accuracy_csv, "_reference_grade", return_value=None),
        patch.object(collect_llm_accuracy_csv, "write_accuracy") as write,
    ):
        assert collect_llm_accuracy_csv.collect(str(tmp_path)) == 1
    kwargs = write.call_args.kwargs
    assert kwargs["dataset_name"] == "grace2"
    assert kwargs["metric_metadata"] is GRACE2_GRADE
    assert kwargs["dataset_metadata"].link == Grace2.doc_url()
