# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Merge LLM perf.yaml update logs from every runtime into the checkout.

Each perf job (genie, geniex) emits a JSON-lines log with two record kinds
(see templates/llm/perf_collection.py): a "scope" line per bucket the run
intended to measure, and a "metric" line per measurement it produced. This
script drops every in-scope bucket and re-adds only the measured ones, so a
device that failed loses its row rather than keeping stale numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from filelock import FileLock

from qai_hub_models import Precision
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models.templates.llm.perf_collection import update_perf_yaml
from qai_hub_models.scorecard.device import DEFAULT_QDC_DEVICE, ScorecardDevice
from qai_hub_models.scorecard.devices_and_chipsets_yaml import load_similar_devices
from qai_hub_models.scorecard.internal_config_yaml import model_perf_component_key
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.release_assets_yaml import (
    NEVER_DROPPED,
    QAIHMModelReleaseAssets,
)
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT

# Genie runs on DEFAULT_QDC_DEVICE and nowhere else, so that one job decides
# whether a precision's release assets are trustworthy at all. gemma_4_e4b_it is
# measured beyond it, so one failure says nothing about its other assets.
# Remove once gemma4 runs on genie: qcom-ai-hub/tetracode#20995
ASSET_REMOVAL_EXEMPT_MODELS = frozenset({"gemma_4_e4b_it"})


def _load_updates_file(path: Path) -> list[dict]:
    """Load one JSON-lines updates file (one update dict per line)."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def collect_updates(paths: list[Path]) -> list[dict]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.jsonl")))
        elif p.exists():
            files.append(p)
        else:
            print(f"Updates file {p} does not exist; skipping.")

    updates: list[dict] = []
    for f in files:
        entries = _load_updates_file(f)
        print(f"Loaded {len(entries)} updates from {f}")
        updates.extend(entries)
    return updates


def _drop_release_assets(model_id: str, precision: Precision) -> set[str]:
    """Remove every droppable asset for one precision. Returns runtimes removed."""
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    prec_details = assets.precisions.get(precision)
    if prec_details is None:
        return set()
    scope: set[tuple[Precision, str | None, ScorecardProfilePath]] = {
        (precision, None, path) for path in prec_details.universal_assets
    }
    for chipset, paths in prec_details.chipset_assets.items():
        scope.update((precision, chipset, path) for path in paths)
    removed = {path.name for (_, _, path) in scope if path not in NEVER_DROPPED}
    if not removed:
        return set()
    assets.drop_entries_in_scope(scope)
    assets.to_model_yaml(model_id)
    return removed


def _scope_key(u: dict) -> tuple[str, Precision, ScorecardProfilePath, ScorecardDevice]:
    return (
        u["model_id"],
        Precision.parse(u["precision"]),
        ScorecardProfilePath(u["profile_path"]),
        ScorecardDevice.get(u["device_name"], return_unregistered=True),
    )


def apply_updates(updates: list[dict]) -> int:
    if not updates:
        print("No perf updates to apply; nothing to do.")
        return 0

    metrics = [u for u in updates if u.get("kind", "metric") == "metric"]

    # Scope is the matrix this run intended to measure, not what it managed to
    # measure, so an in-scope device that failed is dropped and never re-added.
    # Matches the non-LLM writer, which scopes from ModelTestConfig.profile_tests.
    # Metrics imply intent, covering logs written before scope records existed.
    scope_by_model: dict[
        str, set[tuple[Precision, ScorecardProfilePath, ScorecardDevice]]
    ] = defaultdict(set)
    for u in updates:
        model_id, precision, path, device = _scope_key(u)
        scope_by_model[model_id].add((precision, path, device))

    for model_id, scope in scope_by_model.items():
        perf_path = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
        if not perf_path.exists():
            continue
        with FileLock(f"{perf_path}.lock"):
            perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
            # This writer only ever produces the consolidated backbone entry (see
            # _update_perf_yaml_locked). Anything else in this perf.yaml is a standalone
            # component owned by the scorecard, which runs earlier in the same workflow.
            backbone = model_perf_component_key(model_id, None)
            perf.drop_entries_in_scope(scope, only_components={backbone})
            perf.to_model_yaml(model_id)

    for u in metrics:
        update_perf_yaml(
            model_id=u["model_id"],
            device_name=u["device_name"],
            precision=Precision.parse(u["precision"]),
            context_length=u["context_length"],
            tps=u["tps"],
            prefill_tps=u["prefill_tps"],
            profile_path=ScorecardProfilePath(u["profile_path"]),
            desired_compute_unit=u["desired_compute_unit"],
            tool_versions=ToolVersions(**u["tool_versions"])
            if u.get("tool_versions")
            else None,
        )

    similar_devices_mapping = load_similar_devices()
    for model_id in scope_by_model:
        perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        if perf.empty:
            continue
        perf.apply_similar_devices(similar_devices_mapping)
        perf.apply_compute_peer_chipsets()
        perf.to_model_yaml(model_id)

    measured = {_scope_key(u) for u in metrics}
    intended = {(model_id, *t) for model_id, s in scope_by_model.items() for t in s}

    for model_id, precision in sorted(
        {
            (m, p)
            for (m, p, path, device) in intended - measured
            if path is ScorecardProfilePath.GENIE and device == DEFAULT_QDC_DEVICE
        },
        key=lambda k: (k[0], str(k[1])),
    ):
        if model_id in ASSET_REMOVAL_EXEMPT_MODELS:
            print(
                f"{model_id}: genie failed on {DEFAULT_QDC_DEVICE.name} at "
                f"{precision}; exempt, keeping its release assets."
            )
            continue
        if removed := _drop_release_assets(model_id, precision):
            print(
                f"{model_id}: genie failed on {DEFAULT_QDC_DEVICE.name} at "
                f"{precision}; removed {', '.join(sorted(removed))} assets."
            )

    print(
        f"Applied {len(metrics)} perf metrics across {len(scope_by_model)} models; "
        f"{len(intended - measured)} in-scope buckets reported nothing and were removed."
    )
    for model_id in sorted(scope_by_model):
        print(f"  {model_id}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="+",
        help="Perf updates files and/or directories to scan for *.jsonl.",
    )
    args = ap.parse_args()
    return apply_updates(collect_updates([Path(p) for p in args.paths]))


if __name__ == "__main__":
    raise SystemExit(main())
