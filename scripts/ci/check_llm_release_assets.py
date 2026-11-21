# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Report which asset-upload shards made it into the combined LLM release-assets.yaml.

The asset-upload matrix is ``fail-fast: false`` and every shard uploads its own
release-assets.yaml, so combine runs on partial success rather than demanding
every shard pass. Models missing from the combined file fall back to their
committed release-assets.yaml (``load_release_assets_for_model``), so a missing
shard is a warning, not a failure -- this exists so the cause is visible in the
job summary instead of surfacing later as a per-model "No release asset found".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_models(release_assets: Path) -> list[str]:
    if not release_assets.exists():
        return []
    with open(release_assets) as f:
        data = yaml.safe_load(f) or {}
    return sorted((data.get("models") or {}).keys())


def missing_splits(expected: list[str], downloaded_dir: Path) -> list[str]:
    """Expected split names with no downloaded artifact directory.

    download-artifact names each subdirectory after the artifact, which ends in
    ", <split_name>)", so a substring match is enough to pair them up.
    """
    if not downloaded_dir.exists():
        return sorted(expected)
    present = [d.name for d in downloaded_dir.iterdir() if d.is_dir()]
    return sorted(
        name for name in expected if not any(name in dirname for dirname in present)
    )


def parse_expected(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return [str(entry["split_name"]) for entry in json.loads(raw)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-assets", type=Path, required=True)
    ap.add_argument("--downloaded-dir", type=Path, required=True)
    ap.add_argument(
        "--split-matrix",
        default=os.environ.get("LLM_SPLIT_MATRIX", ""),
        help="determine_llm_models' llm_split_matrix JSON.",
    )
    ap.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = ap.parse_args()

    expected = parse_expected(args.split_matrix)
    absent = missing_splits(expected, args.downloaded_dir)
    models = load_models(args.release_assets)

    print(f"Expected {len(expected)} asset-upload shard(s): {expected or '<unknown>'}")
    print(f"Models with fresh release assets ({len(models)}): {models or '<none>'}")
    if absent:
        print(
            f"::warning::No release assets from shard(s) {absent}. Those models fall "
            "back to their committed release-assets.yaml; every other model collects "
            "perf normally."
        )
    if not models:
        print(
            "::warning::Combined release-assets.yaml lists no models; collection runs "
            "entirely against committed assets."
        )

    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"fresh_asset_models={','.join(models)}\n")
            f.write(f"missing_splits={','.join(absent)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
