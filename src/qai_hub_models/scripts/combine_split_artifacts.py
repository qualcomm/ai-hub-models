# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from qai_hub_models_cli.proto_helpers._common import dump_yaml, safe_load_yaml

from qai_hub_models.scorecard.envvars import EnabledModelsEnvvar, SpecialModelSetting
from qai_hub_models.utils.base_config import sorted_mappings

# YAML files that should be merged by combining their key-value pairs
MERGE_YAML_FILES = [
    "component-names.yaml",
    "graph-names.yaml",
    "compile-jobs.yaml",
    "link-jobs.yaml",
    "inference-jobs.yaml",
    "quantize-jobs.yaml",
    "dataset-ids.yaml",
    "cpu-accuracy.yaml",
    "release-assets.yaml",
]

# CSV files that should be merged by concatenating rows
MERGE_CSV_FILES = [
    "accuracy.csv",
    "stage-timings.csv",
]

# Files that should be copied (not merged) - take the first one found
COPY_FILES = [
    "tool-versions.yaml",
    "environment.env",
]


def load_yaml(yaml_filepath: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    with open(yaml_filepath) as yaml_file:
        return safe_load_yaml(yaml_file) or {}


def save_yaml(yaml_filepath: Path, data: dict) -> None:
    """Save a dict to a YAML file."""
    with open(yaml_filepath, "w") as yaml_file:
        dump_yaml(sorted_mappings(data), yaml_file)


def write_empty_release_assets(output_dir: Path) -> None:
    """Write a `models: {}` release-assets.yaml so the file always exists.

    Consumers read an empty ``models`` map as "this run produced no assets" and
    fall back to the committed per-model copies.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_yaml(output_dir / "release-assets.yaml", {"models": {}})
    print(f"Wrote empty release-assets.yaml to {output_dir}")


def find_input_dirs(input_dir: Path) -> list[Path]:
    """
    The directories to combine: every subdirectory, plus the directory itself if it
    holds any file to combine (a lone artifact extracts flat, not into a subdirectory).
    """
    input_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()],
        key=lambda x: x.name,
    )
    combinable = MERGE_YAML_FILES + MERGE_CSV_FILES + COPY_FILES
    if not input_dirs or any((input_dir / f).exists() for f in combinable):
        input_dirs.insert(0, input_dir)
    return input_dirs


def combine_split_artifacts(
    input_dirs: list[Path],
    output_dir: Path,
    models: set[str | SpecialModelSetting] | None = None,
) -> None:
    """Combine artifacts from multiple split directories into a single output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Merge YAML files by combining key-value pairs
    for yaml_file in MERGE_YAML_FILES:
        combined_data: dict = {}
        for input_dir in input_dirs:
            yaml_path = input_dir / yaml_file
            if yaml_path.exists():
                data = load_yaml(yaml_path)
                if data:
                    # release-assets.yaml nests everything under
                    # "models", so merge on that key to avoid later
                    # splits overwriting earlier ones.
                    if yaml_file == "release-assets.yaml":
                        combined_data.setdefault("models", {})
                        entries = data.get("models") or {}
                        merge_into = combined_data["models"]
                    else:
                        entries = data
                        merge_into = combined_data

                    duplicates = set(merge_into.keys()) & set(entries.keys())
                    if duplicates:
                        print(
                            f"Warning: Duplicate keys found in {yaml_file}: {duplicates}"
                        )
                    merge_into.update(entries)

        if combined_data:
            output_path = output_dir / yaml_file
            save_yaml(output_path, combined_data)
            num_entries = len(combined_data.get("models", combined_data))
            print(f"Combined {yaml_file}: {num_entries} entries")

    for csv_file in MERGE_CSV_FILES:
        header = None
        rows: list[str] = []
        for input_dir in input_dirs:
            csv_path = input_dir / csv_file
            if csv_path.exists():
                with open(csv_path) as f:
                    lines = f.readlines()
                    if lines:
                        if header is None:
                            header = lines[0]
                        rows.extend(lines[1:])  # Skip header for all files

        if header and rows:
            output_path = output_dir / csv_file
            with open(output_path, "w") as f:
                f.write(header)
                f.writelines(rows)
            print(f"Combined {csv_file}: {len(rows)} rows")

    # Copy files (take the first one found)
    for file_name in COPY_FILES:
        for input_dir in input_dirs:
            file_path = input_dir / file_name
            if file_path.exists():
                output_path = output_dir / file_name
                shutil.copy2(file_path, output_path)
                print(f"Copied {file_name} from {input_dir}")
                break

    # Fix QAIHM_TEST_MODELS in environment.env if models override provided
    if models:
        env_file = output_dir / "environment.env"
        if env_file.exists():
            models_str = ",".join([str(x) for x in models])
            with open(env_file) as f:
                lines = f.readlines()
            with open(env_file, "w") as f:
                for line in lines:
                    if line.startswith("QAIHM_TEST_MODELS="):
                        f.write(f"QAIHM_TEST_MODELS={models_str}\n")
                    else:
                        f.write(line)
            print("Updated QAIHM_TEST_MODELS in environment.env")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine split scorecard artifacts into a single directory"
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        required=True,
        help="Parent directory containing split artifact subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        required=True,
        help="Output directory for combined artifacts",
    )
    EnabledModelsEnvvar.add_arg(parser)

    args = parser.parse_args()

    if not args.input_dir.exists():
        print(
            f"Input directory does not exist: {args.input_dir}. "
            "Nothing to combine — exiting cleanly."
        )
        write_empty_release_assets(args.output_dir)
        return

    input_dirs = find_input_dirs(args.input_dir)
    print(f"Found {len(input_dirs)} input directories")
    combine_split_artifacts(input_dirs, args.output_dir, args.models)
    if not (args.output_dir / "release-assets.yaml").exists():
        write_empty_release_assets(args.output_dir)
    print(f"\nCombined artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
