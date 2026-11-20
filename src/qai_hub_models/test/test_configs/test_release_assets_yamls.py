# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Tests that validate release-assets.yaml files.

This includes validation that S3 keys referenced in release-assets.yaml
files actually exist, which helps catch issues before release.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.devices_and_chipsets_yaml import DevicesAndChipsetsYaml
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import (
    QAIHMModelPerf,
)
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.test.test_configs.manifest_path_check import (
    get_release_manifest_support_violations,
)
from qai_hub_models.utils.aws import (
    QAIHM_PRIVATE_S3_BUCKET,
    can_access_private_s3,
    get_qaihm_s3,
    s3_file_exists,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_MODELS_ROOT

# S3 keys with this prefix are created by automated CI and assumed to exist
AUTOMATED_S3_KEY_PREFIX = "pre_release_assets/gh_actions/"

# Runtimes whose perf numbers come from a release asset rather than a Hub job.
LLM_PROFILE_PATHS = (
    ScorecardProfilePath.GENIE,
    ScorecardProfilePath.GENIEX_QAIRT,
    ScorecardProfilePath.GENIEX_LLAMACPP,
)


def get_all_s3_keys_from_release_assets(
    model_id: str,
) -> list[tuple[str, str]]:
    """
    Extract all S3 keys from a model's release-assets.yaml.

    Returns a list of (description, s3_key) tuples for error reporting.
    """
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    if assets.empty:
        return []

    s3_keys: list[tuple[str, str]] = []
    for precision, precision_details in assets.precisions.items():
        # Universal assets
        for path, asset_details in precision_details.universal_assets.items():
            if asset_details.s3_key:
                desc = f"{model_id}/{precision}/{path}"
                s3_keys.append((desc, asset_details.s3_key))

        # Chipset-specific assets
        for chipset, chipset_paths in precision_details.chipset_assets.items():
            for path, asset_details in chipset_paths.items():
                if asset_details.s3_key:
                    desc = f"{model_id}/{precision}/{chipset}/{path}"
                    s3_keys.append((desc, asset_details.s3_key))

    return s3_keys


def get_models_with_release_assets() -> list[str]:
    """Get list of model IDs that have release-assets.yaml files."""
    return [
        model_id
        for model_id in MODEL_IDS
        if (QAIHM_MODELS_ROOT / model_id / "release-assets.yaml").exists()
    ]


def test_release_assets_yaml_schema() -> None:
    """Validate that all release-assets.yaml files conform to the schema."""
    chipsets = DevicesAndChipsetsYaml.load().chipsets.keys()
    path_errors: list[str] = []
    for model_id in get_models_with_release_assets():
        try:
            yaml = QAIHMModelReleaseAssets.from_model(model_id)
            manifest = QAIHMModelManifest.from_model(model_id)
            for precision, a in yaml.precisions.items():
                for c in a.chipset_assets:
                    if c not in chipsets:
                        raise ValueError(  # noqa: TRY301
                            f"Chipset {c} is not available in devices-and-chipsets.yaml; thus it can't be referenced here."
                        )

                paths = set(a.universal_assets)
                for chipset_paths in a.chipset_assets.values():
                    paths.update(chipset_paths)
                path_errors.extend(
                    get_release_manifest_support_violations(
                        manifest, "release-assets.yaml", precision, paths
                    )
                )
        except Exception as err:  # noqa: PERF203
            raise AssertionError(
                f"{model_id} release-assets.yaml validation failed: {err!s}"
            ) from err

    assert not path_errors, (
        f"{len(path_errors)} release-assets.yaml entries conflict with manifest.yaml:\n"
        + "\n".join(path_errors)
    )


def test_llm_perf_rows_have_a_release_asset() -> None:
    """Every LLM runtime with perf numbers must still have the asset behind them.

    A missing asset goes unnoticed in the committed data: geniex discovery drops
    an asset-less model from the matrix (discover_llamacpp_models), while genie
    stays in the matrix and only fails once a run reaches
    fetch_genie_bundle_for_perf. Either way perf.yaml keeps publishing numbers
    with nothing behind them. Granularity is (precision, runtime), not per
    chipset -- apply_similar_devices republishes one measurement under sibling
    device names that never had their own asset.
    """
    violations: list[str] = []
    for model_id in MODEL_IDS:
        perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        if perf.empty:
            continue

        measured: set[tuple[Precision, ScorecardProfilePath]] = set()
        for params, entry in perf.entries.items():
            if params.path in LLM_PROFILE_PATHS and entry.llm_metrics:
                assert params.precision is not None
                measured.add((params.precision, params.path))
        if not measured:
            continue

        assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
        available: set[tuple[Precision, ScorecardProfilePath]] = set()
        for precision, asset_details in assets.precisions.items():
            for path in asset_details.universal_assets:
                available.add((precision, path))
            for chipset_paths in asset_details.chipset_assets.values():
                for path in chipset_paths:
                    available.add((precision, path))

        for precision, path in sorted(measured, key=lambda k: (str(k[0]), k[1].name)):
            if (precision, path) in available:
                continue
            violations.append(
                f"  {model_id}: perf.yaml has {path.name} rows at {precision}, "
                f"but release-assets.yaml has no {path.name} asset at {precision}"
            )

    if violations:
        pytest.fail(
            f"Found {len(violations)} LLM perf entries with no backing release "
            "asset. Recover the asset from git history rather than guessing a "
            "replacement:\n" + "\n".join(violations)
        )


@pytest.mark.skipif(
    not can_access_private_s3(),
    reason="Requires AWS credentials to access private S3 bucket",
)
def test_release_assets_s3_keys_exist() -> None:
    """
    Validate that all S3 keys in release-assets.yaml files exist.

    This test iterates through all models with release-assets.yaml
    and verifies that every referenced S3 key exists in the bucket.
    Collects all errors to report them at once rather than failing
    on the first missing key.

    Optimizations:
    - Skips automated keys (gh_actions/) which are assumed to exist
    - Uses threading to parallelize S3 HEAD requests
    """
    bucket, _ = get_qaihm_s3(QAIHM_PRIVATE_S3_BUCKET)

    # Collect all keys to check (excluding automated ones)
    keys_to_check: list[tuple[str, str]] = []
    for model_id in get_models_with_release_assets():
        for desc, s3_key in get_all_s3_keys_from_release_assets(model_id):
            # Skip automated keys - they are created by CI and assumed to exist
            if not s3_key.startswith(AUTOMATED_S3_KEY_PREFIX):
                keys_to_check.append((desc, s3_key))

    if not keys_to_check:
        # No manual keys to check
        return

    missing_keys_by_model: dict[str, list[str]] = {}

    def check_key(desc: str, s3_key: str) -> tuple[str, str] | None:
        if not s3_file_exists(bucket, s3_key):
            # Extract model_id from desc (format: model_id/precision/...)
            model_id, rest = desc.split("/", maxsplit=1)
            return (model_id, f"  - {rest}: {s3_key}")
        return None

    # Run checks in parallel with limited thread count
    with ThreadPoolExecutor(max_workers=min(64, len(keys_to_check))) as pool:
        futures = [
            pool.submit(check_key, desc, s3_key) for desc, s3_key in keys_to_check
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                model_id, msg = result
                if model_id not in missing_keys_by_model:
                    missing_keys_by_model[model_id] = []
                missing_keys_by_model[model_id].append(msg)

    if missing_keys_by_model:
        total_missing = sum(len(v) for v in missing_keys_by_model.values())
        lines = [
            f"Found {total_missing} missing S3 keys in release-assets.yaml files:\n"
        ]
        for model_id, missing_keys in sorted(missing_keys_by_model.items()):
            lines.append(f"{model_id}:")
            lines.extend(missing_keys)
            lines.append("")
        pytest.fail("\n".join(lines))
