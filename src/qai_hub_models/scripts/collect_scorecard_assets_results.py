# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

from qai_hub_models import Precision
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledDevicesEnvvar,
    EnabledModelsEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    SpecialModelSetting,
)
from qai_hub_models.scorecard.internal_config_yaml import QAIHMModelInternalConfig
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.code_gen import (
    remove_asset_failures,
    update_model_publish_status,
)
from qai_hub_models.scorecard.results.scorecard_summary import ModelTestConfig
from qai_hub_models.scorecard.results.yaml import (
    ComponentNamesYaml,
    GraphNamesYaml,
    ScorecardAssetYaml,
    get_model_component_and_graph_names,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.hub_clients import get_default_hub_deployment


def _release_assets_scope(
    manifest: QAIHMModelManifest,
    component_names_yaml: ComponentNamesYaml,
    graph_names_yaml: GraphNamesYaml,
) -> set[tuple[Precision, str | None, ScorecardProfilePath]]:
    """The (precision, chipset, path) tuples this run was configured to produce.

    chipset is the target chipset for AOT-compiled runtimes and None otherwise.
    """
    if manifest.id is None:
        raise ValueError(
            "Cannot compute a release-assets scope for a manifest with no id."
        )
    component_names, graph_names, component_graph_names = (
        get_model_component_and_graph_names(
            manifest.id, component_names_yaml, graph_names_yaml
        )
    )
    test_params = ModelTestConfig.from_recipe_model(
        manifest, component_names, graph_names, component_graph_names
    )
    scope: set[tuple[Precision, str | None, ScorecardProfilePath]] = set()
    for precision, path, device in test_params.profile_tests:
        chipset = device.chipset if path.runtime.is_aot_compiled else None
        scope.add((precision, chipset, path))
    return scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.PYTORCH})
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    EnabledPrecisionsEnvvar.add_arg(parser)
    EnabledPathsEnvvar.add_arg(parser)
    EnabledDevicesEnvvar.add_arg(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--release-assets-yaml",
        type=str,
        default=str(ScorecardArtifact.RELEASE_ASSETS.path),
    )
    return parser.parse_args()


def main() -> None:
    # Verify args are compatible with the chosen deployment.
    args = parse_args()
    pytorch_models, _ = validate_and_split_enabled_models(args.models)

    # An absent file and a present-but-empty `models: {}` both mean "this run
    # produced no assets"; dropping in-scope entries would delete committed ones.
    assets_path = Path(args.release_assets_yaml)
    scorecard_assets = (
        ScorecardAssetYaml.from_yaml(assets_path)
        if assets_path.exists() and assets_path.stat().st_size > 0
        else None
    )
    if scorecard_assets is None or not scorecard_assets.models:
        print("No scorecard release assets found. Not updating any files.")
        return

    modified_files: list[str] = []
    component_names_yaml = ComponentNamesYaml.from_intermediates()
    graph_names_yaml = GraphNamesYaml.from_intermediates()
    for model_id in sorted(pytorch_models):
        try:
            manifest = QAIHMModelManifest.from_model(model_id)
            sc = QAIHMModelInternalConfig.from_model(model_id)
            if (
                sc.skip_hub_tests_and_scorecard
                or sc.skip_scorecard
                or sc.freeze_perf_yaml
            ):
                continue

            # Scoped merge: drop in-scope (precision, chipset, path) tuples
            # from the committed yaml, then fold in this run's results.
            scope = _release_assets_scope(
                manifest, component_names_yaml, graph_names_yaml
            )
            merged = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
            merged.drop_entries_in_scope(scope)

            scorecard_model_assets = scorecard_assets.models.get(model_id)
            if scorecard_model_assets is not None:
                scorecard_model_assets = remove_asset_failures(
                    scorecard_model_assets, manifest.disabled_paths
                )
                if scorecard_model_assets.has_ephemeral_s3_keys:
                    print(
                        f"Skipping {model_id} release-assets.yaml: assets were "
                        "uploaded to ephemeral_test_assets/ (auto-purged)."
                    )
                    continue
                merged.merge_from(scorecard_model_assets)

            # Committed entries outside this run's scope (e.g. Genie, which the default
            # sweep skips) can still be disabled, so strip failures from the merged result.
            merged = remove_asset_failures(merged, manifest.disabled_paths)
            modified_files.append(str(merged.to_model_yaml(model_id)))

            # Update model status & reason, if applicable
            if config_path := update_model_publish_status(manifest):
                modified_files.append(str(config_path))
                print(f"Updated publish status at {config_path}")

        except Exception as e:
            raise ValueError(
                f"Failed to collect accuracy results for {model_id}"
            ) from e


if __name__ == "__main__":
    main()
